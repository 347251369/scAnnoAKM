import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np


def token_marker(gene_info, cell_knowledge, marker_len):
    gene2id = gene_info['gene2id']
    num_classes = len(cell_knowledge)
    marker_lists = [cell['cell_marker'] for cell in cell_knowledge]
    marker_ids = torch.full((num_classes, marker_len), fill_value=-1, dtype=torch.long)
    for i, markers in enumerate(marker_lists):
        ids = [gene2id.get(m, -1) for m in markers]
        marker_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return marker_ids


def token_emb_text_knowledge(model, cell_knowledge):
    tissues = [cell['cell_tissue'] for cell in cell_knowledge]
    roles   = [cell['cell_role'] for cell in cell_knowledge]
    tissue_toks = model.text_tokenizer(tissues)
    role_toks   = model.text_tokenizer(roles)
    with torch.no_grad():
        cell_tissue_emb = model.bio_emb(tissue_toks)
        cell_role_emb   = model.bio_emb(role_toks)
    return cell_tissue_emb, cell_role_emb


def get_gene_id(input_seqs, gene_info, gene_size):

    batch_ids = []
    for seq in input_seqs:
        ids = [gene_info['gene2id'].get(g, -1) for g in seq]
        truncated_ids = ids[:gene_size]
        while len(truncated_ids) < gene_size:
            truncated_ids.append(-1)
        batch_ids.append(truncated_ids)
    return torch.tensor(batch_ids, dtype=torch.long)


def position_weights(L, device, valid=None, decay=0.01):
    k = torch.arange(L, device=device, dtype=torch.float32)
    w = torch.exp(-decay * k)
    if valid is None:
        return w / w.sum()
    w = w[None, :] * valid.float()
    return w / w.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def compute_discriminative(scores, eps=1e-6):
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True)
    z = (scores - mean) / (std + eps)
    return torch.clamp(z, min=-10.0, max=10.0)


def marker_membership_evidence(gene_ids, marker_ids, pos_w, gene_count):
    batch, classes, _ = marker_ids.shape
    member = torch.zeros((batch, classes, int(gene_count) + 1), dtype=torch.bool, device=gene_ids.device)
    member.scatter_(2, marker_ids + 1, True)
    member[:, :, 0] = False
    hit = member.gather(2, (gene_ids + 1)[:, None, :].expand(-1, classes, -1))
    return (hit.to(pos_w.dtype) * pos_w[:, None, :]).sum(dim=-1)


def calibrate_marker_evidence(marker_disc, evidence, weight=0.5):
    effective = (evidence.std(dim=-1, keepdim=True) > 1e-6).to(marker_disc.dtype) * weight
    return (1 - effective) * marker_disc + effective * compute_discriminative(evidence)


def row_zscore(scores, eps=1e-8):
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True)
    return (scores - mean) / (std + eps)


def fuse_scores(coarse_logits, fine_score, reference_mask, seen_bias=1.0, transcriptome_weight=0.25):
    if coarse_logits.shape != fine_score.shape or fine_score.ndim != 2:
        raise ValueError("Scores must have matching [cells, classes] shapes")
    mask = torch.as_tensor(reference_mask, dtype=torch.bool, device=fine_score.device)
    if mask.ndim != 1 or mask.numel() != fine_score.shape[1]:
        raise ValueError("reference_mask must align with the candidate class order")
    if not 0 <= transcriptome_weight <= 1:
        raise ValueError("transcriptome_weight must be between zero and one")
    knowledge = torch.softmax(fine_score - seen_bias * mask[None, :], dim=1)
    if not mask.any():
        return knowledge
    transcriptome = torch.softmax(coarse_logits[:, mask], dim=1)
    weight = transcriptome_weight * mask[knowledge.argmax(dim=1), None]
    result = knowledge.clone()
    mass = knowledge[:, mask].sum(dim=1, keepdim=True)
    result[:, mask] = (1 - weight) * knowledge[:, mask] + weight * mass * transcriptome
    return result


def _safe_metric_value(value):
    return "nan" if np.isnan(value) else f"{value:.6f}"


def _compute_split_metrics(true_labels, preds, mask):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {
            "cells": 0,
            "correct": 0,
            "accuracy": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    true_arr = np.asarray(true_labels, dtype=object)[mask]
    pred_arr = np.asarray(preds, dtype=object)[mask]
    correct = int(np.sum(true_arr == pred_arr))
    _, recall, f1, _ = precision_recall_fscore_support(
        true_arr, pred_arr, average="weighted", zero_division=0
    )
    return {
        "cells": n,
        "correct": correct,
        "accuracy": float(correct / n),
        "recall": float(recall),
        "f1": float(f1),
    }


def _print_seen_unseen_metrics(split_metrics):
    print("\n" + "=" * 80)
    print("  SEEN / UNSEEN METRICS")
    print("=" * 80)
    print(f"  {'Group':<10s} {'Cells':>8s} {'Correct':>8s} {'Accuracy':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 80)
    for name in ["all", "seen", "unseen"]:
        m = split_metrics[name]
        print(
            f"  {name:<10s} {m['cells']:>8d} {m['correct']:>8d} "
            f"{_safe_metric_value(m['accuracy']):>10s} "
            f"{_safe_metric_value(m['recall']):>10s} "
            f"{_safe_metric_value(m['f1']):>10s}"
        )
    print("=" * 80)


def predict_model(model, cell_knowledge, gene_info, tar_seqs, tar_labels, config, known_labels=None):
    model.eval()

    label_list = [cell['cell_type'] for cell in cell_knowledge]
    true_labels = [str(l) for l in tar_labels]
    marker_evidence_weight = float(getattr(config, "marker_evidence_weight", 0.5))
    seen_bias = float(getattr(config, "seen_bias", 1.1))
    transcriptome_weight = float(getattr(config, "transcriptome_weight", 0.25))
    marker_beta = float(getattr(config, "marker_position_beta", 0.03))
    text_beta = float(getattr(config, "tissue_role_position_beta", 0.01))
    if not 0 <= marker_evidence_weight <= 1:
        raise ValueError("marker_evidence_weight must be between zero and one")
    if known_labels is None:
        raise ValueError("Calibrated fusion requires reference labels")
    known_label_set = set(str(x) for x in known_labels)
    if not known_label_set or not known_label_set.issubset(label_list):
        raise ValueError("Reference labels must be non-empty and present in cell knowledge")
    reference_mask = torch.tensor([label in known_label_set for label in label_list], device=model.device)

    preds = []
    coarse_preds = []
    fine_preds = []
    all_final_probabilities = []

    gene_markers = token_marker(gene_info, cell_knowledge, config.marker_len)
    tissu_emb, role_emb = token_emb_text_knowledge(model, cell_knowledge)


    all_fine_scores = []
    all_coarse_logits = []
    all_mk_scores = []
    all_mk_fused_scores = []
    all_marker_evidence = []
    all_tk_scores = []
    all_rk_scores = []

    with torch.no_grad():
        pbar = tqdm(range(0, len(tar_seqs), config.predict_batch_size), desc="[predict]")
        for i in pbar:
            batch_seqs = tar_seqs[i:i+config.predict_batch_size]
            B = len(batch_seqs)

            attention_mask = None
            batch_length = config.seq_len
            if getattr(config, "dynamic_length", False):
                batch_seqs, batch_tokens, attention_mask, batch_length = model.dynamic_gene_batch(batch_seqs, config.seq_len)
            else:
                batch_tokens = []
                for seq in batch_seqs:
                    tok = model.seq_tokenizer(seq).unsqueeze(0).to(model.device)
                    batch_tokens.append(tok)
                batch_tokens = torch.cat(batch_tokens, dim=0)
            batch_geneids = get_gene_id(batch_seqs, gene_info, batch_length).to(model.device)
            valid = batch_geneids != -1
            pos_w = position_weights(batch_length, model.device, valid, text_beta)[:, None, :]
            marker_pos_w = position_weights(batch_length, model.device, valid, marker_beta)[:, None, :]

            if getattr(config, "dynamic_marker_length", False):
                cur_markers = model.dynamic_marker_batch(batch_seqs, gene_markers).to(model.device)
            else:
                cur_markers = gene_markers.unsqueeze(0).expand(B, -1, -1).to(model.device)
            cur_tissue = tissu_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            cur_role = role_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)

            coarse_logits, marker_view, tissue_view, role_view = model(
                batch_tokens, batch_geneids, cur_markers, cur_tissue, cur_role, attention_mask=attention_mask
            )


            mk_raw = (marker_view.masked_fill(~valid[:, None, :], 0) * marker_pos_w).sum(dim=-1)
            tk_raw = (tissue_view.masked_fill(~valid[:, None, :], 0) * pos_w).sum(dim=-1)
            rk_raw = (role_view.masked_fill(~valid[:, None, :], 0) * pos_w).sum(dim=-1)


            mk_disc = compute_discriminative(mk_raw)
            tk_disc = compute_discriminative(tk_raw)
            rk_disc = compute_discriminative(rk_raw)
            marker_evidence = marker_membership_evidence(batch_geneids, cur_markers, marker_pos_w[:, 0, :], gene_info["num"])
            mk_fused = calibrate_marker_evidence(mk_disc, marker_evidence, marker_evidence_weight)


            fixed_view_weights = getattr(config, "fixed_view_weights", None)
            if fixed_view_weights is None:
                w = F.softmax(model.view_weights, dim=0)
            else:
                w = torch.tensor(fixed_view_weights, dtype=torch.float32, device=model.device)
                w = w / w.sum()
            fine_score = w[0] * mk_fused + w[1] * tk_disc + w[2] * rk_disc

            final_probabilities = fuse_scores(
                coarse_logits, fine_score, reference_mask,
                seen_bias=seen_bias, transcriptome_weight=transcriptome_weight,
            )
            final_idx = final_probabilities.argmax(dim=-1)
            coarse_idx = row_zscore(coarse_logits).argmax(dim=-1)
            fine_idx = row_zscore(fine_score).argmax(dim=-1)
            all_final_probabilities.append(final_probabilities.detach().cpu())

            all_fine_scores.append(fine_score.detach().cpu())
            all_coarse_logits.append(coarse_logits.detach().cpu())
            all_mk_scores.append(mk_disc.detach().cpu())
            all_mk_fused_scores.append(mk_fused.detach().cpu())
            all_marker_evidence.append(marker_evidence.detach().cpu())
            all_tk_scores.append(tk_disc.detach().cpu())
            all_rk_scores.append(rk_disc.detach().cpu())

            preds.extend(label_list[j] for j in final_idx.cpu().tolist())
            coarse_preds.extend(label_list[j] for j in coarse_idx.cpu().tolist())
            fine_preds.extend(label_list[j] for j in fine_idx.cpu().tolist())

    total = len(preds)
    correct = sum(p == t for p, t in zip(preds, true_labels))
    coarse_correct = sum(p == t for p, t in zip(coarse_preds, true_labels))
    consistent = np.asarray(coarse_preds) == np.asarray(fine_preds)
    correct_mask = np.asarray(preds) == np.asarray(true_labels)
    consistent_cnt = int(consistent.sum())
    consist_correct = int((consistent & correct_mask).sum())
    inconsist_correct = int((~consistent & correct_mask).sum())

    seen_mask = np.array([t in known_label_set for t in true_labels], dtype=bool)
    all_mask = np.ones(len(true_labels), dtype=bool)
    split_metrics = {
        "all": _compute_split_metrics(true_labels, preds, all_mask),
        "seen": _compute_split_metrics(true_labels, preds, seen_mask),
        "unseen": _compute_split_metrics(true_labels, preds, ~seen_mask),
    }


    label_to_id = {lbl: i for i, lbl in enumerate(label_list)}
    true_ids = [label_to_id[t] for t in true_labels]
    pred_ids = [label_to_id[p] for p in preds]

    _, recall, f1, _ = precision_recall_fscore_support(
        true_ids, pred_ids, average='weighted', zero_division=0
    )
    n_classes = len(label_list)
    y_true_onehot = np.eye(n_classes)[true_ids]
    y_score = torch.cat(all_fine_scores, dim=0).numpy()
    coarse_score_array = torch.cat(all_coarse_logits, dim=0).numpy()
    mk_score_array = torch.cat(all_mk_scores, dim=0).numpy()
    mk_fused_score_array = torch.cat(all_mk_fused_scores, dim=0).numpy()
    marker_evidence_array = torch.cat(all_marker_evidence, dim=0).numpy()
    tk_score_array = torch.cat(all_tk_scores, dim=0).numpy()
    rk_score_array = torch.cat(all_rk_scores, dim=0).numpy()
    knowledge_auprc = average_precision_score(y_true_onehot, y_score, average='weighted')
    final_probability_array = torch.cat(all_final_probabilities, dim=0).numpy()
    auprc = average_precision_score(y_true_onehot, final_probability_array, average='weighted')

    score_dump_addr = getattr(config, "score_dump_addr", None)
    if score_dump_addr:
        import os
        os.makedirs(os.path.dirname(score_dump_addr), exist_ok=True)
        np.savez_compressed(
            score_dump_addr,
            coarse_logits=coarse_score_array,
            fine_score=y_score,
            final_probabilities=final_probability_array,
            predictions=np.asarray(preds, dtype=object),
            reference_class_mask=reference_mask.cpu().numpy(),
            calibrated_fusion=np.asarray(True),
            mk_disc=mk_score_array,
            mk_fused=mk_fused_score_array,
            marker_evidence=marker_evidence_array,
            marker_evidence_weight=np.asarray([marker_evidence_weight], dtype=np.float32),
            seen_bias=np.asarray([seen_bias], dtype=np.float32),
            transcriptome_weight=np.asarray([transcriptome_weight], dtype=np.float32),
            tk_disc=tk_score_array,
            rk_disc=rk_score_array,
            true_labels=np.asarray(true_labels, dtype=object),
            label_list=np.asarray(label_list, dtype=object),
            seen_mask=seen_mask,
            fixed_view_weights=np.asarray([w[0].item(), w[1].item(), w[2].item()], dtype=np.float32),
        )
        print(f"Saved prediction score dump: {score_dump_addr}")


    print("\n" + "=" * 80)
    print(f"  AGGREGATE METRICS")
    print(f"  Accuracy:  {correct}/{total} = {correct/total:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1:        {f1:.6f}")
    print(f"  AUPRC:     {auprc:.6f}")
    print(f"  Coarse Acc:{coarse_correct}/{total} = {coarse_correct/total:.6f}")
    print(f"  Consistency: {consistent_cnt}/{total} = {consistent_cnt/total:.6f}")
    print(f"  Consist-correct: {consist_correct}, Inconsist-correct: {inconsist_correct}")
    print(f"  View weights: mk={w[0].item():.3f} tk={w[1].item():.3f} rk={w[2].item():.3f}")
    print("=" * 80)

    _print_seen_unseen_metrics(split_metrics)

    return {
        "predictions": preds,
        "coarse_predictions": coarse_preds,
        "details": [],
        "true_labels": true_labels,
        "label_list": label_list,
        "metrics": {
            "accuracy": correct / total if total > 0 else 0.0,
            "coarse_accuracy": coarse_correct / total if total > 0 else 0.0,
            "consistency_rate": consistent_cnt / total if total > 0 else 0.0,
            "consist_correct": consist_correct,
            "inconsist_correct": inconsist_correct,
            "recall": float(recall),
            "f1": float(f1),
            "auprc": float(auprc),
            "knowledge_auprc": float(knowledge_auprc),
            "all_accuracy": split_metrics["all"]["accuracy"],
            "all_recall": split_metrics["all"]["recall"],
            "all_f1": split_metrics["all"]["f1"],
            "seen_accuracy": split_metrics["seen"]["accuracy"],
            "seen_recall": split_metrics["seen"]["recall"],
            "seen_f1": split_metrics["seen"]["f1"],
            "unseen_accuracy": split_metrics["unseen"]["accuracy"],
            "unseen_recall": split_metrics["unseen"]["recall"],
            "unseen_f1": split_metrics["unseen"]["f1"],
            "split_metrics": split_metrics
        }
    }
