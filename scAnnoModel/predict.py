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
    """Preserve original gene order; mark genes not in gene_info as -1."""
    batch_ids = []
    for seq in input_seqs:
        ids = [gene_info['gene2id'].get(g, -1) for g in seq]
        truncated_ids = ids[:gene_size]
        while len(truncated_ids) < gene_size:
            truncated_ids.append(-1)
        batch_ids.append(truncated_ids)
    return torch.tensor(batch_ids, dtype=torch.long)


def position_weights(L, device):
    k = torch.arange(L, device=device, dtype=torch.float32)
    w = torch.exp(-0.01 * k)
    return w / w.sum()


def compute_discriminative(scores, eps=1e-6):
    """Z-score normalization: how many std above/below mean per class.
    Eliminates absolute-magnitude bias across views."""
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True)
    z = (scores - mean) / (std + eps)
    return torch.clamp(z, min=-10.0, max=10.0)


def row_zscore(scores, eps=1e-8):
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True)
    return (scores - mean) / (std + eps)


def top2_margin(scores):
    top2 = torch.topk(scores, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def normalized_entropy(probs, eps=1e-8):
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
    return entropy / torch.log(torch.tensor(probs.shape[1], dtype=probs.dtype, device=probs.device) + eps)


def transcriptome_knowledge_router(coarse_logits, fine_score, config):
    coarse_z = row_zscore(coarse_logits)
    fine_z = row_zscore(fine_score)

    coarse_idx = coarse_z.argmax(dim=-1)
    fine_idx = fine_z.argmax(dim=-1)

    coarse_prob = F.softmax(coarse_logits, dim=-1).max(dim=-1).values
    coarse_margin = top2_margin(coarse_z)
    coarse_entropy = normalized_entropy(F.softmax(coarse_logits, dim=-1))

    prob_threshold = getattr(config, "router_prob", 0.25)
    margin_threshold = getattr(config, "router_margin", 2.5)
    entropy_threshold = getattr(config, "router_entropy", 0.25)

    use_coarse = (
        (coarse_prob >= prob_threshold)
        & (coarse_margin >= margin_threshold)
        & (coarse_entropy <= entropy_threshold)
    )
    final_idx = fine_idx.clone()
    final_idx[use_coarse] = coarse_idx[use_coarse]

    return {
        "coarse_z": coarse_z,
        "fine_z": fine_z,
        "coarse_idx": coarse_idx,
        "fine_idx": fine_idx,
        "final_idx": final_idx,
        "use_coarse": use_coarse,
        "coarse_prob": coarse_prob,
        "coarse_margin": coarse_margin,
        "coarse_entropy": coarse_entropy,
    }




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


def _print_per_class_results(label_list, true_labels, preds, details):
    """Print per-class accuracy, recall, F1, and average per-view scores."""
    n_classes = len(label_list)

    # Build per-class stats
    class_true_counts = {lbl: 0 for lbl in label_list}
    class_correct_counts = {lbl: 0 for lbl in label_list}
    class_tp_counts = {lbl: 0 for lbl in label_list}
    class_fp_counts = {lbl: 0 for lbl in label_list}
    class_fn_counts = {lbl: 0 for lbl in label_list}
    # Per-view score accumulators per class
    class_mk_sum = {lbl: 0.0 for lbl in label_list}
    class_tk_sum = {lbl: 0.0 for lbl in label_list}
    class_rk_sum = {lbl: 0.0 for lbl in label_list}
    class_fine_sum = {lbl: 0.0 for lbl in label_list}

    for true_lbl, pred_lbl, d in zip(true_labels, preds, details):
        class_true_counts[true_lbl] += 1
        if true_lbl == pred_lbl:
            class_correct_counts[true_lbl] += 1
        # TP
        if true_lbl == pred_lbl:
            class_tp_counts[true_lbl] += 1
        # FN
        if true_lbl != pred_lbl:
            class_fn_counts[true_lbl] += 1
        # FP
        if true_lbl != pred_lbl:
            class_fp_counts[pred_lbl] += 1

        # Accumulate per-view scores for the TRUE class
        true_idx = label_list.index(true_lbl)
        class_mk_sum[true_lbl] += d['scores']['mk_disc'][true_idx]
        class_tk_sum[true_lbl] += d['scores']['tk_disc'][true_idx]
        class_rk_sum[true_lbl] += d['scores']['rk_disc'][true_idx]
        class_fine_sum[true_lbl] += d['scores']['fine'][true_idx]

    print("\n" + "=" * 120)
    print("  PER-CLASS DETAILED RESULTS")
    print("=" * 120)
    print(f"  {'Class':<30s} {'N':>5s} {'Corr':>5s} {'Acc':>8s} {'Recall':>8s} {'F1':>8s} {'mk_disc':>8s} {'tk_disc':>8s} {'rk_disc':>8s} {'fine':>8s}")
    print("-" * 120)

    for lbl in label_list:
        N = class_true_counts[lbl]
        corr = class_correct_counts[lbl]
        tp = class_tp_counts[lbl]
        fp = class_fp_counts[lbl]
        fn = class_fn_counts[lbl]
        acc = corr / N if N > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_val = 2 * precision_val * recall / (precision_val + recall) if (precision_val + recall) > 0 else 0.0
        mk_avg = class_mk_sum[lbl] / N if N > 0 else 0.0
        tk_avg = class_tk_sum[lbl] / N if N > 0 else 0.0
        rk_avg = class_rk_sum[lbl] / N if N > 0 else 0.0
        fine_avg = class_fine_sum[lbl] / N if N > 0 else 0.0
        print(f"  {lbl:<30s} {N:>5d} {corr:>5d} {acc:>8.4f} {recall:>8.4f} {f1_val:>8.4f} {mk_avg:>8.3f} {tk_avg:>8.3f} {rk_avg:>8.3f} {fine_avg:>8.3f}")
    print("-" * 120)


def _print_misclassifications(label_list, true_labels, preds, details, max_show=30):
    """Print misclassified samples with per-view scores for both true and predicted class."""
    misclass = [(i, true_labels[i], preds[i], details[i])
                for i in range(len(true_labels))
                if true_labels[i] != preds[i]]

    print("\n" + "=" * 120)
    print(f"  MISCLASSIFICATION DETAILS ({len(misclass)} total errors, showing first {min(len(misclass), max_show)})")
    print("=" * 120)

    shown = 0
    for idx, true_lbl, pred_lbl, d in misclass:
        if shown >= max_show:
            print(f"  ... and {len(misclass) - max_show} more errors (omitted)")
            break

        true_idx_val = label_list.index(true_lbl)
        pred_idx_val = label_list.index(pred_lbl)

        # Per-view scores for true and predicted class
        mk_t = d['scores']['mk_disc'][true_idx_val]
        tk_t = d['scores']['tk_disc'][true_idx_val]
        rk_t = d['scores']['rk_disc'][true_idx_val]
        fine_t = d['scores']['fine'][true_idx_val]

        mk_p = d['scores']['mk_disc'][pred_idx_val]
        tk_p = d['scores']['tk_disc'][pred_idx_val]
        rk_p = d['scores']['rk_disc'][pred_idx_val]
        fine_p = d['scores']['fine'][pred_idx_val]

        # Top-3 fine_score indices for context
        fine_scores = np.array(d['scores']['fine'])
        top3_idx = np.argsort(fine_scores)[::-1][:3]
        top3_str = ", ".join(f"{label_list[i]}[{fine_scores[i]:.3f}]" for i in top3_idx)

        print(f"  [{shown+1}] #{idx}  TRUE={true_lbl}  -->  PRED={pred_lbl}")
        print(f"      True scores:  mk={mk_t:+.3f}  tk={tk_t:+.3f}  rk={rk_t:+.3f}  fine={fine_t:+.3f}")
        print(f"      Pred scores:  mk={mk_p:+.3f}  tk={tk_p:+.3f}  rk={rk_p:+.3f}  fine={fine_p:+.3f}")
        # coarse_logits per class
        cl = d['coarse_logits']
        cl_str = ", ".join(f"{label_list[i]}[{cl[i]:+.3f}]" for i in range(len(label_list)))

        print(f"      gap(fine): {fine_p - fine_t:+.4f}  |  Top-3: {top3_str}")
        print(f"      coarse_logits: {cl_str}")
        print(f"      use_coarse={d['use_coarse']}, fine_adv={d['fine_adv']:+.4f}, coarse={d['coarse']}, fine={d['fine']}")
        shown += 1

    print("=" * 120)


def _print_confusion_summary(label_list, true_labels, preds, details, top_k=3):
    """Print confusion summary: for each class, show what it gets confused with."""
    confusion = {lbl: {} for lbl in label_list}

    for true_lbl, pred_lbl in zip(true_labels, preds):
        if true_lbl != pred_lbl:
            confusion[true_lbl][pred_lbl] = confusion[true_lbl].get(pred_lbl, 0) + 1

    print("\n" + "=" * 120)
    print("  PER-CLASS CONFUSION SUMMARY (True -> Predicted)")
    print("=" * 120)
    for lbl in label_list:
        conf = confusion[lbl]
        if not conf:
            print(f"  {lbl:<30s} -> (no errors)")
        else:
            sorted_conf = sorted(conf.items(), key=lambda x: -x[1])
            top_items = sorted_conf[:top_k]
            conf_str = ", ".join(f"{t} ({c}x)" for t, c in top_items)
            total_err = sum(conf.values())
            print(f"  {lbl:<30s} -> [{total_err} errors] {conf_str}")
    print("=" * 120)


def predict_model(model, cell_knowledge, gene_info, tar_seqs, tar_labels, config, known_labels=None):
    model.eval()

    label_list = [cell['cell_type'] for cell in cell_knowledge]
    true_labels = [str(l) for l in tar_labels]
    num_classes = len(label_list)

    preds = []
    coarse_preds = []
    details = []

    gene_markers = token_marker(gene_info, cell_knowledge, config.marker_len)
    tissu_emb, role_emb = token_emb_text_knowledge(model, cell_knowledge)

    pos_w = position_weights(config.seq_len, model.device)

    all_fine_scores = []
    all_coarse_logits = []
    all_mk_scores = []
    all_tk_scores = []
    all_rk_scores = []

    with torch.no_grad():
        pbar = tqdm(range(0, len(tar_seqs), config.predict_batch_size), desc="[predict]")
        for i in pbar:
            batch_seqs = tar_seqs[i:i+config.predict_batch_size]
            batch_truth = true_labels[i:i+config.predict_batch_size]
            B = len(batch_seqs)

            batch_tokens = []
            for seq in batch_seqs:
                tok = model.seq_tokenizer(seq).unsqueeze(0).to(model.device)
                batch_tokens.append(tok)
            batch_tokens = torch.cat(batch_tokens, dim=0)
            batch_geneids = get_gene_id(batch_seqs, gene_info, config.seq_len).to(model.device)

            cur_markers = gene_markers.unsqueeze(0).expand(B, -1, -1).to(model.device)
            cur_tissue = tissu_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            cur_role = role_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)

            coarse_logits, marker_view, tissue_view, role_view = model(
                batch_tokens, batch_geneids, cur_markers, cur_tissue, cur_role
            )

            # Step 1: Position-weighted per-view aggregation
            mk_raw = (marker_view * pos_w).sum(dim=-1)
            tk_raw = (tissue_view * pos_w).sum(dim=-1)
            rk_raw = (role_view   * pos_w).sum(dim=-1)

            # Step 2: Z-score discriminative normalization
            mk_disc = compute_discriminative(mk_raw)
            tk_disc = compute_discriminative(tk_raw)
            rk_disc = compute_discriminative(rk_raw)

            # Step 3: View fusion
            fixed_view_weights = getattr(config, "fixed_view_weights", None)
            if fixed_view_weights is None:
                w = F.softmax(model.view_weights, dim=0)
            else:
                w = torch.tensor(fixed_view_weights, dtype=torch.float32, device=model.device)
                w = w / w.sum()
            fine_score = w[0] * mk_disc + w[1] * tk_disc + w[2] * rk_disc

            # Step 4: Transcriptome-knowledge confidence-aware routing
            router_out = transcriptome_knowledge_router(coarse_logits, fine_score, config)
            coarse_idx = router_out["coarse_idx"]
            fine_idx = router_out["fine_idx"]
            final_idx = router_out["final_idx"]
            use_coarse = router_out["use_coarse"]

            fine_z = router_out["fine_z"]
            coarse_z = router_out["coarse_z"]
            fine_top1 = fine_z[torch.arange(B), fine_idx]
            coarse_fs = fine_z[torch.arange(B), coarse_idx]
            fine_adv = fine_top1 - coarse_fs
            coarse_adv = coarse_z[torch.arange(B), coarse_idx] - coarse_z[torch.arange(B), fine_idx]
            consensus = (coarse_idx == fine_idx)

            all_fine_scores.append(fine_score.detach().cpu())
            all_coarse_logits.append(coarse_logits.detach().cpu())
            all_mk_scores.append(mk_disc.detach().cpu())
            all_tk_scores.append(tk_disc.detach().cpu())
            all_rk_scores.append(rk_disc.detach().cpu())

            for b in range(B):
                c_idx = coarse_idx[b].item()
                f_idx = fine_idx[b].item()
                final_i = final_idx[b].item()
                final_str = label_list[final_i]
                is_correct = (final_str == batch_truth[b])

                preds.append(final_str)
                coarse_preds.append(label_list[c_idx])

                details.append({
                    "true_label": batch_truth[b],
                    "coarse": label_list[c_idx],
                    "fine": label_list[f_idx],
                    "final": final_str,
                    "correct": is_correct,
                    "consistent": c_idx == f_idx,
                    "use_coarse": use_coarse[b].item(),
                    "fine_adv": fine_adv[b].item(),
                    "coarse_adv": coarse_adv[b].item(),
                    "coarse_prob": router_out["coarse_prob"][b].item(),
                    "coarse_margin": router_out["coarse_margin"][b].item(),
                    "coarse_entropy": router_out["coarse_entropy"][b].item(),
                    "coarse_logits": coarse_logits[b].cpu().tolist(),
                    "scores": {
                        "mk_raw": mk_raw[b].cpu().tolist(),
                        "tk_raw": tk_raw[b].cpu().tolist(),
                        "rk_raw": rk_raw[b].cpu().tolist(),
                        "mk_disc": mk_disc[b].cpu().tolist(),
                        "tk_disc": tk_disc[b].cpu().tolist(),
                        "rk_disc": rk_disc[b].cpu().tolist(),
                        "fine": fine_score[b].cpu().tolist(),
                    }
                })

    # ── Metrics ──
    total = len(preds)
    correct = sum(p == t for p, t in zip(preds, true_labels))
    coarse_correct = sum(p == t for p, t in zip(coarse_preds, true_labels))
    consistent_cnt = sum(d["consistent"] for d in details)
    consist_correct = sum(d["consistent"] and d["correct"] for d in details)
    inconsist_correct = sum(not d["consistent"] and d["correct"] for d in details)

    known_label_set = set(str(x) for x in known_labels) if known_labels is not None else set(label_list)
    seen_mask = np.array([t in known_label_set for t in true_labels], dtype=bool)
    all_mask = np.ones(len(true_labels), dtype=bool)
    split_metrics = {
        "all": _compute_split_metrics(true_labels, preds, all_mask),
        "seen": _compute_split_metrics(true_labels, preds, seen_mask),
        "unseen": _compute_split_metrics(true_labels, preds, ~seen_mask),
    }

    # ── Compute Recall, F1, AUPRC ──
    label_to_id = {lbl: i for i, lbl in enumerate(label_list)}
    true_ids = [label_to_id[t] for t in true_labels]
    pred_ids = [label_to_id[p] for p in preds]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_ids, pred_ids, average='weighted', zero_division=0
    )
    n_classes = len(label_list)
    y_true_onehot = np.eye(n_classes)[true_ids]
    y_score = torch.cat(all_fine_scores, dim=0).numpy()
    coarse_score_array = torch.cat(all_coarse_logits, dim=0).numpy()
    mk_score_array = torch.cat(all_mk_scores, dim=0).numpy()
    tk_score_array = torch.cat(all_tk_scores, dim=0).numpy()
    rk_score_array = torch.cat(all_rk_scores, dim=0).numpy()
    auprc = average_precision_score(y_true_onehot, y_score, average='weighted')

    score_dump_addr = getattr(config, "score_dump_addr", None)
    if score_dump_addr:
        import os
        os.makedirs(os.path.dirname(score_dump_addr), exist_ok=True)
        np.savez_compressed(
            score_dump_addr,
            coarse_logits=coarse_score_array,
            fine_score=y_score,
            mk_disc=mk_score_array,
            tk_disc=tk_score_array,
            rk_disc=rk_score_array,
            true_labels=np.asarray(true_labels, dtype=object),
            label_list=np.asarray(label_list, dtype=object),
            seen_mask=seen_mask,
            fixed_view_weights=np.asarray([w[0].item(), w[1].item(), w[2].item()], dtype=np.float32),
            router_prob=np.asarray([float(getattr(config, "router_prob", 0.25))], dtype=np.float32),
            router_margin=np.asarray([float(getattr(config, "router_margin", 2.5))], dtype=np.float32),
            router_entropy=np.asarray([float(getattr(config, "router_entropy", 0.25))], dtype=np.float32),
        )
        print(f"Saved prediction score dump: {score_dump_addr}")

    # ════════════════════════════════════════
    #  DETAILED PRINTING
    # ════════════════════════════════════════

    print("\n" + "=" * 80)
    print(f"  AGGREGATE METRICS")
    print(f"  Accuracy:  {correct}/{total} = {correct/total:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1:        {f1:.6f}")
    print(f"  AUPRC:     {auprc:.6f}")
    print(f"  Coarse Acc:{coarse_correct}/{total} = {coarse_correct/total:.6f}")
    print(f"  Consistency: {consistent_cnt}/{total} = {consistent_cnt/total:.6f}")
    print(f"  Consist-correct: {consist_correct}, Inconsist-correct: {inconsist_correct}")
    print(f"  Learned view_weights: mk={w[0].item():.3f} tk={w[1].item():.3f} rk={w[2].item():.3f}")
    print("=" * 80)

    _print_seen_unseen_metrics(split_metrics)
    _print_per_class_results(label_list, true_labels, preds, details)
    _print_misclassifications(label_list, true_labels, preds, details)
    _print_confusion_summary(label_list, true_labels, preds, details)

    return {
        "predictions": preds,
        "coarse_predictions": coarse_preds,
        "details": details,
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
