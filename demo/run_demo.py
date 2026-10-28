import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def canonical_hash(values):
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_inputs(path, seq_len, limit):
    records = []
    ids = set()
    source_rows = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            cell_id = item.get("cell_id")
            source_row = item.get("source_row")
            genes = item.get("genes")
            if not isinstance(cell_id, str) or not cell_id or cell_id in ids:
                raise ValueError(f"Input line {line_number}: cell_id must be a unique nonempty string.")
            if type(source_row) is not int or source_row < 0 or source_row in source_rows:
                raise ValueError(f"Input line {line_number}: source_row must be a unique nonnegative integer.")
            if not isinstance(genes, list) or len(genes) != seq_len:
                raise ValueError(f"Input line {line_number}: expected exactly {seq_len} ranked genes.")
            if not all(isinstance(gene, str) for gene in genes):
                raise ValueError(f"Input line {line_number}: every gene identifier must be a string.")


            records.append({"cell_id": cell_id, "source_row": source_row, "genes": genes})
            ids.add(cell_id)
            source_rows.add(source_row)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("The input contains no cells.")
    return records


def validate_resources(config, gene_info, knowledge):
    classes = config["class_order"]
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("config.class_order must contain at least two distinct classes in trained order.")
    if not isinstance(knowledge, list) or [c["cell_type"] for c in knowledge] != classes:
        raise ValueError("The cell-knowledge class order does not match config.class_order.")
    gene2id = gene_info["gene2id"]
    if not isinstance(gene2id, dict) or not gene2id:
        raise ValueError("gene_info.json must contain a nonempty gene2id mapping.")
    if not all(isinstance(gene, str) and gene for gene in gene2id):
        raise ValueError("gene2id mapping keys must be nonempty gene identifiers.")
    if any(type(v) is not int or v < 0 or v > 4096 for v in gene2id.values()):
        raise ValueError("gene2id indices are outside the model embedding range.")
    if len(set(gene2id.values())) != len(gene2id):
        raise ValueError("gene2id must preserve a one-to-one historical mapping.")
    gene_count = gene_info.get("num")
    if type(gene_count) is not int or gene_count <= max(gene2id.values()):
        raise ValueError("gene_info.num must exceed every gene2id index.")
    known_labels = config.get("known_labels")
    if (not isinstance(known_labels, list) or not known_labels
            or not all(isinstance(label, str) for label in known_labels)
            or len(set(known_labels)) != len(known_labels)
            or not set(known_labels).issubset(classes)):
        raise ValueError("known_labels must contain distinct reference labels from class_order.")
    for cell in knowledge:
        if isinstance(cell["cell_marker"], str):
            cell["cell_marker"] = cell["cell_marker"].split()
        if not isinstance(cell["cell_marker"], list) or len(cell["cell_marker"]) > int(config["marker_len"]):
            raise ValueError("Cell marker list does not fit the original marker length.")
        if not all(isinstance(g, str) for g in cell["cell_marker"]):
            raise ValueError("Cell marker identifiers must be strings.")
        if not all(isinstance(cell[k], str) for k in ("cell_tissue", "cell_role")):
            raise ValueError("Cell tissue and role knowledge must be strings.")
    return classes, gene2id


def infer(model, records, knowledge, gene_info, config, batch_size):
    from types import SimpleNamespace
    marker_len = int(config["marker_len"])
    seq_len = int(config["seq_len"])
    gene2id = gene_info["gene2id"]
    marker_evidence_weight = float(config.get("marker_evidence_weight", 0.5))
    seen_bias = float(config.get("seen_bias", 1.1))
    transcriptome_weight = float(config.get("transcriptome_weight", 0.25))
    if not 0 <= marker_evidence_weight <= 1 or not np.isfinite(seen_bias):
        raise ValueError("Marker evidence weight must be in [0, 1] and seen bias must be finite.")
    if not 0 <= transcriptome_weight <= 1:
        raise ValueError("Transcriptome weight must be in [0, 1].")
    fixed_weights = config.get("fixed_view_weights", [0.15, 0.30, 0.55])
    if (not isinstance(fixed_weights, list) or len(fixed_weights) != 3
            or not all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0
                       for value in fixed_weights) or sum(fixed_weights) <= 0):
        raise ValueError("fixed_view_weights must contain three finite nonnegative values with positive sum.")
    w = torch.tensor(fixed_weights, dtype=torch.float32, device=model.device)
    w = w / w.sum()
    known_label_set = set(config["known_labels"])
    reference_mask = torch.tensor([label in known_label_set for label in config["class_order"]],
                                  device=model.device)
    router_config = SimpleNamespace(**config)
    marker_ids = torch.full((len(knowledge), marker_len), -1, dtype=torch.long)
    for index, cell in enumerate(knowledge):
        ids = [gene2id.get(g, -1) for g in cell["cell_marker"]]
        marker_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    model.eval()
    outputs = {key: [] for key in ("coarse_logits", "mk_disc", "mk_fused", "marker_evidence",
                                   "tk_disc", "rk_disc", "final_probabilities")}
    routes = {key: [] for key in ("coarse_idx", "fine_idx", "baseline_idx", "final_idx", "used_coarse",
                                  "transcriptome_mix_applied", "coarse_probability", "coarse_margin",
                                  "coarse_entropy", "fine_score")}
    with torch.inference_mode():
        tissues = model.text_tokenizer([cell["cell_tissue"] for cell in knowledge])
        roles = model.text_tokenizer([cell["cell_role"] for cell in knowledge])
        tissue_emb = model.bio_emb(tissues)
        role_emb = model.bio_emb(roles)
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            B = len(batch)
            batch_seqs = [record["genes"] for record in batch]
            batch_seqs, batch_tokens, attention_mask, batch_length = model.dynamic_gene_batch(batch_seqs, seq_len)
            batch_ids = get_gene_id(batch_seqs, gene_info, batch_length).to(model.device)
            valid = batch_ids != -1
            pos_w = position_weights(batch_length, model.device, valid)[:, None, :]
            marker_pos_w = position_weights(batch_length, model.device, valid, 0.03)[:, None, :]
            current_markers = model.dynamic_marker_batch(batch_seqs, marker_ids).to(model.device)
            current_tissue = tissue_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            current_role = role_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            coarse_logits, marker_view, tissue_view, role_view = model(
                batch_tokens, batch_ids, current_markers, current_tissue, current_role, attention_mask=attention_mask
            )
            mk_raw = (marker_view.masked_fill(~valid[:, None, :], 0) * marker_pos_w).sum(dim=-1)
            tk_raw = (tissue_view.masked_fill(~valid[:, None, :], 0) * pos_w).sum(dim=-1)
            rk_raw = (role_view.masked_fill(~valid[:, None, :], 0) * pos_w).sum(dim=-1)
            mk_disc = compute_discriminative(mk_raw)
            tk_disc = compute_discriminative(tk_raw)
            rk_disc = compute_discriminative(rk_raw)
            marker_evidence = marker_membership_evidence(batch_ids, current_markers,
                                                         marker_pos_w[:, 0, :], gene_info["num"])
            mk_fused = calibrate_marker_evidence(mk_disc, marker_evidence, marker_evidence_weight)
            fine_score = w[0] * mk_fused + w[1] * tk_disc + w[2] * rk_disc
            baseline = transcriptome_knowledge_router(coarse_logits, fine_score, router_config)
            final_probabilities = fuse_scores(coarse_logits, fine_score, reference_mask,
                                               seen_bias=seen_bias, transcriptome_weight=transcriptome_weight)
            final_idx = final_probabilities.argmax(dim=-1)
            transcriptome_mix_applied = reference_mask[(fine_score - seen_bias * reference_mask.float()).argmax(dim=-1)]
            current_scores = {
                "coarse_logits": coarse_logits, "mk_disc": mk_disc, "mk_fused": mk_fused,
                "marker_evidence": marker_evidence, "tk_disc": tk_disc, "rk_disc": rk_disc,
                "final_probabilities": final_probabilities,
            }
            current_routes = {
                "coarse_idx": baseline["coarse_idx"], "fine_idx": baseline["fine_idx"],
                "baseline_idx": baseline["final_idx"], "final_idx": final_idx,
                "used_coarse": baseline["use_coarse"], "transcriptome_mix_applied": transcriptome_mix_applied,
                "coarse_probability": baseline["coarse_prob"], "coarse_margin": baseline["coarse_margin"],
                "coarse_entropy": baseline["coarse_entropy"], "fine_score": fine_score,
            }
            for key, value in current_scores.items():
                outputs[key].append(value.detach().cpu().numpy())
            for key, value in current_routes.items():
                routes[key].append(value.detach().cpu().numpy())
            print(f"Predicted {min(start + B, len(records))}/{len(records)} cells", flush=True)
    scores = {key: np.concatenate(values, axis=0) for key, values in outputs.items()}
    routed = {key: np.concatenate(values, axis=0) for key, values in routes.items()}
    return scores, routed


def write_predictions(path, records, classes, scores, routed):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "source_row", "coarse_prediction", "fine_prediction", "baseline_prediction", "prediction",
              "used_coarse", "transcriptome_mix_applied", "coarse_probability", "coarse_margin", "coarse_entropy",
              "class_order", "coarse_logits", "mk_disc", "mk_fused", "marker_evidence", "tk_disc", "rk_disc",
              "fine_score", "final_probabilities"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            row = {
                "cell_id": record["cell_id"], "source_row": record["source_row"],
                "coarse_prediction": classes[int(routed["coarse_idx"][i])],
                "fine_prediction": classes[int(routed["fine_idx"][i])],
                "baseline_prediction": classes[int(routed["baseline_idx"][i])],
                "prediction": classes[int(routed["final_idx"][i])],
                "used_coarse": bool(routed["used_coarse"][i]),
                "transcriptome_mix_applied": bool(routed["transcriptome_mix_applied"][i]),
                "coarse_probability": float(routed["coarse_probability"][i]),
                "coarse_margin": float(routed["coarse_margin"][i]),
                "coarse_entropy": float(routed["coarse_entropy"][i]),
                "class_order": json.dumps(classes, ensure_ascii=False),
                "fine_score": json.dumps(routed["fine_score"][i].tolist()),
            }
            for key, array in scores.items():
                row[key] = json.dumps(array[i].tolist())
            writer.writerow(row)


def classification_metrics(truth, predictions):
    total = len(truth)
    if not total:
        return {"cells": 0, "accuracy": None, "weighted_precision": None,
                "weighted_recall": None, "weighted_f1": None}
    truth = np.asarray(truth, dtype=str)
    predictions = np.asarray(predictions, dtype=str)
    precision = recall = f1 = 0.0
    for label in np.unique(truth):
        support = int(np.sum(truth == label))
        tp = int(np.sum((truth == label) & (predictions == label)))
        fp = int(np.sum((truth != label) & (predictions == label)))
        fn = support - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precision += support * p / total
        recall += support * r / total
        f1 += support * f / total
    return {"cells": total, "accuracy": float(np.mean(truth == predictions)),
            "weighted_precision": precision, "weighted_recall": recall, "weighted_f1": f1}


def evaluate(path, records, predictions):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {}
    for row in rows:
        if not row.get("cell_id") or row["cell_id"] in lookup or not row.get("true_label"):
            raise ValueError("Evaluation CSV requires unique cell_id and a nonempty true_label.")
        lookup[row["cell_id"]] = row
    selected = []
    for record in records:
        if record["cell_id"] not in lookup:
            raise ValueError(f"Evaluation label is missing for {record['cell_id']}.")
        row = lookup[record["cell_id"]]
        if row.get("source_row", "") and int(row["source_row"]) != record["source_row"]:
            raise ValueError("Evaluation source_row does not match the prediction input.")
        selected.append(row)
    truth = [row["true_label"] for row in selected]
    result = {"all": classification_metrics(truth, predictions)}
    if all(row.get("seen", "") for row in selected):
        def seen_value(value):
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError("Evaluation seen values must be true/false or 1/0.")
            return value.lower() in ("true", "1")
        seen = [seen_value(row["seen"]) for row in selected]
        for split, flag in (("seen", True), ("unseen", False)):
            indices = [i for i, value in enumerate(seen) if value == flag]
            result[split] = classification_metrics([truth[i] for i in indices], [predictions[i] for i in indices])
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline scAnnoAKM full24 inference from ranked genes and a complete checkpoint.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/predictions.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluation-labels", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size, threads, and limit must be positive.")
    if args.metrics_output and not args.evaluation_labels:
        parser.error("--metrics-output requires --evaluation-labels.")
    root = args.config.resolve().parent
    config = read_json(args.config)
    if "known_labels" not in config:
        provenance = read_json(resolve_path(root, config.get("provenance", "provenance.json")))
        if provenance.get("class_order") != config["class_order"]:
            raise ValueError("The provenance class order does not match config.class_order.")
        config["known_labels"] = provenance.get("known_labels")
    gene_info = read_json(resolve_path(root, config.get("gene_info", "resources/gene_info.json")))
    knowledge = read_json(resolve_path(root, config.get("cell_knowledge", "resources/cell_knowledge.json")))
    classes, _ = validate_resources(config, gene_info, knowledge)
    input_path = args.input or resolve_path(root, config.get("input", "examples/ranked_genes.jsonl"))
    records = read_inputs(input_path, int(config["seq_len"]), args.limit)
    checkpoint = args.checkpoint or resolve_path(root, config.get("checkpoint", "weights/train_model_seed42.pth"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Complete trained checkpoint not found: {checkpoint}. Supply --checkpoint PATH.")
    from model import load_complete_model
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = load_complete_model(config, checkpoint, root, torch.device(args.device))
    scores, routed = infer(model, records, knowledge, gene_info, config, args.batch_size)
    write_predictions(args.output, records, classes, scores, routed)
    predictions = [classes[int(i)] for i in routed["final_idx"]]
    print(json.dumps({"prediction_file": str(args.output.resolve()), "cells": len(records),
                      "policy": "full24_calibrated_fusion_marker_evidence_half_no_renorm", "device": args.device,
                      "prediction_sha256": canonical_hash(predictions)}, ensure_ascii=False))
    if args.evaluation_labels:
        metrics = evaluate(args.evaluation_labels, records, predictions)
        metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Evaluation metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
