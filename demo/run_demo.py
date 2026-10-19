"""Offline scAnnoAKM inference from a complete checkpoint and ranked genes.

The neural forward is unchanged. The final decision follows the recovered
paper NumPy policy. Ground-truth labels are read only after prediction and
only when --evaluation-labels is explicitly supplied.
"""
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


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def row_z(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(p.shape[1])


def top2_margin(x):
    s = np.sort(x, axis=1)
    return s[:, -1] - s[:, -2]


def paper_route(coarse, mk_disc, tk_disc, rk_disc):
    """Exact recovered paper policy; do not normalize individual views again."""
    arrays = (coarse, mk_disc, tk_disc, rk_disc)
    if coarse.ndim != 2 or coarse.shape[1] < 2:
        raise ValueError("Prediction requires at least two candidate classes.")
    if any(a.shape != coarse.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Model scores must be finite, aligned N-by-C arrays.")
    W = (0.15, 0.30, 0.55)
    fine = W[0] * mk_disc + W[1] * tk_disc + W[2] * rk_disc
    coarse_z = row_z(coarse)
    coarse_idx = coarse_z.argmax(1)
    fine_idx = row_z(fine).argmax(1)
    prob = softmax(coarse)
    coarse_prob = prob.max(1)
    margin = top2_margin(coarse_z)
    ent = entropy(prob)
    use = (coarse_prob >= 0.25) & (margin >= 2.5) & (ent <= 0.25)
    final_idx = np.where(use, coarse_idx, fine_idx)
    return {
        "coarse_idx": coarse_idx, "fine_idx": fine_idx,
        "final_idx": final_idx, "used_coarse": use,
        "coarse_probability": coarse_prob, "coarse_margin": margin,
        "coarse_entropy": ent, "fine_score": fine,
    }


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
            # Empty exported entries preserve the original sequence position.
            # They map to Geneformer <mask> and marker ID -1, like an unknown gene.
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


def infer(model, records, knowledge, gene2id, config, batch_size):
    import torch
    marker_len = int(config["marker_len"])
    seq_len = int(config["seq_len"])
    marker_ids = torch.full((len(knowledge), marker_len), -1, dtype=torch.long)
    for index, cell in enumerate(knowledge):
        ids = [gene2id.get(g, -1) for g in cell["cell_marker"]]
        marker_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    model.eval()
    outputs = {key: [] for key in ("coarse_logits", "mk_disc", "tk_disc", "rk_disc")}
    with torch.inference_mode():
        tissues = model.text_tokenizer([cell["cell_tissue"] for cell in knowledge])
        roles = model.text_tokenizer([cell["cell_role"] for cell in knowledge])
        tissue_emb = model.bio_emb(tissues)
        role_emb = model.bio_emb(roles)
        k = torch.arange(seq_len, device=model.device, dtype=torch.float32)
        pos_w = torch.exp(-0.01 * k)
        pos_w = pos_w / pos_w.sum()
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            B = len(batch)
            batch_tokens = torch.cat([
                model.seq_tokenizer(record["genes"]).unsqueeze(0).to(model.device)
                for record in batch
            ], dim=0)
            batch_ids = torch.tensor([
                [gene2id.get(g, -1) for g in record["genes"]] for record in batch
            ], dtype=torch.long, device=model.device)
            current_markers = marker_ids.unsqueeze(0).expand(B, -1, -1).to(model.device)
            current_tissue = tissue_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            current_role = role_emb.unsqueeze(0).expand(B, -1, -1).to(model.device)
            coarse, marker_view, tissue_view, role_view = model(
                batch_tokens, batch_ids, current_markers, current_tissue, current_role
            )
            outputs["coarse_logits"].append(coarse.detach().cpu().numpy())
            for name, view in (("mk_disc", marker_view), ("tk_disc", tissue_view), ("rk_disc", role_view)):
                raw = (view * pos_w).sum(dim=-1)
                mean = raw.mean(dim=-1, keepdim=True)
                std = raw.std(dim=-1, keepdim=True)
                disc = torch.clamp((raw - mean) / (std + 1e-6), min=-10.0, max=10.0)
                outputs[name].append(disc.detach().cpu().numpy())
            print(f"Predicted {min(start + B, len(records))}/{len(records)} cells", flush=True)
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def write_predictions(path, records, classes, scores, routed):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "source_row", "coarse_prediction", "fine_prediction", "prediction",
              "used_coarse", "coarse_probability", "coarse_margin", "coarse_entropy",
              "class_order", "coarse_logits", "mk_disc", "tk_disc", "rk_disc", "fine_score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            row = {
                "cell_id": record["cell_id"], "source_row": record["source_row"],
                "coarse_prediction": classes[int(routed["coarse_idx"][i])],
                "fine_prediction": classes[int(routed["fine_idx"][i])],
                "prediction": classes[int(routed["final_idx"][i])],
                "used_coarse": bool(routed["used_coarse"][i]),
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
    parser = argparse.ArgumentParser(description=__doc__)
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
    gene_info = read_json(resolve_path(root, config.get("gene_info", "resources/gene_info.json")))
    knowledge = read_json(resolve_path(root, config.get("cell_knowledge", "resources/cell_knowledge.json")))
    classes, gene2id = validate_resources(config, gene_info, knowledge)
    input_path = args.input or resolve_path(root, config.get("input", "examples/ranked_genes.jsonl"))
    records = read_inputs(input_path, int(config["seq_len"]), args.limit)
    checkpoint = args.checkpoint or resolve_path(root, config.get("checkpoint", "weights/train_model_seed42.pth"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Complete trained checkpoint not found: {checkpoint}. Supply --checkpoint PATH.")
    import torch
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
    scores = infer(model, records, knowledge, gene2id, config, args.batch_size)
    routed = paper_route(scores["coarse_logits"], scores["mk_disc"], scores["tk_disc"], scores["rk_disc"])
    write_predictions(args.output, records, classes, scores, routed)
    predictions = [classes[int(i)] for i in routed["final_idx"]]
    print(json.dumps({"prediction_file": str(args.output.resolve()), "cells": len(records),
                      "policy": "published_numpy_and_router", "device": args.device,
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
