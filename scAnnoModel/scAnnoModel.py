import os
import torch
from .model import scAnnoModel
from .train import train_model
from .predict import predict_model
from .pretrain import pretrain_model

def convert_cell_knowledge(cell_knowledge):
    for cell in cell_knowledge:
        if isinstance(cell["cell_marker"], str):
            cell["cell_marker"] = cell["cell_marker"].split()
    return cell_knowledge

def convert_gene_knowledge(gene_knowledge):
    if isinstance(gene_knowledge, dict):
        return [{"gene": gn, "tissues": v.get("tissues", []), "roles": v.get("roles", [])} for gn, v in gene_knowledge.items()]

    gene_map = {}
    for item in gene_knowledge:
        if isinstance(item, str):
            continue
        gn = item.get("gene") or item.get("gene_name")
        if not gn:
            continue
        if gn not in gene_map:
            gene_map[gn] = {"gene": gn, "tissues": [], "roles": []}
        tissue = (item.get("cell_tissue") or "").strip()
        role = (item.get("cell_role") or "").strip()
        if tissue and tissue not in gene_map[gn]["tissues"]:
            gene_map[gn]["tissues"].append(tissue)
        if role and role not in gene_map[gn]["roles"]:
            gene_map[gn]["roles"].append(role)
    return list(gene_map.values())

def add_gene_cell_links(genes_knowledge, cells_knowledge):
    cell_name_to_id = {cell["cell_type"]: idx for idx, cell in enumerate(cells_knowledge)}
    tissue_to_cell_ids = {}
    role_to_cell_ids = {}

    for cell in cells_knowledge:
        cell_type = cell['cell_type']
        tissue = cell['cell_tissue']
        role = cell['cell_role']
        cell_id = cell_name_to_id[cell_type]

        if tissue not in tissue_to_cell_ids:
            tissue_to_cell_ids[tissue] = []
        tissue_to_cell_ids[tissue].append(cell_id)

        if role not in role_to_cell_ids:
            role_to_cell_ids[role] = []
        role_to_cell_ids[role].append(cell_id)

    for gene in genes_knowledge:
        gene_tissue_cell_ids = []
        for t in gene['tissues']:
            gene_tissue_cell_ids.extend(tissue_to_cell_ids.get(t, []))
        gene['gene_tissue_cell'] = list(set(gene_tissue_cell_ids))

        gene_role_cell_ids = []
        for r in gene['roles']:
            gene_role_cell_ids.extend(role_to_cell_ids.get(r, []))
        gene['gene_role_cell'] = list(set(gene_role_cell_ids))

    return genes_knowledge


def run_scAnnoModel(file_addr, gene_knowledge, cell_knowledge, ref_seq, ref_label, tar_seq, tar_label, config):


    if not (config.is_train or config.is_predict):
        return None

    gene_knowledge = convert_gene_knowledge(gene_knowledge)


    cell_knowledge = convert_cell_knowledge(cell_knowledge)


    gene_knowledge = add_gene_cell_links(gene_knowledge, cell_knowledge)

    os.makedirs(f"results/{file_addr}", exist_ok=True)
    model = scAnnoModel(len(cell_knowledge), config.seq_len, config.marker_len)
    model.to(config.device)

    def load_pair(weights_path):
        for path in (weights_path, config.gene_info_addr):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return torch.load(config.gene_info_addr, map_location="cpu", weights_only=True)

    if not config.is_train:
        gene_info = load_pair(config.train_model_addr)
    else:
        if os.path.lexists(config.train_model_addr):
            raise FileExistsError(config.train_model_addr)
        if getattr(config, "reuse_pretrain", False):
            gene_info = load_pair(config.pretrain_model_addr)
        else:
            for path in (config.pretrain_model_addr, config.gene_info_addr):
                if os.path.lexists(path):
                    raise FileExistsError(path)
            model, gene_info = pretrain_model(model, gene_knowledge, {}, cell_knowledge, config)
        model = train_model(model, cell_knowledge, gene_info, gene_knowledge, ref_seq, ref_label, config)


    results = None
    if config.is_predict == True:
        results = predict_model(model, cell_knowledge, gene_info, tar_seq, tar_label, config, known_labels=ref_label)

        m = results["metrics"]
        results_file = f'results/{file_addr}/results_seed{config.seed}.txt'
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"seed={config.seed}\n")
            f.write(f"accuracy={m['accuracy']:.6f}\n")
            f.write(f"recall={m['recall']:.6f}\n")
            f.write(f"f1={m['f1']:.6f}\n")
            f.write(f"auprc={m['auprc']:.6f}\n")
            f.write(f"all_accuracy={m['all_accuracy']:.6f}\n")
            f.write(f"all_recall={m['all_recall']:.6f}\n")
            f.write(f"all_f1={m['all_f1']:.6f}\n")
            f.write(f"seen_accuracy={m['seen_accuracy']:.6f}\n")
            f.write(f"seen_recall={m['seen_recall']:.6f}\n")
            f.write(f"seen_f1={m['seen_f1']:.6f}\n")
            f.write(f"unseen_accuracy={m['unseen_accuracy']:.6f}\n")
            f.write(f"unseen_recall={m['unseen_recall']:.6f}\n")
            f.write(f"unseen_f1={m['unseen_f1']:.6f}\n")

    return results
