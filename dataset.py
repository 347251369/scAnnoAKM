import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import os
import json

def load_dataset(file_addr: str) -> sc.AnnData:
    data_addr = "./datasets/" + file_addr + "/data.h5ad"
    adata = sc.read_h5ad(data_addr)
  
    return adata

def split_dataset(file_addr, dataset: sc.AnnData, test_size: float, random_seed : int) -> Tuple[sc.AnnData, sc.AnnData]:
    ref_addr = "./datasets/" + file_addr + "/ref_data.h5ad"
    tar_addr = "./datasets/" + file_addr + "/tar_data.h5ad"
    label_addr = "./datasets/" + file_addr + "/labels.txt"
    unlabel_addr = "./datasets/" + file_addr + "/unlabels.txt"
    
    if os.path.exists(ref_addr) and os.path.exists(tar_addr) and os.path.exists(label_addr) and os.path.exists(unlabel_addr):
        ref_data = sc.read_h5ad(ref_addr)
        tar_data = sc.read_h5ad(tar_addr)
        return ref_data, tar_data
    
    
    # Step1: select cell types with at least 2 samples
    cell_type_counts = dataset.obs['cell_type'].value_counts()
    valid_cell_types = cell_type_counts[cell_type_counts >= 2].index.tolist()
    
    # Step2: filter dataset
    dataset_filtered = dataset[dataset.obs['cell_type'].isin(valid_cell_types)].copy()

    # Step3: split dataset 70% ref and 30% tar 
    ref_indices, tar_indices = train_test_split(
        np.arange(dataset_filtered.n_obs),
        test_size=test_size,
        random_state=random_seed,
        stratify=dataset_filtered.obs['cell_type']
    )
    ref_data = dataset_filtered[ref_indices].copy()
    tar_data = dataset_filtered[tar_indices].copy()

    # step4: transfer last 30% cell types to target as unknown cell types
    sorted_types = dataset_filtered.obs['cell_type'].value_counts().sort_values(ascending=False).index.tolist()
    split_idx = int(len(sorted_types)*0.7)
    unknown_types = sorted_types[split_idx:]
    known_types = sorted_types[:split_idx]
    with open(label_addr, 'w', encoding='utf-8') as f:
        for t in known_types:
            f.write(f"{t}\n")
    with open(unlabel_addr, 'w', encoding='utf-8') as f:
        for t in unknown_types:
            f.write(f"{t}\n")
    # separate known and unknown cell types
    ref_unknown_mask = ref_data.obs['cell_type'].isin(unknown_types)
    temp = ref_data[ref_unknown_mask].copy()
    ref_data = ref_data[~ref_unknown_mask].copy()
    tar_data = sc.concat([tar_data, temp], axis=0)

    # Step5: Save datasets
    ref_data.write_h5ad(ref_addr)
    tar_data.write_h5ad(tar_addr)
    
    return ref_data, tar_data

def get_global_normalization_factors(full_adata: sc.AnnData, target_sum: float = 1e4) -> Dict[str, any]:
    if hasattr(full_adata.X, 'toarray'):
        cell_totals = full_adata.X.toarray().sum(axis=1).flatten()
    else:
        cell_totals = full_adata.X.sum(axis=1).flatten()
    scaling_factors = target_sum / np.maximum(cell_totals, 1e-6)
    cell2factor = dict(zip(full_adata.obs_names, scaling_factors))
    return {
        "cell2factor": cell2factor,
        "target_sum": target_sum
    }

def knowledge_acquire(file_addr: str):
    #1. load label
    label_addr = "datasets/" + file_addr + "/labels.txt"
    unlabel_addr = "datasets/" + file_addr + "/unlabels.txt"
    with open(label_addr, 'r', encoding='utf-8') as f:
            label = [line.strip() for line in f if line.strip()]
    with open(unlabel_addr, 'r', encoding='utf-8') as f:
            unlabel = [line.strip() for line in f if line.strip()] 
    label_total = label + unlabel
    
    #2. load knowledges
    knowledge_addr = "datasets/" + file_addr + "/knowledge/" 
    markers = []
    descriptions = []
    for i in label_total:
        cell_addr = knowledge_addr + i
        description_file = cell_addr + "/description.txt"
        marker_file = cell_addr + "/marker.txt"

        if not os.path.exists(cell_addr):
            os.makedirs(cell_addr, exist_ok=True)

        with open(description_file, 'r', encoding='utf-8') as f:
            description = f.read()     
        descriptions.append(description)

        with open(marker_file, 'r', encoding='utf-8') as f:
            marker = [line.strip() for line in f if line.strip()]  
        markers.append(marker)

    return label_total, descriptions, markers

def load_gene_knowledge(file_addr: str) -> Dict[str, Dict[str, List[str]]]:
    path = f"datasets/{file_addr}/knowledge/gene_knowledge.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    gene_knowledge: Dict[str, Dict[str, List[str]]] = {}
    for item in data:
        gene = item["gene"]
        tissue = item.get("cell_tissue", "")
        role = item.get("cell_role", "")
        
        if gene not in gene_knowledge:
            gene_knowledge[gene] = {"tissues": [], "roles": []}
        
        if tissue and tissue not in gene_knowledge[gene]["tissues"]:
            gene_knowledge[gene]["tissues"].append(tissue)
        if role and role not in gene_knowledge[gene]["roles"]:
            gene_knowledge[gene]["roles"].append(role)
    
    return gene_knowledge

if __name__ == '__main__':
    # 测试加载
    dataset = load_dataset()
    ref, target = split_dataset(dataset)