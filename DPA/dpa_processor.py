import scanpy as sc
import numpy as np
from typing import Tuple, List, Dict
import os
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DPA_TOP_GENES = 2000
DPA_PCA_COMPONENTS = 64
DPA_PCA_BOOST = 2.0
EPS = 1e-6

def save_seq_to_csv(ref_addr, tar_addr, ref_seq: List[List[str]], ref_label: List[str], tar_seq: List[List[str]], tar_label: List[str]) -> None:
    max_gene_len = max(len(ref_seq[0]), len(tar_seq[0]))
    gene_col_names = [f'gene_{i}' for i in range(max_gene_len)]
    df_ref = pd.DataFrame(ref_seq, columns=gene_col_names)
    df_ref['cell_type'] = ref_label
    df_tar = pd.DataFrame(tar_seq, columns=gene_col_names)
    df_tar['cell_type'] = tar_label
    df_ref.to_csv(ref_addr, encoding='utf-8', index=False)
    df_tar.to_csv(tar_addr, encoding='utf-8', index=False)

def normalize_log_matrix(adata: sc.AnnData, global_norm_factors: Dict[str, any]):
    cell2factor = global_norm_factors["cell2factor"]
    adata_scaling_factors = np.array([cell2factor.get(cell, 1.0) for cell in adata.obs_names])
    expr = adata.X
    if sparse.issparse(expr):
        expr = expr.tocsr(copy=True).astype(np.float32)
        expr = expr.multiply(adata_scaling_factors[:, np.newaxis])
        expr.data = np.log1p(np.nan_to_num(expr.data, nan=0.0, posinf=0.0, neginf=0.0))
        expr.eliminate_zeros()
    else:
        expr = np.asarray(expr, dtype=np.float32) * adata_scaling_factors[:, np.newaxis]
        expr = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)
        expr = np.log1p(expr)
    return expr


def reference_background(ref_data: sc.AnnData, global_norm_factors: Dict[str, any]):
    ref_expr = normalize_log_matrix(ref_data, global_norm_factors)
    n_cells = max(ref_data.n_obs, 1)
    if sparse.issparse(ref_expr):
        ref_csc = ref_expr.tocsc()
        mean = np.asarray(ref_csc.sum(axis=0)).ravel() / n_cells
        second = np.asarray(ref_csc.power(2).sum(axis=0)).ravel() / n_cells
        detected = np.diff(ref_csc.indptr).astype(np.float32)
    else:
        mean = ref_expr.mean(axis=0)
        second = np.square(ref_expr).mean(axis=0)
        detected = np.count_nonzero(ref_expr > 0, axis=0).astype(np.float32)

    std = np.sqrt(np.maximum(second - mean * mean, EPS)).astype(np.float32)
    idf = np.log1p(n_cells / (1.0 + detected)).astype(np.float32)
    return mean.astype(np.float32), std, idf


def pca_gene_leverage(ref_data: sc.AnnData, global_norm_factors: Dict[str, any]):
    ref_expr = normalize_log_matrix(ref_data, global_norm_factors)
    n_components = min(
        DPA_PCA_COMPONENTS,
        max(2, ref_expr.shape[0] - 1),
        max(2, ref_expr.shape[1] - 1),
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(ref_expr)
    component_weight = svd.explained_variance_ratio_.astype(np.float32)
    gene_weight = np.abs(svd.components_).T @ component_weight
    gene_weight = np.asarray(gene_weight, dtype=np.float32)
    low, high = np.percentile(gene_weight, [1, 99])
    if high <= low:
        return np.zeros_like(gene_weight, dtype=np.float32)
    gene_weight = np.clip((gene_weight - low) / (high - low), 0.0, 1.0)
    return gene_weight.astype(np.float32)


def background_aware_seq(
    adata: sc.AnnData,
    global_norm_factors: Dict[str, any],
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    ref_idf: np.ndarray,
    pca_weight: np.ndarray,
    top_n: int = DPA_TOP_GENES,
) -> List[List[str]]:
    expr = normalize_log_matrix(adata, global_norm_factors)
    genes = np.asarray(adata.var_names.astype(str))
    seqs = []

    if sparse.issparse(expr):
        expr = expr.tocsr()
        for row_idx in range(expr.shape[0]):
            start, end = expr.indptr[row_idx], expr.indptr[row_idx + 1]
            gene_idx = expr.indices[start:end]
            values = expr.data[start:end]
            if len(gene_idx) == 0:
                seqs.append([""] * top_n)
                continue
            z_pos = np.maximum((values - ref_mean[gene_idx]) / ref_std[gene_idx], 0.0)
            scores = (
                values
                * (1.0 + z_pos)
                * ref_idf[gene_idx]
                * (1.0 + DPA_PCA_BOOST * pca_weight[gene_idx])
            )
            if len(scores) > top_n:
                selected = np.argpartition(scores, -top_n)[-top_n:]
                ordered = selected[np.argsort(scores[selected])[::-1]]
            else:
                ordered = np.argsort(scores)[::-1]
            names = genes[gene_idx[ordered]].tolist()
            names.extend([""] * max(0, top_n - len(names)))
            seqs.append(names[:top_n])
    else:
        for values in expr:
            z_pos = np.maximum((values - ref_mean) / ref_std, 0.0)
            scores = values * (1.0 + z_pos) * ref_idf * (1.0 + DPA_PCA_BOOST * pca_weight)
            if len(scores) > top_n:
                selected = np.argpartition(scores, -top_n)[-top_n:]
                ordered = selected[np.argsort(scores[selected])[::-1]]
            else:
                ordered = np.argsort(scores)[::-1]
            seqs.append(genes[ordered[:top_n]].tolist())
    return seqs

def DPA(brain, file_addr: str, ref_data: sc.AnnData, tar_data: sc.AnnData, global_norm_factors: Dict[str, any])-> Tuple[List[List[str]], List[str], List[List[str]], List[str]]:
    ref_addr = "datasets/" + file_addr + "/ref_seq.csv"
    tar_addr = "datasets/" + file_addr + "/tar_seq.csv"
    if os.path.exists(ref_addr) and os.path.exists(tar_addr):
        ref_seq_df = pd.read_csv(ref_addr, encoding='utf-8')
        tar_seq_df = pd.read_csv(tar_addr, encoding='utf-8')
        gene_cols = [col for col in ref_seq_df.columns if col.startswith('gene_')]
        if len(gene_cols) >= DPA_TOP_GENES:
            ref_seq = ref_seq_df[gene_cols].values.tolist()
            ref_label = ref_seq_df['cell_type'].tolist()
            tar_seq = tar_seq_df[gene_cols].values.tolist()
            tar_label = tar_seq_df['cell_type'].tolist()
            return ref_seq, ref_label, tar_seq, tar_label

    ref_label = ref_data.obs['cell_type'].tolist()
    tar_label = tar_data.obs['cell_type'].tolist()
    print("DPA: computing reference background statistics")
    ref_mean, ref_std, ref_idf = reference_background(ref_data, global_norm_factors)
    print("DPA: computing PCA gene leverage")
    pca_weight = pca_gene_leverage(ref_data, global_norm_factors)
    print(f"DPA: ranking PCA-background-aware top{DPA_TOP_GENES} genes")
    ref_seq = background_aware_seq(ref_data, global_norm_factors, ref_mean, ref_std, ref_idf, pca_weight)
    tar_seq = background_aware_seq(tar_data, global_norm_factors, ref_mean, ref_std, ref_idf, pca_weight)

    save_seq_to_csv(ref_addr, tar_addr, ref_seq, ref_label, tar_seq, tar_label)

    return ref_seq, ref_label, tar_seq, tar_label


