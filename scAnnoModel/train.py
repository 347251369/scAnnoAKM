import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn.functional as F


def position_weights(L, device):
    k = torch.arange(L, device=device, dtype=torch.float32)
    w = torch.exp(-0.01 * k)
    return w / w.sum()


def compute_discriminative(scores, eps=1e-6):
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True)
    z = (scores - mean) / (std + eps)
    return torch.clamp(z, min=-10.0, max=10.0)


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

def token_gene(model, input_seqs):
    tokenized_seqs = []
    for seq in input_seqs:
        tok = model.seq_tokenizer(seq)
        tokenized_seqs.append(tok)
    tokenized_seqs = torch.stack(tokenized_seqs).to(model.device)
    return tokenized_seqs

def get_gene_id(input_seqs, gene_info, gene_size):
    batch_ids = []
    for seq in input_seqs:
        ids = [gene_info['gene2id'].get(g, -1) for g in seq]
        truncated_ids = ids[:gene_size]
        while len(truncated_ids) < gene_size:
            truncated_ids.append(-1)
        batch_ids.append(truncated_ids)
    return torch.tensor(batch_ids, dtype=torch.long)

def get_coarse_labels(batch_labels, cell_knowledge):
    cell2id = {cell['cell_type']: idx for idx, cell in enumerate(cell_knowledge)}
    label_ids = [cell2id[label] for label in batch_labels]
    return torch.tensor(label_ids, dtype=torch.long)

def get_marker_labels(batch_gene_ids, gene_markers, device):
    B, L = batch_gene_ids.shape
    C, M = gene_markers.shape

    seq_exp = batch_gene_ids.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    mark_exp = gene_markers.unsqueeze(0).unsqueeze(-1)  # [1, C, M, 1]
    match = (mark_exp == seq_exp)                        # [B, C, M, L]
    hit = match.any(dim=-2)                              # [B, C, L]

    valid = (batch_gene_ids != -1).unsqueeze(1)          # [B, 1, L]
    marker_labels = torch.where(valid, hit.float(), 0.0) # [B, C, L]
    return marker_labels

def get_text_view_labels(batch_gene_ids, gene_info, gene_knowledge, cell_knowledge, label_type, device):
    B, L = batch_gene_ids.shape
    num_cells = len(cell_knowledge)
    cell_key = f"gene_{label_type}_cell"
    
    gene_to_cell_ids = {g["gene"]: g[cell_key] for g in gene_knowledge}
    id2gene = gene_info["id2gene"]
    
    cell_to_genes = [set() for _ in range(num_cells)]
    for gene, cids in gene_to_cell_ids.items():
        for cid in cids:
            cell_to_genes[cid].add(gene)
    
    cell_scores = torch.zeros(B, num_cells, L, dtype=torch.float32, device=device)
    
    unique_gids = torch.unique(batch_gene_ids)
    unique_gids = unique_gids[unique_gids != -1]
    
    gid_to_cell_mask = {}
    for gid in unique_gids:
        gid_item = gid.item()
        gene_name = id2gene.get(gid_item)
        mask = torch.zeros(num_cells, dtype=torch.float32, device=device)
        if gene_name:
            for cid in range(num_cells):
                if gene_name in cell_to_genes[cid]:
                    mask[cid] = 1.0
        gid_to_cell_mask[gid_item] = mask
    
    for pos_idx in range(L):
        pos_gids = batch_gene_ids[:, pos_idx]
        valid_mask = (pos_gids != -1)
        if not valid_mask.any():
            continue
        valid_gids = pos_gids[valid_mask]
        batch_masks = torch.stack([gid_to_cell_mask[gid.item()] for gid in valid_gids])
        cell_scores[valid_mask, :, pos_idx] = batch_masks
    
    return cell_scores

def train_model(model, cell_knowledge, gene_info, gene_knowledge, ref_seqs, ref_labels, config):
    '''
    cell_knowledge = {
        'check': True,
        'number': 0,
        'cell_type': 'eye photoreceptor cell',
        'cell_marker': ['ENSG00000265203', 'ENSG00000170345'],
        'cell_tissue': 'retina',
        'cell_role': 'phototransduction'
    }
    ref_labels = ['eye photoreceptor cell', 'enteroglial cell']
    '''

    gene_markers = token_marker(gene_info, cell_knowledge, config.marker_len).to(model.device) # [class_num, marker_len]
    tissu_emb, role_emb = token_emb_text_knowledge(model, cell_knowledge) #[class_num, emb]

    for p in list(model.gf.parameters()):
        p.requires_grad = True
    for p in list(model.bio.parameters()):
        p.requires_grad = False

    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(trainable, lr=config.train_lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.train_epochs, eta_min=config.train_lr / 100)

    best_loss = float("inf")
    for epoch in range(config.train_epochs):
        model.train()

        total_loss = 0.0
        total_coarse_loss = 0.0
        total_marker_loss = 0.0
        total_tissue_loss = 0.0
        total_role_loss = 0.0
        total_fine_loss = 0.0
        n_batches = 0

        indices = list(range(len(ref_labels)))
        pbar = tqdm(range(0, len(indices), config.train_batch_size), desc=f"[train] Epoch {epoch+1}/{config.train_epochs}")
        for i in pbar:
            batch_indices = indices[i:i + config.train_batch_size]
            batch_seqs = [ref_seqs[i] for i in batch_indices]
            batch_seqs_token= token_gene(model, batch_seqs)   # seqs_token [batch_size, seq_len]
            batch_gene_ids = get_gene_id(batch_seqs, gene_info, config.seq_len).to(model.device) # gene_ids [batch_size, gene_size]

            cur_gene_markers = gene_markers.unsqueeze(0).expand(len(batch_indices), -1, -1).to(model.device)
            cur_tissu_emb = tissu_emb.unsqueeze(0).expand(len(batch_indices), -1, -1).to(model.device)
            cur_role_emb = role_emb.unsqueeze(0).expand(len(batch_indices), -1, -1).to(model.device)

            coarse_logits, marker_view_logits, tissue_view_logits, role_view_logits = model(batch_seqs_token, batch_gene_ids, cur_gene_markers, cur_tissu_emb, cur_role_emb)
            '''
            coarse_logits [B, C]
            marker_view_logits [B, C, L]
            tissue_view_logits [B, C, L]
            role_view_logits [B, C, L]
            '''
            batch_labels = [ref_labels[i] for i in batch_indices]
            coarse_labels = get_coarse_labels(batch_labels, cell_knowledge).to(model.device)
            marker_labels = get_marker_labels(batch_gene_ids, gene_markers, model.device)
            tissue_labels = get_text_view_labels(batch_gene_ids, gene_info, gene_knowledge, cell_knowledge, "tissue", model.device)
            role_labels = get_text_view_labels(batch_gene_ids, gene_info, gene_knowledge, cell_knowledge, "role", model.device)
            
            
            coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)
            marker_loss = F.binary_cross_entropy_with_logits(marker_view_logits, marker_labels)
            tissue_loss = F.binary_cross_entropy_with_logits(tissue_view_logits, tissue_labels)
            role_loss = F.binary_cross_entropy_with_logits(role_view_logits, role_labels)

            # ---- Fixed multi-view evidence fusion ----
            B = len(batch_indices)
            pos_w = position_weights(config.seq_len, model.device)
            mk_raw = (marker_view_logits * pos_w).sum(dim=-1)
            tk_raw = (tissue_view_logits * pos_w).sum(dim=-1)
            rk_raw = (role_view_logits   * pos_w).sum(dim=-1)
            mk_disc = compute_discriminative(mk_raw)
            tk_disc = compute_discriminative(tk_raw)
            rk_disc = compute_discriminative(rk_raw)
            fixed_view_weights = getattr(config, "fixed_view_weights", None)
            if fixed_view_weights is None:
                w = F.softmax(model.view_weights, dim=0)
            else:
                w = torch.tensor(fixed_view_weights, dtype=torch.float32, device=model.device)
                w = w / w.sum()
            fine_score = w[0]*mk_disc + w[1]*tk_disc + w[2]*rk_disc
            fine_loss = F.cross_entropy(fine_score, coarse_labels)

            total_batch_loss = coarse_loss + marker_loss + tissue_loss + role_loss + fine_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_coarse_loss += coarse_loss.item()
            total_marker_loss += marker_loss.item()
            total_tissue_loss += tissue_loss.item()
            total_role_loss += role_loss.item()
            total_fine_loss += fine_loss.item()
            n_batches += 1

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{total_loss/n_batches:.4f}",
                "c": f"{total_coarse_loss/n_batches:.4f}",
                "m": f"{total_marker_loss/n_batches:.4f}",
                "t": f"{total_tissue_loss/n_batches:.4f}",
                "r": f"{total_role_loss/n_batches:.4f}",
                "f": f"{total_fine_loss/n_batches:.4f}"
            })

        scheduler.step()
        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.train_model_addr)
            print(f"[train] Epoch {epoch+1}/{config.train_epochs} | loss={avg_loss:.4f} * (best)")
        else:
            print(f"[train] Epoch {epoch+1}/{config.train_epochs} | loss={avg_loss:.4f}")

    model.load_state_dict(torch.load(config.train_model_addr, map_location=model.device))
    return model
