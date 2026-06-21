import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def token_emb_text_knowledge(model, cell_knowledge):
    tissues = [cell['cell_tissue'] for cell in cell_knowledge]
    roles   = [cell['cell_role'] for cell in cell_knowledge]
    tissue_toks = model.text_tokenizer(tissues)
    role_toks   = model.text_tokenizer(roles)
    with torch.no_grad():
        cell_tissue_emb = model.bio_emb(tissue_toks)
        cell_role_emb   = model.bio_emb(role_toks)
    return cell_tissue_emb, cell_role_emb

def valid_gene_info(model, gene_names, gene_info):
    valid_genes = []
    for gene in gene_names:
        tok = model.seq_tokenizer([gene])
        if tok[1] != model.unk_id:
            valid_genes.append(gene)
    gene_info['num'] = len(valid_genes)
    gene_info['gene2id'] = {g: i for i, g in enumerate(valid_genes)}
    gene_info['id2gene'] = {i: g for g, i in gene_info['gene2id'].items()}
    return gene_info

def randint_with_neg1(shape, low, high, p_neg):
    x = torch.randint(low, high, shape)
    mask = torch.rand(shape) < p_neg
    x[mask] = -1
    return x

def gene_seq_data(gene_markers, batch_size, gene_num, seq_len):
    device = gene_markers.device
    seqs = randint_with_neg1((batch_size, seq_len), -1, gene_num, p_neg=0.3).to(device)
    seq_exp = seqs.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L] → [32,1,1,100]
    mark_exp = gene_markers.unsqueeze(-1)# [B, C, M, 1]
    match = (mark_exp == seq_exp)  # [B, C, M, L]
    hit = match.any(dim=-2) # [B, C, L]
    seq_mask =  (seqs != -1)
    valid_seq = seq_mask.unsqueeze(1)  # [B, 1, L]
    marker_mask = torch.where(valid_seq, hit.float(), 0.0)  # [B, C, L]
    return seqs, seq_mask, marker_mask

def get_text_labels(gene_seqs_ids, gene_info, gene_knowledge, cell_knowledge, label_type, device):
    B, L = gene_seqs_ids.shape
    num_cells = len(cell_knowledge)
    cell_key = f"gene_{label_type}_cell"
    
    gene_to_cell_ids = {g["gene"]: g[cell_key] for g in gene_knowledge}
    id2gene = gene_info["id2gene"]

    cell_to_genes = [set() for _ in range(num_cells)]
    for gene, cids in gene_to_cell_ids.items():
        for cid in cids:
            cell_to_genes[cid].add(gene)

    cell_scores = torch.zeros(B, num_cells, L, dtype=torch.float32, device=device)
    
    unique_gids = torch.unique(gene_seqs_ids)
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
        pos_gids = gene_seqs_ids[:, pos_idx]
        valid_mask = (pos_gids != -1)
        if not valid_mask.any():
            continue
            
        valid_gids = pos_gids[valid_mask]
        batch_masks = torch.stack([gid_to_cell_mask[gid.item()] for gid in valid_gids])
        cell_scores[valid_mask, :, pos_idx] = batch_masks

    return cell_scores

def generate_seq_by_ids(gene_seqs_ids, id2gene):
    max_id = max(id2gene.keys())
    vocab_list = [id2gene.get(i, "<mask>") for i in range(max_id + 1)]
    gene_names = [vocab_list[idx] for idx in gene_seqs_ids.flatten().tolist()]
    B, L = gene_seqs_ids.shape
    return [gene_names[i*L : (i+1)*L] for i in range(B)]

def token_marker(gene_info, cell_knowledge, marker_len):
    gene2id = gene_info['gene2id']
    num_classes = len(cell_knowledge)

    marker_lists = [cell['cell_marker'] for cell in cell_knowledge]
    marker_ids = torch.full((num_classes, marker_len), fill_value=-1, dtype=torch.long)

    for i, markers in enumerate(marker_lists):
        ids = [gene2id.get(m, -1) for m in markers]
        marker_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return marker_ids

def pretrain_model(model, gene_knowledge, gene_info, cell_knowledge, config):
    for p in list(model.gf.parameters()) + list(model.bio.parameters()):
        p.requires_grad = False

    gene_names = [gk["gene"] for gk in gene_knowledge]
    gene_info = valid_gene_info(model, gene_names, gene_info)
    torch.save(gene_info, config.gene_info_addr)

    gene_markers = token_marker(gene_info, cell_knowledge, config.marker_len) # [class_num, marker_len]
    gene_markers = gene_markers.unsqueeze(0).expand(config.pretrain_batch_size, -1, -1).to(model.device) # [B, class_num, marker_len]
    
    tissu_emb, role_emb = token_emb_text_knowledge(model, cell_knowledge) #[class_num, emb]
    tissu_emb = tissu_emb.unsqueeze(0).expand(config.pretrain_batch_size, -1, -1).to(model.device)  # [B, class_num, emb]
    role_emb = role_emb.unsqueeze(0).expand(config.pretrain_batch_size, -1, -1).to(model.device)    # [B, class_num, emb]

    # Separate parameter groups for marker vs text views
    marker_params = []
    text_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ['marker_id_emb', 'marker_scorer']):
            marker_params.append(p)
        else:
            text_params.append(p)

    opt = AdamW([
        {'params': marker_params, 'lr': 1e-2},
        {'params': text_params,   'lr': 1e-3},
    ], weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=config.pretrain_epochs, eta_min=1e-5)

    best_loss = float("inf")
    for epoch in range(config.pretrain_epochs):
        model.train()
        pbar = tqdm(range(40), desc=f"[pretrain] Epoch {epoch+1}/{config.pretrain_epochs}")
        total_loss, n_batches = 0.0, 0
        for i in pbar: # aoto add pretrain_batch_size for i
            # seq marker mask generate  [B, L], [B, C, L]
            gene_seqs_ids, seq_mask, marker_mask = gene_seq_data(gene_markers, config.pretrain_batch_size, gene_info['num'], config.seq_len) 
            gene_seqs_ids = gene_seqs_ids.to(model.device)
            
            # seq generate
            gene_seq = generate_seq_by_ids(gene_seqs_ids, gene_info['id2gene'])
            gene_toks = []
            for seq in gene_seq:
                tok = model.seq_tokenizer(seq)
                gene_toks.append(tok)
            gene_emb = model.gf_emb(torch.stack(gene_toks).to(model.device))
    
            #marker view:
            marker_mask = marker_mask.to(model.device)
            mask_m = model.marker_view_score(gene_seqs_ids, gene_markers)
            loss_m = F.binary_cross_entropy_with_logits(mask_m, marker_mask)

            #tissue view:
            tissue_labels = get_text_labels(gene_seqs_ids, gene_info, gene_knowledge, cell_knowledge, "tissue", config.device)
            mask_t = model.text_view_score(gene_emb, seq_mask, tissu_emb, "t_view", config.seq_len)
            loss_t = F.binary_cross_entropy_with_logits(mask_t, tissue_labels)

            #role view:
            role_labels = get_text_labels(gene_seqs_ids, gene_info, gene_knowledge, cell_knowledge, "role", config.device)
            mask_r = model.text_view_score(gene_emb, seq_mask, role_emb, "r_view", config.seq_len)
            loss_r = F.binary_cross_entropy_with_logits(mask_r, role_labels)

            loss = 0.5*loss_m + 0.25*loss_t + 0.25*loss_r

            opt.zero_grad()
            loss.backward()
            all_trainable = marker_params + text_params
            torch.nn.utils.clip_grad_norm_(all_trainable, config.gradient_clip_norm)
            opt.step()

            total_loss += loss.item()
            n_batches += 1
            
            #loss loss_m loss_t loss_r  
            pbar.set_postfix({
                "loss": f"{total_loss/n_batches:.4f}",
                "m": f"{loss_m.item():.4f}",
                "t": f"{loss_t.item():.4f}",
                "r": f"{loss_r.item():.4f}"
            })

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.pretrain_model_addr)
            print(f"[pretrain] Epoch {epoch+1}/{config.pretrain_epochs} | loss={avg_loss:.4f} * (best)")
        else:
            print(f"[pretrain] Epoch {epoch+1}/{config.pretrain_epochs} | loss={avg_loss:.4f}")
            
    return model, gene_info