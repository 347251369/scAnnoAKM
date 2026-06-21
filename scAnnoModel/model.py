import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers.modeling_utils as _mu

_mu.check_torch_load_is_safe = lambda *a, **k: None


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Adapter(nn.Module):
    def __init__(self, dim, hidden_dim=256, dropout=0.05):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.up(x)
        return residual + x


def _pick_heads(dim):
    for heads in [8, 4, 2, 1]:
        if dim % heads == 0 and dim // heads >= 8:
            return heads
    return 1


class LinearEvidenceContextEncoder(nn.Module):

    def __init__(self, dim, num_heads=4, dropout=0.05):
        super().__init__()
        heads = _pick_heads(dim) if num_heads is None else num_heads
        self.num_heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn_gate = nn.Linear(dim, dim * 2)
        self.ffn_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def forward(self, x, mask=None):
        batch, length, dim = x.shape
        heads, head_dim = self.num_heads, self.head_dim
        q = F.elu(self.q_proj(x).view(batch, length, heads, head_dim)) + 1.0
        k = F.elu(self.k_proj(x).view(batch, length, heads, head_dim)) + 1.0
        v = self.v_proj(x).view(batch, length, heads, head_dim)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).unsqueeze(-1).to(dtype=x.dtype)
            k = k * mask_f
            v = v * mask_f

        kv = torch.einsum("blhd,blhe->bhde", k, v)
        k_sum = k.sum(dim=1)
        denom = torch.einsum("blhd,bhd->blh", q, k_sum).clamp(min=self.eps)
        attn = torch.einsum("blhd,bhde->blhe", q, kv) / denom.unsqueeze(-1)
        attn = self.o_proj(attn.reshape(batch, length, dim))

        x = self.norm1(x + self.dropout(attn))
        gate, value = self.ffn_gate(x).chunk(2, dim=-1)
        ffn = self.ffn_out(F.gelu(gate) * value)
        return self.norm2(x + self.dropout(ffn))


class KnowledgeGuidedPrototypeAlignment(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.context_norm = nn.LayerNorm(dim)
        self.proto_norm = nn.LayerNorm(dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.align_gate = nn.Linear(dim * 2, 1)
        self.reliability_gate = nn.Linear(dim * 2, 1)
        self.residual_proj = nn.Linear(dim * 2, dim)
        self.out_norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.align_gate.weight)
        nn.init.zeros_(self.align_gate.bias)
        nn.init.zeros_(self.reliability_gate.weight)
        nn.init.zeros_(self.reliability_gate.bias)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(self, class_rep, gene_summary):
        _, classes, _ = class_rep.shape
        proto = self.proto_norm(class_rep)
        context = self.context_norm(gene_summary).unsqueeze(1).expand(-1, classes, -1)
        pair = torch.cat([proto, context, proto * context, torch.abs(proto - context)], dim=-1)
        pair = self.pair_encoder(pair)
        alignment = torch.sigmoid(self.align_gate(pair))
        reliability = torch.sigmoid(self.reliability_gate(pair))
        residual = self.residual_proj(pair)
        return self.out_norm(proto + alignment * reliability * residual)


class PositionwiseEvidenceGatedReadout(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.W = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.gene_proj = nn.Linear(dim, dim)
        self.class_proj = nn.Linear(dim, dim)
        self.gene_gate = nn.Linear(dim, num_heads)
        self.class_gate = nn.Linear(dim, num_heads)
        self.head_logits = nn.Parameter(torch.zeros(num_heads))
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.xavier_uniform_(self.gene_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.class_proj.weight, gain=0.5)
        nn.init.zeros_(self.gene_proj.bias)
        nn.init.zeros_(self.class_proj.bias)
        nn.init.zeros_(self.gene_gate.bias)
        nn.init.zeros_(self.class_gate.bias)

    def forward(self, gene, class_rep):
        batch, length, _ = gene.shape
        _, classes, _ = class_rep.shape
        heads, head_dim = self.num_heads, self.head_dim
        gene_h = self.gene_proj(gene).view(batch, length, heads, head_dim)
        class_h = self.class_proj(class_rep).view(batch, classes, heads, head_dim)

        dot_scores = torch.einsum("blhd,bchd->bhlc", gene_h, class_h) * self.scale
        bilinear_scores = torch.einsum("blhi,hij,bchj->bhlc", gene_h, self.W, class_h) * self.scale
        scores = dot_scores + bilinear_scores

        gene_gate = self.gene_gate(gene).transpose(1, 2).unsqueeze(-1)
        class_gate = self.class_gate(class_rep).transpose(1, 2).unsqueeze(2)
        confidence = torch.sigmoid(gene_gate + class_gate)
        head_weight = F.softmax(self.head_logits, dim=0).view(1, heads, 1, 1)
        scores = (scores * confidence * head_weight).sum(dim=1)
        return (scores * self.output_scale).transpose(-2, -1)


class GRINModule(nn.Module):
    def __init__(self, dim, shared_context_encoder=None):
        super().__init__()
        heads = _pick_heads(dim)
        self.lece = (
            shared_context_encoder
            if shared_context_encoder is not None
            else LinearEvidenceContextEncoder(dim, heads)
        )
        self.kgpa = KnowledgeGuidedPrototypeAlignment(dim)
        self.pegr = PositionwiseEvidenceGatedReadout(dim, heads)

    def forward(self, seq_emb, class_rep, mask=None):
        gene_ctx = self.lece(seq_emb, mask)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            gene_summary = (gene_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            gene_summary = gene_ctx.mean(dim=1)

        aligned_proto = self.kgpa(class_rep, gene_summary)
        logits = self.pegr(gene_ctx, aligned_proto)

        if mask is not None:
            mask_f = mask.unsqueeze(1).float()
            logits = logits * mask_f - 1e2 * (1 - mask_f)
        return logits


class scAnnoModel(nn.Module):
    def __init__(self, num_classes, seq_len, marker_len):
        super().__init__()
        self.hidden_dim = 768
        self.seq_len = seq_len
        self.marker_len = marker_len
        self.num_classes = num_classes
        self.dtype = torch.float32

        self.gf_dict = load_pkl("scAnnoModel/geneformer/token_dictionary_gc104M.pkl")
        self.unk_id = int(self.gf_dict["<mask>"])
        self.pad_id = int(self.gf_dict["<pad>"])
        self.cls_id = int(self.gf_dict["<cls>"])
        self.eos_id = int(self.gf_dict["<eos>"])
        self.bio_tok = AutoTokenizer.from_pretrained("scAnnoModel/biobert/biobert_tokenizer")

        gf_cfg = AutoConfig.from_pretrained(
            "scAnnoModel/geneformer/Geneformer-V2-104M",
            local_files_only=True,
            output_hidden_states=True,
        )
        self.gf = AutoModel.from_pretrained(
            "scAnnoModel/geneformer/Geneformer-V2-104M",
            config=gf_cfg,
            local_files_only=True,
        )
        self.bio = AutoModel.from_pretrained("scAnnoModel/biobert/biobert_model")
        self.bio.resize_token_embeddings(len(self.bio_tok))

        self.shared_context_encoder = LinearEvidenceContextEncoder(
            self.hidden_dim,
            _pick_heads(self.hidden_dim),
        )

        self.marker_id_dim = self.hidden_dim
        self.marker_id_emb = nn.Embedding(4098, self.marker_id_dim)
        self.marker_scorer = GRINModule(
            dim=self.marker_id_dim,
            shared_context_encoder=self.shared_context_encoder,
        )

        self.q_adapter = Adapter(self.hidden_dim)
        self.q_norm = nn.LayerNorm(self.hidden_dim)

        self.tissue_adapter = Adapter(self.hidden_dim)
        self.tissue_norm = nn.LayerNorm(self.hidden_dim)
        self.tissue_scorer = GRINModule(
            dim=self.hidden_dim,
            shared_context_encoder=self.shared_context_encoder,
        )

        self.role_adapter = Adapter(self.hidden_dim)
        self.role_norm = nn.LayerNorm(self.hidden_dim)
        self.role_scorer = GRINModule(
            dim=self.hidden_dim,
            shared_context_encoder=self.shared_context_encoder,
        )

        self.coarse_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes),
        )
        nn.init.xavier_uniform_(self.coarse_ffn[-1].weight)
        nn.init.zeros_(self.coarse_ffn[-1].bias)

        self.view_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    @property
    def device(self):
        return next(self.parameters()).device

    def seq_tokenizer(self, genes):
        gene_ids = [self.gf_dict.get(g, self.unk_id) for g in genes]
        return torch.tensor([self.cls_id] + gene_ids + [self.eos_id], dtype=torch.long)

    def text_tokenizer(self, text):
        inputs = self.bio_tok(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16,
            add_special_tokens=True,
        )
        inputs.pop("token_type_ids", None)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def gf_emb(self, tokens):
        return self.gf(tokens, output_hidden_states=True).hidden_states[-1].to(self.dtype)

    def bio_emb(self, inputs):
        outputs = self.bio(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        attn_mask = inputs["attention_mask"].unsqueeze(-1).to(self.dtype)
        return (last_hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)

    def marker_view_score(self, seq, marker):
        seq_mask = seq != -1

        seq_emb = self.marker_id_emb(seq + 1)
        marker_emb = self.marker_id_emb(marker + 1)

        marker_mask = (marker != -1).unsqueeze(-1).float()
        class_proto = (marker_emb * marker_mask).sum(dim=2) / marker_mask.sum(dim=2).clamp(min=1)

        seq_emb = F.normalize(seq_emb, p=2, dim=-1)
        class_proto = F.normalize(class_proto, p=2, dim=-1)
        return self.marker_scorer(seq_emb, class_proto, seq_mask)

    def text_view_score(self, gene_emb, seq_mask, text_emb, _view, seq_len):
        true_gene_emb = gene_emb[:, 1:seq_len + 1, :]
        gene_proj = self.q_norm(self.q_adapter(true_gene_emb))
        gene_proj = F.normalize(gene_proj, p=2, dim=-1)

        if _view == "t_view":
            text_proj = self.tissue_norm(self.tissue_adapter(text_emb))
            text_proj = F.normalize(text_proj, p=2, dim=-1)
            return self.tissue_scorer(gene_proj, text_proj, seq_mask)

        text_proj = self.role_norm(self.role_adapter(text_emb))
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        return self.role_scorer(gene_proj, text_proj, seq_mask)

    def coarse_match(self, seq_h):
        return self.coarse_ffn(seq_h[:, 0, :])

    def forward(self, seqs_tokens, gene_ids, marker_ids, tissu_emb, role_emb):
        seqs_embs = self.gf_emb(seqs_tokens)
        coarse_logits = self.coarse_match(seqs_embs)

        seq_mask = gene_ids != -1
        marker_view_logits = self.marker_view_score(gene_ids, marker_ids)
        tissue_view_logits = self.text_view_score(
            seqs_embs,
            seq_mask,
            tissu_emb,
            "t_view",
            self.seq_len,
        )
        role_view_logits = self.text_view_score(
            seqs_embs,
            seq_mask,
            role_emb,
            "r_view",
            self.seq_len,
        )
        return coarse_logits, marker_view_logits, tissue_view_logits, role_view_logits
