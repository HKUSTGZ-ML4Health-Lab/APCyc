import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.format import VOCAB
from models.dyMEAN.modules.am_egnn import AMEGNN
from models.dyMEAN.nn_utils import SeparatedAminoAcidFeature
from utils.nn_utils import variadic_meshgrid
from .pairs import ValidPairGenerator

try:
    from torch_geometric.nn import global_mean_pool
except Exception:  # pragma: no cover - optional dependency
    global_mean_pool = None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length]


class PocketEncoderDyMEAN(nn.Module):
    def __init__(
        self,
        embed_size: int = 256,
        hidden_size: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        channel_nf: int = 64,
        radial_nf: int = 128,
        max_position: int = 2048,
        backbone_only: bool = False,
        edge_mode: str = "knn",
        edge_k: int = 16,
        edge_radius: float = 8.0,
        relative_position: bool = True,
        fix_atom_weights: bool = False,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.backbone_only = backbone_only
        self.edge_mode = edge_mode
        self.edge_k = edge_k
        self.edge_radius = edge_radius

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size,
            atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=fix_atom_weights,
            backbone_only=backbone_only,
        )

        self.gnn = AMEGNN(
            embed_size,
            hidden_size,
            hidden_size,
            n_channel=4 if backbone_only else VOCAB.MAX_ATOM_NUMBER,
            channel_nf=atom_embed_size,
            radial_nf=radial_nf,
            in_edge_nf=0,
            n_layers=n_layers,
            residual=True,
            dropout=dropout,
            dense=False,
        )

    def _build_edges(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.edge_mode == "full":
            total = int(lengths.sum().item())
            row, col = variadic_meshgrid(
                input1=torch.arange(total, device=device),
                size1=lengths,
                input2=torch.arange(total, device=device),
                size2=lengths,
            )
            return torch.stack([row, col], dim=0)

        ca_idx = VOCAB.ca_channel_idx
        coords = X[:, ca_idx]
        edges = []
        offset = 0
        for n in lengths.tolist():
            n = int(n)
            if n <= 0:
                continue
            if n == 1:
                row = torch.tensor([offset], device=device)
                col = torch.tensor([offset], device=device)
                edges.append(torch.stack([row, col], dim=0))
                offset += 1
                continue
            sub = coords[offset : offset + n]
            dist = torch.cdist(sub, sub, p=2)
            dist.fill_diagonal_(float("inf"))
            if self.edge_mode == "radius":
                mask = dist <= self.edge_radius
                row, col = mask.nonzero(as_tuple=True)
                if row.numel() == 0:
                    k = min(self.edge_k, n - 1)
                    topk = dist.topk(k, largest=False).indices
                    row = torch.arange(n, device=device).unsqueeze(1).repeat(1, k).reshape(-1)
                    col = topk.reshape(-1)
            else:
                k = min(self.edge_k, n - 1)
                topk = dist.topk(k, largest=False).indices
                row = torch.arange(n, device=device).unsqueeze(1).repeat(1, k).reshape(-1)
                col = topk.reshape(-1)
            edges.append(torch.stack([row + offset, col + offset], dim=0))
            offset += n
        if not edges:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        position_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        H_0, (atom_embeddings, atom_weights) = self.aa_feature(S, position_ids)
        edges = self._build_edges(X, lengths, X.device)
        H, _ = self.gnn(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        return H

    def load_pretrained(self, ckpt_path: str, mode: str = "partial") -> Dict[str, int]:
        if ckpt_path is None:
            return {"loaded": 0, "missing": 0, "mismatch": 0}
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt if isinstance(ckpt, dict) else {}

        target = self.state_dict()
        loaded = {}
        mismatch = 0
        for key, val in state_dict.items():
            clean = key
            for prefix in (
                "module.",
                "encoder.",
                "backbone.",
                "model.",
                "autoencoder.",
                "autoencoder.encoder.",
                "Ephi.",
            ):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
            if clean in target and target[clean].shape == val.shape:
                loaded[clean] = val
            elif clean in target:
                mismatch += 1

        missing = len(set(target.keys()) - set(loaded.keys()))
        if mode == "full" and missing == 0 and mismatch == 0:
            self.load_state_dict(loaded, strict=True)
        else:
            self.load_state_dict(loaded, strict=False)
        return {"loaded": len(loaded), "missing": missing, "mismatch": mismatch}

    def freeze(self, mode: str = "none", freeze_layers: int = 0) -> List[str]:
        frozen = []
        if mode == "all":
            for name, param in self.named_parameters():
                param.requires_grad = False
                frozen.append(name)
        elif mode == "partial":
            for name, param in self.residue_embed.named_parameters():
                param.requires_grad = False
                frozen.append(f"residue_embed.{name}")
            for name, param in self.channel_mlp.named_parameters():
                param.requires_grad = False
                frozen.append(f"channel_mlp.{name}")
            n_freeze = min(freeze_layers, self.n_layers)
            for i in range(n_freeze):
                layer = self.gnn._modules.get(f"gcl_{i}")
                if layer is None:
                    continue
                for name, param in layer.named_parameters():
                    param.requires_grad = False
                    frozen.append(f"gnn.gcl_{i}.{name}")
        return frozen


class TypeHead(nn.Module):
    def __init__(self, d_model: int, num_types: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_types),
        )

    def forward(self, hb: torch.Tensor) -> torch.Tensor:
        return self.net(hb)


class SiteHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        num_types: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        type_condition: str = "add",
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.type_condition = type_condition
        self.use_cross_attention = use_cross_attention

        self.pos_embed = nn.Embedding(max_len, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.type_embed = nn.Embedding(num_types, d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pair_proj = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.pair_proj)

        if type_condition == "film":
            self.type_film = nn.Linear(d_model, d_model * 2)
        else:
            self.type_film = None

    def _peptide_embed(self, n: int, device: torch.device) -> torch.Tensor:
        if n > self.max_len:
            raise ValueError(f"Peptide length {n} exceeds max_len {self.max_len}")
        positions = torch.arange(n, device=device)
        return self.pos_embed(positions) + self.pos_enc(n).to(device)

    def _apply_type_condition(self, h: torch.Tensor, c_idx: int) -> torch.Tensor:
        type_vec = self.type_embed(torch.tensor(c_idx, device=h.device))
        if self.type_condition == "film":
            gamma, beta = self.type_film(type_vec).chunk(2, dim=-1)
            return h * (1 + gamma) + beta
        return h + type_vec

    def forward(
        self,
        pocket_nodes: torch.Tensor,
        c_idx: int,
        n: int,
        pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        h = self._peptide_embed(n, pocket_nodes.device)
        if self.use_cross_attention and pocket_nodes.numel() > 0:
            h_attn, _ = self.cross_attn(
                h.unsqueeze(0), pocket_nodes.unsqueeze(0), pocket_nodes.unsqueeze(0)
            )
            h = h_attn.squeeze(0)
        h = self._apply_type_condition(h, c_idx)
        if pairs is None:
            scores = torch.einsum("id,dk,jk->ij", h, self.pair_proj, h)
            return scores
        if len(pairs) == 0:
            return h.new_empty((0,))
        pair_idx = torch.tensor(pairs, device=h.device, dtype=torch.long)
        h_i = h[pair_idx[:, 0]]
        h_j = h[pair_idx[:, 1]]
        logits = torch.einsum("md,dk,mk->m", h_i, self.pair_proj, h_j)
        return logits


class RouterModel(nn.Module):
    def __init__(
        self,
        type_list: List[str],
        embed_size: int = 256,
        hidden_size: int = 256,
        encoder_layers: int = 4,
        encoder_dropout: float = 0.1,
        channel_nf: int = 64,
        radial_nf: int = 128,
        max_position: int = 2048,
        backbone_only: bool = False,
        edge_mode: str = "knn",
        edge_k: int = 16,
        edge_radius: float = 8.0,
        max_peptide_len: int = 128,
        site_heads: int = 4,
        site_dropout: float = 0.1,
        type_condition: str = "add",
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.type_list = type_list
        self.num_types = len(type_list)
        self.encoder = PocketEncoderDyMEAN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            n_layers=encoder_layers,
            dropout=encoder_dropout,
            channel_nf=channel_nf,
            radial_nf=radial_nf,
            max_position=max_position,
            backbone_only=backbone_only,
            edge_mode=edge_mode,
            edge_k=edge_k,
            edge_radius=edge_radius,
        )
        self.type_head = TypeHead(hidden_size, self.num_types, dropout=encoder_dropout)
        self.site_head = SiteHead(
            hidden_size,
            max_peptide_len,
            self.num_types,
            n_heads=site_heads,
            dropout=site_dropout,
            type_condition=type_condition,
            use_cross_attention=use_cross_attention,
        )

    def encode_pocket(self, batch_graph) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_mean_pool is None:
            raise RuntimeError("torch_geometric is required for pooling.")
        lengths = torch.bincount(batch_graph.batch, minlength=batch_graph.num_graphs)
        hb_nodes = self.encoder(
            batch_graph.X, batch_graph.S, batch_graph.position_ids, lengths
        )
        hb = global_mean_pool(hb_nodes, batch_graph.batch)
        return hb_nodes, hb

    def predict_type(self, hb: torch.Tensor) -> torch.Tensor:
        return self.type_head(hb)

    def predict_site_logits(
        self,
        hb_nodes: torch.Tensor,
        batch_graph,
        peptide_lengths: List[int],
        c_indices: List[int],
        pair_gen: ValidPairGenerator,
    ) -> List[Tuple[List[Tuple[int, int]], torch.Tensor]]:
        outputs = []
        for idx in range(len(peptide_lengths)):
            mask = batch_graph.batch == idx
            pocket_nodes = hb_nodes[mask]
            c_idx = int(c_indices[idx])
            n = int(peptide_lengths[idx])
            pairs = pair_gen.get_pairs(c_idx, n)
            logits = self.site_head(pocket_nodes, c_idx, n, pairs)
            outputs.append((pairs, logits))
        return outputs

    @torch.no_grad()
    def infer_topk(
        self,
        batch_graph,
        peptide_lengths: List[int],
        pair_gen: ValidPairGenerator,
        k_c: int = 3,
        k_p: int = 5,
    ) -> List[List[Tuple[str, int, int, float]]]:
        hb_nodes, hb = self.encode_pocket(batch_graph)
        logits_c = self.predict_type(hb)
        p_c = F.softmax(logits_c, dim=-1)
        results = []
        for b_idx in range(hb.size(0)):
            type_scores = p_c[b_idx]
            top_types = torch.topk(type_scores, min(k_c, self.num_types))
            sample_results = []
            for c_idx in top_types.indices.tolist():
                n = int(peptide_lengths[b_idx])
                mask = batch_graph.batch == b_idx
                pocket_nodes = hb_nodes[mask]
                pairs = pair_gen.get_pairs(c_idx, n)
                logits = self.site_head(pocket_nodes, c_idx, n, pairs)
                if logits.numel() == 0:
                    continue
                p_ij = F.softmax(logits, dim=-1)
                top_pairs = torch.topk(p_ij, min(k_p, p_ij.numel()))
                for p_score, p_idx in zip(top_pairs.values.tolist(), top_pairs.indices.tolist()):
                    i, j = pairs[p_idx]
                    score = float(type_scores[c_idx] * p_score)
                    sample_results.append((self.type_list[c_idx], i, j, score))
            results.append(sample_results)
        return results
