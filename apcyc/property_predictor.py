#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from models.dyMEAN.modules.am_egnn import AM_E_GCL


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding for scalar timesteps."""
    if t.dim() == 0:
        t = t.view(1)
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PropertyAMEGNN(nn.Module):
    """AMEGNN-style trunk with FiLM modulation per layer."""

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        n_channel: int,
        channel_nf: int,
        radial_nf: int,
        in_edge_nf: int = 0,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_rbf: int = 0,
        cutoff: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.linear_in = nn.Linear(in_node_nf, hidden_nf)
        self.layers = nn.ModuleList([
            AM_E_GCL(
                hidden_nf, hidden_nf, hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, residual=True, dropout=dropout, n_rbf=n_rbf, cutoff=cutoff
            )
            for _ in range(n_layers)
        ])
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor,
        channel_attr: torch.Tensor,
        channel_weights: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
        edge_gate: Optional[torch.Tensor] = None,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = self.linear_in(h)
        h = self.dropout(h)
        gamma, beta = film if film is not None else (None, None)
        for layer in self.layers:
            h, x = layer(
                h, edges, x, channel_attr, channel_weights,
                edge_attr=edge_attr, edge_bias=edge_bias, edge_gate=edge_gate
            )
            if gamma is not None:
                h = gamma * h + beta
        h = self.dropout(h)
        h = self.linear_out(h)
        return h


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.proj(h).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        counts = scatter_sum(mask.float(), batch_ids, dim=0)
        valid = counts > 0
        max_scores = scatter_max(scores, batch_ids, dim=0)[0]
        max_scores = torch.where(valid, max_scores, torch.zeros_like(max_scores))
        scores_exp = torch.exp(scores - max_scores[batch_ids])
        scores_exp = torch.where(mask, scores_exp, torch.zeros_like(scores_exp))
        denom = scatter_sum(scores_exp, batch_ids, dim=0)
        denom = torch.where(valid, denom, torch.ones_like(denom))
        weights = scores_exp / (denom[batch_ids] + 1e-8)
        pooled = scatter_sum(weights[:, None] * h, batch_ids, dim=0)
        return pooled


class PropertyPredictor(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        n_channel: int,
        n_layers: int = 3,
        edge_size: int = 0,
        n_rbf: int = 0,
        cutoff: float = 1.0,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
        n_properties: int = 7,
        edge_type_vocab_size: int = 4,
        pos_embed_size: int = 4096,
        stats_topk: int = 5,
        use_attn_pool: bool = True,
        use_pair_stats: bool = True,
        mlp_hidden_size: int = 0,
    ) -> None:
        super().__init__()
        self.n_properties = n_properties
        self.stats_topk = stats_topk
        self.time_embed_dim = time_embed_dim
        self.edge_size = edge_size
        self.use_attn_pool = use_attn_pool
        self.use_pair_stats = use_pair_stats
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_embed = nn.Embedding(pos_embed_size, latent_size)
        self.atom_embed_size = hidden_size // 4
        mlp_hidden = mlp_hidden_size if mlp_hidden_size > 0 else hidden_size

        in_node_nf = latent_size + latent_size
        atom_embed_size = self.atom_embed_size
        in_edge_nf = edge_size

        self.trunk = PropertyAMEGNN(
            in_node_nf, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=in_edge_nf, n_layers=n_layers,
            dropout=dropout, n_rbf=n_rbf, cutoff=cutoff
        )
        self.film = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.attn_pool = AttentionPool(hidden_size) if use_attn_pool else None
        pool_dim = hidden_size * (6 if use_attn_pool else 4)
        stats_dim = 3 if use_pair_stats else 0
        self.shared_mlp = nn.Sequential(
            nn.Linear(pool_dim + stats_dim + 1 + time_embed_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(mlp_hidden, 1) for _ in range(n_properties)
        ])

    def _pool_group(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        masked = h.masked_fill(~mask[:, None], 0.0)
        sum_pool = scatter_sum(masked, batch_ids, dim=0)
        counts = scatter_sum(mask.float(), batch_ids, dim=0).clamp_min(1.0)
        mean = sum_pool / counts[:, None]
        max_pool = scatter_max(h.masked_fill(~mask[:, None], float('-inf')), batch_ids, dim=0)[0]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        if not self.use_attn_pool:
            return torch.cat([mean, max_pool], dim=-1)
        attn = self.attn_pool(h, mask, batch_ids)
        return torch.cat([attn, mean, max_pool], dim=-1)

    def _pair_stats(
        self,
        s_ij: Optional[torch.Tensor],
        pep_indices: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if s_ij is None:
            bs = len(pep_indices)
            return torch.zeros(bs, 3, device=next(self.parameters()).device)
        stats = []
        for pep_idx in pep_indices:
            if pep_idx.numel() == 0:
                stats.append(torch.zeros(3, device=s_ij.device))
                continue
            s_pp = s_ij[pep_idx][:, pep_idx].contiguous().view(-1)
            max_val = s_pp.max()
            k = min(self.stats_topk, s_pp.numel())
            topk = torch.topk(s_pp, k=k).values.mean()
            p = s_pp / (s_pp.sum() + 1e-8)
            entropy = -(p * (p + 1e-8).log()).sum()
            stats.append(torch.stack([max_val, topk, entropy]))
        return torch.stack(stats, dim=0)

    def forward(
        self,
        H_noisy: torch.Tensor,
        X_noisy: torch.Tensor,
        mask_generate: torch.Tensor,
        atom_mask: torch.Tensor,
        position_ids: torch.Tensor,
        ctx_edges: torch.Tensor,
        inter_edges: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_gate: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        pair_struct: Optional[torch.Tensor] = None,
        pair_cyc: Optional[torch.Tensor] = None,
        type_probs: Optional[torch.Tensor] = None,
        s_ij: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        batch_ids: Optional[torch.Tensor] = None,
        atom_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch_ids is None:
            batch_ids = torch.zeros(H_noisy.size(0), device=H_noisy.device, dtype=torch.long)
        if t is None:
            t = torch.zeros(batch_ids.max() + 1, device=H_noisy.device, dtype=H_noisy.dtype)
        if t.dim() == 0:
            t = t.view(1)
        t_embed = sinusoidal_time_embedding(t.to(H_noisy.dtype), self.time_embed_dim)
        gamma, beta = self.film(t_embed).chunk(2, dim=-1)
        gamma = gamma[batch_ids]
        beta = beta[batch_ids]

        pos_embed = self.pos_embed(position_ids.clamp_max(self.pos_embed.num_embeddings - 1))
        h_in = torch.cat([H_noisy, pos_embed], dim=-1)

        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        if edge_type is None:
            edge_type = torch.zeros(edges.size(1), device=edges.device, dtype=torch.long)
        if edge_attr is None:
            edge_attr = torch.zeros(
                edges.size(1), self.edge_size,
                device=H_noisy.device, dtype=H_noisy.dtype
            )

        if atom_embeddings is None:
            atom_embeddings = torch.zeros(
                H_noisy.size(0), atom_mask.size(1), self.atom_embed_size,
                device=H_noisy.device, dtype=H_noisy.dtype
            )

        h = self.trunk(
            h_in, X_noisy, edges,
            channel_attr=atom_embeddings,
            channel_weights=atom_mask.float(),
            edge_attr=edge_attr,
            edge_bias=edge_bias,
            edge_gate=edge_gate,
            film=(gamma, beta),
        )

        pep_mask = mask_generate
        poc_mask = ~mask_generate
        g_pep = self._pool_group(h, pep_mask, batch_ids)
        g_poc = self._pool_group(h, poc_mask, batch_ids)

        if self.use_pair_stats:
            stats = self._pair_stats(s_ij, self._build_pep_indices(mask_generate, batch_ids))
        else:
            stats_rows = int(batch_ids.max().item()) + 1
            stats = torch.zeros(stats_rows, 0, device=batch_ids.device)
        type_ent = torch.zeros(stats.size(0), 1, device=stats.device)
        if type_probs is not None:
            type_ent = -(type_probs * (type_probs + 1e-8).log()).sum(dim=-1, keepdim=True)

        t_embed_nodes = t_embed if t_embed.dim() == 2 else t_embed.view(1, -1)
        g = torch.cat([g_pep, g_poc, stats, type_ent, t_embed_nodes], dim=-1)
        u = self.shared_mlp(g)
        outs = [head(u) for head in self.heads]
        y_hat = torch.cat(outs, dim=-1)
        return y_hat

    @staticmethod
    def _build_pep_indices(mask_generate: torch.Tensor, batch_ids: torch.Tensor):
        pep_indices = []
        bs = int(batch_ids.max().item()) + 1
        for b in range(bs):
            idx = torch.nonzero((batch_ids == b) & mask_generate).view(-1)
            pep_indices.append(idx)
        return pep_indices
