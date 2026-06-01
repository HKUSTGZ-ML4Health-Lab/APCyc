#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.dyMEAN.modules.radial_basis import RadialBasis


def get_anchor_coords(
    X: torch.Tensor,
    mode: str = 'ca',
    ca_idx: int = 1,
    channel_idx: int = 0,
) -> torch.Tensor:
    """
    X: [N, n_channel, 3] or [N, 3]
    returns: [N, 3]
    """
    if X.dim() == 2:
        return X
    if mode == 'mean':
        return X.mean(dim=1)
    if mode == 'channel':
        idx = channel_idx if channel_idx < X.size(1) else 0
        return X[:, idx]
    if X.size(1) == 1:
        return X[:, 0]
    idx = ca_idx if ca_idx < X.size(1) else 0
    return X[:, idx]


class OnlinePairInit(nn.Module):
    """
    Online pair feature constructor (no persistent pair state).
    """

    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        pair_struct_dim: int,
        rbf_dim: int,
        rbf_cutoff: float,
        edge_type_dim: int = 16,
        hidden_dim: int = 256,
        use_sep: bool = False,
        max_sep: int = 64,
        sep_dim: int = 8,
        ca_idx: int = 1,
        anchor_mode: str = 'ca',
        anchor_channel_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.pair_struct_dim = pair_struct_dim
        self.use_sep = use_sep
        self.max_sep = max_sep
        self.ca_idx = ca_idx
        self.anchor_mode = anchor_mode
        self.anchor_channel_idx = anchor_channel_idx

        self.node_proj = nn.Linear(node_dim, pair_dim)
        self.node_ln = nn.LayerNorm(pair_dim)
        self.edge_type_embed = nn.Embedding(2, edge_type_dim)
        self.rbf = RadialBasis(rbf_dim, rbf_cutoff)

        sep_dim = sep_dim if use_sep else 0
        in_dim = pair_dim * 3 + rbf_dim + edge_type_dim + sep_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pair_dim),
        )

        if use_sep:
            self.sep_embed = nn.Embedding(max_sep + 1, sep_dim)

    def forward(
        self,
        node_feat: torch.Tensor,
        X: torch.Tensor,
        part_mask: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pair_base: [N, N, pair_dim]
            pair_struct: [N, N, pair_struct_dim]
            pair_cyc: [N, N, pair_dim - pair_struct_dim]
            dist: [N, N]
        """
        device = node_feat.device
        N = node_feat.size(0)
        phi = self.node_ln(self.node_proj(node_feat))  # [N, pair_dim]

        phi_i = phi[:, None, :].expand(N, N, -1)
        phi_j = phi[None, :, :].expand(N, N, -1)
        phi_mul = phi_i * phi_j

        anchor = get_anchor_coords(
            X,
            mode=self.anchor_mode,
            ca_idx=self.ca_idx,
            channel_idx=self.anchor_channel_idx,
        )
        dist = torch.cdist(anchor, anchor, p=2)

        if batch_ids is not None:
            same_batch = (batch_ids[:, None] == batch_ids[None, :]).to(dist.dtype)
            dist = dist * same_batch

        rbf = self.rbf(dist.view(-1)).view(N, N, -1)

        edge_type = (part_mask[:, None] != part_mask[None, :]).long()
        edge_emb = self.edge_type_embed(edge_type)

        feats = [phi_i, phi_j, phi_mul, rbf, edge_emb]

        if self.use_sep and position_ids is not None:
            sep = (position_ids[:, None] - position_ids[None, :]).abs().clamp(max=self.max_sep)
            sep_emb = self.sep_embed(sep)
            feats.append(sep_emb)

        feat_ij = torch.cat(feats, dim=-1)
        pair_base = self.mlp(feat_ij)
        pair_struct = pair_base[..., : self.pair_struct_dim]
        pair_cyc = pair_base[..., self.pair_struct_dim :]
        return pair_base, pair_struct, pair_cyc, dist
