#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn


def cyclic_lambda(t: torch.Tensor, num_steps: int, max_lambda: float, schedule: str = 'linear') -> torch.Tensor:
    if num_steps <= 0:
        return torch.zeros_like(t, dtype=torch.float)
    t = t.float()
    if schedule == 'linear':
        return max_lambda * (1.0 - t / float(num_steps))
    if schedule == 'cosine':
        return max_lambda * (1.0 - torch.cos(0.5 * torch.pi * (1.0 - t / float(num_steps))))
    raise ValueError(f'Unknown schedule {schedule}')


class EdgeFeatureInjector(nn.Module):
    def __init__(
        self,
        pair_struct_dim: int,
        struct_proj_dim: int,
        edge_type_dim: int = 8,
        edge_type_vocab_size: int = 3,
        covalent_edge_type_id: int = 3,
        separate_params: bool = False,
        include_type: bool = True,
        include_pair_score: bool = True,
        include_cycle_edge: bool = True,
        include_edge_type: bool = True,
    ) -> None:
        super().__init__()
        self.include_type = include_type
        self.include_pair_score = include_pair_score
        self.include_cycle_edge = include_cycle_edge
        self.include_edge_type = include_edge_type
        self.separate_params = separate_params
        self.covalent_edge_type_id = covalent_edge_type_id
        self.struct_proj = nn.Linear(pair_struct_dim, struct_proj_dim)
        if separate_params:
            self.struct_proj_covalent = nn.Linear(pair_struct_dim, struct_proj_dim)
        if include_edge_type:
            self.edge_type_embed = nn.Embedding(edge_type_vocab_size, edge_type_dim)

    def forward(
        self,
        edges: torch.Tensor,
        batch_ids: torch.Tensor,
        pair_struct: torch.Tensor,
        s_ij: torch.Tensor,
        p_c: torch.Tensor,
        lambda_cyc: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        hard_pair: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            edges: [2, E]
            batch_ids: [N]
            pair_struct: [N, N, d_struct]
            s_ij: [N, N]
            p_c: [bs, 3]
            lambda_cyc: [bs]
            edge_type: [N, N] with 0/1/2 or None
            hard_pair: [bs, 2] global indices or None
        Returns:
            edge_attr: [E, d_edge]
        """
        row, col = edges
        bs = p_c.size(0)
        edge_batch = batch_ids[row]

        struct_feat = self.struct_proj(pair_struct[row, col])
        edge_type_ids = None
        if edge_type is not None:
            if edge_type.dim() == 2:
                edge_type_ids = edge_type[row, col]
            else:
                edge_type_ids = edge_type
        if self.separate_params and edge_type_ids is not None:
            cov_mask = edge_type_ids == self.covalent_edge_type_id
            if cov_mask.any():
                cov_feat = self.struct_proj_covalent(pair_struct[row, col][cov_mask])
                struct_feat = struct_feat.clone()
                struct_feat[cov_mask] = cov_feat
        feats = [struct_feat]

        if self.include_pair_score:
            s_edge = s_ij[row, col] * lambda_cyc[edge_batch]
            feats.append(s_edge.unsqueeze(-1))

        if self.include_type:
            feats.append(p_c[edge_batch])

        if self.include_edge_type and edge_type_ids is not None:
            feats.append(self.edge_type_embed(edge_type_ids))

        if self.include_cycle_edge:
            if hard_pair is None:
                cycle_flag = torch.zeros_like(edge_batch, dtype=struct_feat.dtype)
            else:
                cycle_flag = torch.zeros_like(edge_batch, dtype=struct_feat.dtype)
                for b in range(bs):
                    i, j = hard_pair[b].tolist()
                    mask_ij = (row == i) & (col == j)
                    mask_ji = (row == j) & (col == i)
                    cycle_flag[mask_ij | mask_ji] = 1.0
            feats.append(cycle_flag.unsqueeze(-1))

        return torch.cat(feats, dim=-1)


class BiasGateInjector(nn.Module):
    def __init__(self, pair_struct_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.bias_proj = nn.Linear(pair_struct_dim, hidden_dim)
        self.gate_proj = nn.Linear(pair_struct_dim, hidden_dim)

    def forward(
        self,
        edges: torch.Tensor,
        batch_ids: torch.Tensor,
        pair_struct: torch.Tensor,
        s_ij: torch.Tensor,
        lambda_b: torch.Tensor,
        lambda_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_bias: [E, hidden_dim]
            edge_gate: [E, hidden_dim]
        """
        row, col = edges
        edge_batch = batch_ids[row]
        b_base = self.bias_proj(pair_struct[row, col])
        g_base = torch.sigmoid(self.gate_proj(pair_struct[row, col]))

        b_cyc = s_ij[row, col].unsqueeze(-1) * lambda_b[edge_batch].unsqueeze(-1)
        g_cyc = torch.sigmoid(s_ij[row, col].unsqueeze(-1) * lambda_g[edge_batch].unsqueeze(-1))

        edge_bias = b_base + b_cyc
        edge_gate = g_base * g_cyc
        return edge_bias, edge_gate
