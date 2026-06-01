#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import torch
import torch.nn as nn


class PocketCrossAttn(nn.Module):
    """
    Single-head cross attention: peptide queries attend to pocket keys/values.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, H: torch.Tensor, mask_generate: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        H: [N, hidden_dim]
        mask_generate: [N] (True for peptide)
        batch_ids: [N]
        """
        H_out = H.clone()
        bs = int(batch_ids.max().item()) + 1
        for b in range(bs):
            pep_idx = torch.nonzero((batch_ids == b) & mask_generate).view(-1)
            pocket_idx = torch.nonzero((batch_ids == b) & (~mask_generate)).view(-1)
            if pep_idx.numel() == 0 or pocket_idx.numel() == 0:
                continue
            Q = self.q_proj(H[pep_idx])
            K = self.k_proj(H[pocket_idx])
            V = self.v_proj(H[pocket_idx])
            attn = torch.softmax((Q @ K.t()) * self.scale, dim=-1)
            attn = self.dropout(attn)
            update = attn @ V
            H_out[pep_idx] = H_out[pep_idx] + self.out_proj(update)
        return H_out
