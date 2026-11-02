#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from data.format import VOCAB
from .constants import CYCLE_TYPES, TYPE2IDX
from .chemistry import is_iso_pair


def _resolve_symbol(symbol: str) -> str:
    if hasattr(VOCAB, 'resolve_symbol'):
        return VOCAB.resolve_symbol(symbol)
    if hasattr(VOCAB, 'alias_symbol'):
        return VOCAB.alias_symbol.get(symbol, symbol)
    return symbol


def _upper_triangle_mask(L: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((L, L), device=device, dtype=torch.bool)
    return torch.triu(mask, diagonal=1)


def _pep_symbols(S_pep: torch.Tensor) -> List[str]:
    symbols = []
    for i in S_pep.tolist():
        symbol = VOCAB.idx_to_symbol(int(i))
        symbols.append(_resolve_symbol(symbol))
    return symbols


def _valid_mask_ht(L: int, device: torch.device, allow_alt: bool = False, min_end: int = 1) -> torch.Tensor:
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    if not allow_alt:
        mask[0, L - 1] = True
        return mask
    # Allow near-end alternatives if enabled.
    i_candidates = [0]
    j_candidates = list(range(max(min_end, 1), L))
    for i in i_candidates:
        for j in j_candidates:
            if i < j:
                mask[i, j] = True
    return mask


def _valid_mask_ss(symbols: List[str], min_sep: int, device: torch.device) -> torch.Tensor:
    L = len(symbols)
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    is_cys = [s == 'C' for s in symbols]
    for i in range(L):
        if not is_cys[i]:
            continue
        for j in range(i + min_sep, L):
            if is_cys[j]:
                mask[i, j] = True
    return mask


def _valid_mask_iso(symbols: List[str], device: torch.device, paper=False) -> torch.Tensor:
    L = len(symbols)
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    is_lys = [s == 'K' for s in symbols]
    is_asp = [s in ('D', 'N') if paper else s == 'D' for s in symbols]
    for i in range(L):
        for j in range(i + 1, L):
            compatible = is_iso_pair(symbols[i], symbols[j]) if paper else (
                (is_lys[i] and is_asp[j]) or (is_asp[i] and is_lys[j]))
            if compatible:
                mask[i, j] = True
    return mask


class CyclizationHeads(nn.Module):
    def __init__(
        self,
        d_cyc: int,
        type_hidden: int = 128,
        pair_hidden: int = 128,
        min_sep_all: int = 1,
        min_sep_ss: int = 3,
        allow_ht_alt_pairs: bool = False,
        ht_alt_min_end: int = 1,
        mask_mode: str = 'teacher',
        relax_residue_constraints: bool = True,
        eps: float = 1e-8,
        chemistry_profile: str = 'legacy',
    ) -> None:
        super().__init__()
        self.min_sep_ss = min_sep_ss
        self.min_sep_all = min_sep_all
        self.allow_ht_alt_pairs = allow_ht_alt_pairs
        self.ht_alt_min_end = ht_alt_min_end
        self.mask_mode = mask_mode
        self.relax_residue_constraints = relax_residue_constraints
        self.eps = eps
        if chemistry_profile not in {'legacy', 'paper'}:
            raise ValueError('Unknown chemistry_profile')
        self.chemistry_profile = chemistry_profile

        self.type_mlp = nn.Sequential(
            nn.Linear(d_cyc, type_hidden),
            nn.SiLU(),
            nn.Linear(type_hidden, len(CYCLE_TYPES)),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_cyc, pair_hidden),
            nn.SiLU(),
            nn.Linear(pair_hidden, 1),
        )

    def _valid_mask_by_type(self, symbols: List[str], type_idx: int, device: torch.device) -> torch.Tensor:
        if CYCLE_TYPES[type_idx] == 'HT':
            return _valid_mask_ht(len(symbols), device, self.allow_ht_alt_pairs, self.ht_alt_min_end)
        if CYCLE_TYPES[type_idx] == 'SS':
            mask = _valid_mask_ss(symbols, self.min_sep_ss, device)
            if self.relax_residue_constraints and not mask.any():
                mask = _upper_triangle_mask(len(symbols), device)
            return mask
        if CYCLE_TYPES[type_idx] == 'ISO':
            mask = _valid_mask_iso(symbols, device, paper=self.chemistry_profile == 'paper')
            if self.relax_residue_constraints and not mask.any():
                mask = _upper_triangle_mask(len(symbols), device)
            return mask
        raise ValueError(f'Unknown type index {type_idx}')

    def forward_single(
        self,
        pair_cyc: torch.Tensor,
        S_pep: torch.Tensor,
        type_label: Optional[int] = None,
        type_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pair_cyc: [L, L, d_cyc]
        Returns:
            type_logits: [3]
            pair_logits: [L, L]
            s_ij: [L, L]
        """
        device = pair_cyc.device
        L = pair_cyc.size(0)
        symbols = _pep_symbols(S_pep)

        pooled = pair_cyc.mean(dim=(0, 1))
        if type_logits is None:
            type_logits = self.type_mlp(pooled)
        if self.chemistry_profile == 'paper':
            available = torch.tensor([bool(self._valid_mask_by_type(symbols, i, device).any())
                                      for i in range(len(CYCLE_TYPES))], device=device)
            if type_label is not None and not bool(available[type_label]):
                raise ValueError('Ground-truth cyclization type has no chemically valid pair.')
            if not available.any():
                raise ValueError('No chemically valid cyclization topology for the decoded sequence.')
            # Finite sentinel avoids 0 * infinity in downstream diagnostics.
            type_logits = type_logits.masked_fill(~available, torch.finfo(type_logits.dtype).min)
        pair_logits = self.pair_mlp(pair_cyc).squeeze(-1)

        upper_mask = _upper_triangle_mask(L, device)

        if self.mask_mode == 'teacher':
            if type_label is None:
                pred_idx = int(torch.argmax(type_logits).item())
                valid_mask = self._valid_mask_by_type(symbols, pred_idx, device)
            else:
                valid_mask = self._valid_mask_by_type(symbols, type_label, device)
        elif self.mask_mode == 'predicted':
            pred_idx = int(torch.argmax(type_logits).item())
            valid_mask = self._valid_mask_by_type(symbols, pred_idx, device)
        elif self.mask_mode == 'mixture':
            probs = torch.softmax(type_logits, dim=-1)
            valid_mask = torch.zeros((L, L), device=device, dtype=pair_logits.dtype)
            for t_idx in range(len(CYCLE_TYPES)):
                mask_t = self._valid_mask_by_type(symbols, t_idx, device).float()
                valid_mask = valid_mask + probs[t_idx] * mask_t
        elif self.mask_mode in ('none', 'no_type', 'unconditional'):
            # Do not condition on cyclization type; allow all peptide pairs.
            valid_mask = upper_mask
        else:
            raise ValueError(f'Unknown mask_mode {self.mask_mode}')

        if self.min_sep_all > 1:
            sep_mask = torch.triu(
                torch.ones((L, L), device=device, dtype=torch.bool),
                diagonal=self.min_sep_all,
            )
            if valid_mask.dtype == torch.bool:
                valid_mask = valid_mask & sep_mask
            else:
                valid_mask = valid_mask * sep_mask.float()

        if valid_mask.dtype == torch.bool:
            valid_mask = valid_mask & upper_mask
            logits_masked = pair_logits.masked_fill(~valid_mask, float('-inf'))
        else:
            valid_mask = valid_mask * upper_mask.float()
            logits_masked = pair_logits + torch.log(valid_mask + self.eps)

        if not torch.isfinite(logits_masked).any():
            raise ValueError('No valid cyclization pair; refusing an all-masked softmax.')
        s_ij = torch.softmax(logits_masked.view(-1), dim=0).view(L, L)
        return type_logits, pair_logits, s_ij

    def forward(
        self,
        pair_cyc: torch.Tensor,
        pep_indices: List[torch.Tensor],
        S: torch.Tensor,
        type_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns batch-level tensors padded into full [N, N] for pair outputs.
        """
        device = pair_cyc.device
        N = pair_cyc.size(0)
        bs = len(pep_indices)
        type_logits_all = []
        pair_logits_full = pair_cyc.new_zeros((N, N))
        s_ij_full = pair_cyc.new_zeros((N, N))

        for b in range(bs):
            pep_idx = pep_indices[b]
            if pep_idx.numel() == 0:
                type_logits_all.append(torch.zeros(len(CYCLE_TYPES), device=device))
                continue
            pair_cyc_b = pair_cyc[pep_idx][:, pep_idx]
            S_pep = S[pep_idx]
            type_label = int(type_labels[b].item()) if type_labels is not None else None
            type_logits, pair_logits, s_ij = self.forward_single(
                pair_cyc_b, S_pep, type_label=type_label
            )
            type_logits_all.append(type_logits)
            pair_logits_full[pep_idx[:, None], pep_idx[None, :]] = pair_logits
            s_ij_full[pep_idx[:, None], pep_idx[None, :]] = s_ij

        return {
            'type_logits': torch.stack(type_logits_all, dim=0),
            'pair_logits_full': pair_logits_full,
            's_ij_full': s_ij_full,
        }
