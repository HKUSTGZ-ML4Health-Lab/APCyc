#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apcyc.property_predictor import PropertyPredictor


def parse_guidance_w(w: Optional[Iterable[float]], k: int, device: torch.device) -> torch.Tensor:
    if w is None:
        return torch.ones(k, device=device)
    if isinstance(w, torch.Tensor):
        return w.to(device=device, dtype=torch.float)
    w_list = list(w)
    if len(w_list) != k:
        raise ValueError(f'guidance_w length {len(w_list)} != k={k}')
    return torch.tensor(w_list, device=device, dtype=torch.float)


def clip_guidance_w(w: torch.Tensor, w_min: Optional[float], w_max: Optional[float]) -> torch.Tensor:
    if w_min is None and w_max is None:
        return w
    return torch.clamp(w, min=w_min, max=w_max)


def guidance_lambda(t: int, start: int, end: int, lambda_max: float) -> float:
    if t > start:
        return 0.0
    if t <= end:
        return float(lambda_max)
    if start == end:
        return float(lambda_max)
    ratio = (start - t) / max(start - end, 1)
    return float(lambda_max * ratio)


def sigma2_from_transition(trans, t: int, batch_ids: torch.Tensor, expand_shape: Tuple[int, ...]) -> torch.Tensor:
    sigma = trans.var_sched.sigmas[t][batch_ids].view(*expand_shape)
    return sigma * sigma


def grad_norm_balance(
    weights: torch.Tensor,
    norms: torch.Tensor,
    eps: float = 1e-6,
    ref: str = 'mean',
) -> torch.Tensor:
    valid = norms > 0
    if not torch.any(valid):
        return weights
    if ref == 'median':
        c = torch.median(norms[valid])
    else:
        c = torch.mean(norms[valid])
    return weights * (c / (norms + eps))


def energy_from_pred(y_hat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return -torch.sum(y_hat * w[None, :], dim=-1).mean()


class MultiPropertyPredictor(nn.Module):
    def __init__(self, predictors: Sequence[nn.Module], keys: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.predictors = nn.ModuleList(predictors)
        self.property_keys = list(keys) if keys is not None else None
        self.n_properties = len(self.predictors)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = []
        for pred in self.predictors:
            y = pred(*args, **kwargs)
            if y.dim() == 0:
                y = y.view(1, 1)
            elif y.dim() == 1:
                y = y.view(-1, 1)
            outputs.append(y)
        if not outputs:
            raise ValueError('No predictors provided for MultiPropertyPredictor.')
        return torch.cat(outputs, dim=-1)


def load_property_predictor(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    cfg = ckpt.get('model_cfg', {}) if isinstance(ckpt, dict) else {}
    from apcyc.property_predictor import PropertyPredictor
    predictor = PropertyPredictor(**cfg)
    predictor.load_state_dict(state_dict, strict=False)
    if isinstance(ckpt, dict):
        if 'property_mean' in ckpt and 'property_std' in ckpt:
            predictor.property_mean = torch.tensor(ckpt['property_mean'])
            predictor.property_std = torch.tensor(ckpt['property_std'])
        if 'property_keys' in ckpt:
            predictor.property_keys = list(ckpt['property_keys'])
    predictor.to(device)
    predictor.eval()
    return predictor


def load_property_predictors(ckpt_paths, device: torch.device, keys: Optional[Sequence[str]] = None):
    predictors = [load_property_predictor(path, device) for path in ckpt_paths]
    resolved_keys = None
    if keys is not None:
        resolved_keys = list(keys)
    else:
        collected = []
        for pred in predictors:
            pred_keys = getattr(pred, 'property_keys', None)
            collected.append(pred_keys[0] if pred_keys else None)
        if any(k is not None for k in collected):
            resolved_keys = collected
    wrapper = MultiPropertyPredictor(predictors, keys=resolved_keys)
    wrapper.to(device)
    wrapper.eval()
    return wrapper
