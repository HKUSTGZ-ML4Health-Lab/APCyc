#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_array(values: Iterable[Optional[float]]) -> np.ndarray:
    vals = [v for v in values if v is not None]
    if not vals:
        return np.array([], dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    arr = _safe_array(values)
    if arr.size == 0:
        return {
            'mean': None,
            'std': None,
            'p10': None,
            'p50': None,
            'p90': None,
            'med': None,
            'min': None,
        }
    p50 = float(np.percentile(arr, 50))
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'p10': float(np.percentile(arr, 10)),
        'p50': p50,
        'p90': float(np.percentile(arr, 90)),
        'med': p50,
        'min': float(arr.min()),
    }


def best_of_k(values: Iterable[Optional[float]], higher_better: bool) -> Optional[float]:
    arr = _safe_array(values)
    if arr.size == 0:
        return None
    return float(arr.max()) if higher_better else float(arr.min())


def mean_over_valid(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = _safe_array(values)
    if arr.size == 0:
        return None
    return float(arr.mean())
