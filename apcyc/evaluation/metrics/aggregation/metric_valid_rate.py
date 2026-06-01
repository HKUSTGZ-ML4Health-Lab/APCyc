#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional


def compute_valid_rate(metrics_per_sample) -> Optional[float]:
    if not metrics_per_sample:
        return None
    return sum([1 for row in metrics_per_sample if row.get('valid')]) / len(metrics_per_sample)
