#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

from apcyc.evaluation.metrics import tm_score


def get_tm_metrics(
    valid_pdbs,
    peptide_chain: str,
    cfg: dict,
    cache: Optional[dict] = None,
):
    cache_key = (tuple(valid_pdbs), peptide_chain)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if not valid_pdbs:
        metrics = {
            'tm_mean': None,
            'tm_median': None,
        }
        if cache is not None:
            cache[cache_key] = metrics
        return metrics
    tm_stats = tm_score.tm_pairwise_diversity(valid_pdbs, peptide_chain, cfg)
    tm_mean_1_minus = tm_stats.get('mean_1_minus_tm')
    tm_median_1_minus = tm_stats.get('median_1_minus_tm')
    metrics = {
        'tm_mean': None if tm_mean_1_minus is None else 1.0 - tm_mean_1_minus,
        'tm_median': None if tm_median_1_minus is None else 1.0 - tm_median_1_minus,
    }
    if cache is not None:
        cache[cache_key] = metrics
    return metrics
