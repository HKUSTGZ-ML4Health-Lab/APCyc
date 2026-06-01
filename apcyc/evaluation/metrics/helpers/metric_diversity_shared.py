#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

from apcyc.evaluation.metrics import pepglad_divcons


def get_diversity_metrics(
    valid_pdbs,
    seqs,
    cas,
    cfg: dict,
    cache: Optional[dict] = None,
):
    cache_key = (tuple(valid_pdbs),)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if not valid_pdbs:
        metrics = {
            'co_diversity': None,
            'cramers_v': None,
        }
        if cache is not None:
            cache[cache_key] = metrics
        return metrics
    diversity = pepglad_divcons.diversity_consistency(
        seqs,
        cas,
        cfg.get('clustering', {}).get('seq_similarity_threshold', 0.4),
        cfg.get('clustering', {}).get('rmsd_threshold', 4.0),
    )
    metrics = {
        'co_diversity': diversity.get('co_diversity'),
        'cramers_v': diversity.get('consistency'),
    }
    if cache is not None:
        cache[cache_key] = metrics
    return metrics
