#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics.helpers.metric_diversity_shared import get_diversity_metrics


def compute_co_diversity(valid_pdbs, seqs, cas, cfg: dict, cache=None):
    metrics = get_diversity_metrics(valid_pdbs, seqs, cas, cfg, cache)
    return metrics.get('co_diversity')
