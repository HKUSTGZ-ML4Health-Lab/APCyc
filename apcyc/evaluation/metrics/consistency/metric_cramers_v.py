#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics.helpers.metric_diversity_shared import get_diversity_metrics


def compute_cramers_v(valid_pdbs, seqs, cas, cfg: dict, cache=None):
    metrics = get_diversity_metrics(valid_pdbs, seqs, cas, cfg, cache)
    return metrics.get('cramers_v')
