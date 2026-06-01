#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics.helpers.metric_tm_shared import get_tm_metrics


def compute_tm_mean(valid_pdbs, peptide_chain: str, cfg: dict, cache=None):
    metrics = get_tm_metrics(valid_pdbs, peptide_chain, cfg, cache)
    return metrics.get('tm_mean')
