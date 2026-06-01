#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics.helpers.metric_rama_shared import get_rama_metrics


def compute_rama_favored(pdb_path: str, peptide_chain: str, rama_data_dir: str, failures: dict, cache=None):
    rama = get_rama_metrics(pdb_path, peptide_chain, rama_data_dir, failures, cache)
    return rama.get('favored')
