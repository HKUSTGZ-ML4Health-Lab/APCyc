#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics import vina_score


def compute_vina_score(pdb_path: str, vina_cfg: dict, failures: dict):
    score_val, vina_status = vina_score.vina_score(pdb_path, vina_cfg)
    if vina_status not in ('ok', 'disabled'):
        failures[vina_status] += 1
    return score_val
