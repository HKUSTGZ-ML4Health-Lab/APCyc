#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy

from apcyc.evaluation.metrics import rosetta_score


def compute_rosetta_total_score(
    pdb_path: str,
    peptide_chain: str,
    rec_chain: str,
    rec_chain_obj,
    rosetta_cfg: dict,
    failures: dict,
):
    total_cfg = deepcopy(rosetta_cfg)
    total_cfg['energy_mode'] = 'total_score'
    total_scope = total_cfg.get('total_score_scope', 'peptide')
    if total_scope != 'peptide' and rosetta_cfg.get('require_receptor', False) and rec_chain_obj is None:
        total_status = 'missing_receptor'
        failures[f'rosetta_total_{total_status}'] += 1
        return None
    rosetta_total_score_val, total_status = rosetta_score.rosetta_total_score(
        pdb_path,
        peptide_chain,
        total_cfg,
        rec_chain=rec_chain,
    )
    if total_status not in ('ok', 'disabled'):
        failures[f'rosetta_total_{total_status}'] += 1
    return rosetta_total_score_val
