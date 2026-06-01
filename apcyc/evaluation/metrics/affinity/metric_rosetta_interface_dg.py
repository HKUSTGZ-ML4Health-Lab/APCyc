#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy

from apcyc.evaluation.metrics import rosetta_score


def compute_rosetta_interface_dg(
    pdb_path: str,
    peptide_chain: str,
    rec_chain: str,
    rec_chain_obj,
    rosetta_cfg: dict,
    failures: dict,
):
    interface_cfg = deepcopy(rosetta_cfg)
    interface_cfg['energy_mode'] = 'interface_dg'
    if rosetta_cfg.get('require_receptor', False) and rec_chain_obj is None:
        rosetta_status = 'missing_receptor'
        failures[rosetta_status] += 1
        return None
    rosetta_score_val, rosetta_status = rosetta_score.rosetta_total_score(
        pdb_path,
        peptide_chain,
        interface_cfg,
        rec_chain=rec_chain,
    )
    if rosetta_status not in ('ok', 'disabled'):
        failures[rosetta_status] += 1
    return rosetta_score_val
