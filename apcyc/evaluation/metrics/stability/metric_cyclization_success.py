#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics import cpsea_geom


def compute_cyclization_success(
    pdb_path: str,
    peptide_chain: str,
    pair_tuple,
    cb_low: float,
    cb_high: float,
    fallback_atom: str,
    failures: dict,
):
    cycl_ok, cycl_reason = cpsea_geom.cyclization_success(
        pdb_path, peptide_chain, pair_tuple, cb_low, cb_high, fallback_atom
    )
    if cycl_ok is False:
        failures[cycl_reason] += 1
    return cycl_ok
