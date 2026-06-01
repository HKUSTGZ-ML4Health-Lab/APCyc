#!/usr/bin/python
# -*- coding:utf-8 -*-
from apcyc.evaluation.metrics import ppflow_validity


def compute_bond_length_validity(
    pdb_path: str,
    peptide_chain: str,
    ppflow_enabled: bool,
    ppflow_tol: float,
    failures: dict,
):
    if not ppflow_enabled:
        return None
    bond_ok, bond_reason = ppflow_validity.bond_length_validity(pdb_path, peptide_chain, ppflow_tol)
    if bond_ok is False or bond_ok is None:
        failures[bond_reason] += 1
    return bond_ok
