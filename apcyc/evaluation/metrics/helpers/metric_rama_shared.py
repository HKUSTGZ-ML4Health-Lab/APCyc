#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

from apcyc.evaluation.metrics import cpsea_geom


def get_rama_metrics(
    pdb_path: str,
    peptide_chain: str,
    rama_data_dir: str,
    failures: dict,
    cache: Optional[dict] = None,
):
    cache_key = (pdb_path, peptide_chain, rama_data_dir)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    rama = cpsea_geom.rama_allowed_favored(pdb_path, peptide_chain, rama_data_dir)
    if rama.get('allowed') is None and rama.get('favored') is None:
        failures['rama_unavailable'] += 1
    if cache is not None:
        cache[cache_key] = rama
    return rama
