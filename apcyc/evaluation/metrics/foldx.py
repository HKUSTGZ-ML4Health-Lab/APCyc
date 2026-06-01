#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import tempfile
from typing import Optional, Tuple

from utils.logger import print_log

from apcyc.evaluation.io.pdb_utils import split_pdb_by_chain


def _resolve_foldx_path(config: dict) -> Optional[str]:
    path = config.get('bin_path')
    if not path:
        candidate = os.path.abspath('ppflow/bin/FoldX/foldx')
        if os.path.exists(candidate):
            return candidate
        return None
    path = os.path.expanduser(path)
    return path if os.path.exists(path) else None


def foldx_stability(pdb_path: str, pep_chain: str, rec_chain: str, config: dict) -> Tuple[Optional[float], str]:
    if not config.get('enabled', True):
        return None, 'disabled'
    foldx_path = _resolve_foldx_path(config)
    if not foldx_path:
        return None, 'missing_foldx'

    try:
        from ppflow.tools.score.foldx_energy import FoldXGibbsEnergy
    except Exception as exc:  # pylint: disable=broad-except
        print_log(f"FoldX wrapper unavailable: {exc}")
        return None, 'foldx_import_failed'

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_files = split_pdb_by_chain(pdb_path, tmpdir)
            if pep_chain not in chain_files or rec_chain not in chain_files:
                return None, 'missing_chain'
            with FoldXGibbsEnergy(foldx_path=foldx_path) as scorer:
                scorer.set_ligand(chain_files[pep_chain])
                scorer.set_receptor(chain_files[rec_chain])
                if config.get('sidechain_pack', False):
                    scorer.side_chain_packing('ligand')
                    scorer.side_chain_packing('receptor')
                dg = scorer.cal_interface_energy()
            return float(dg), 'ok'
    except Exception as exc:  # pylint: disable=broad-except
        print_log(f"FoldX failed for {os.path.basename(pdb_path)}: {exc}")
        return None, 'foldx_failed'


def imp_percent_g(gen_dg_list, ref_dg) -> Optional[float]:
    if ref_dg is None:
        return None
    if not gen_dg_list:
        return None
    total = len(gen_dg_list)
    better = sum([1 for dg in gen_dg_list if dg is not None and dg < ref_dg])
    return float(better / total) if total > 0 else None
