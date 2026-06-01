#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from utils.logger import print_log
from apcyc.evaluation.io import pdb_utils

_TM_RE = re.compile(r"TM-score\s*=\s*([0-9\.]+)")


def resolve_tm_exec(config: dict) -> Optional[str]:
    tm_exec = config.get('tm_score_bin')
    if tm_exec:
        tm_exec = os.path.expanduser(tm_exec)
        if os.path.exists(tm_exec):
            return tm_exec
    # try ppflow default
    candidate = os.path.abspath('ppflow/bin/TMscore/TMscore')
    if os.path.exists(candidate):
        return candidate
    # try PATH
    return shutil.which('TMscore') or shutil.which('USalign') or shutil.which('TMscore.exe')


def tm_score(pdb_a: str, pdb_b: str, tm_exec: str) -> Optional[float]:
    try:
        proc = subprocess.run([tm_exec, pdb_a, pdb_b], capture_output=True, text=True, check=False)
        out = proc.stdout + proc.stderr
        match = _TM_RE.search(out)
        if match:
            return float(match.group(1))
        return None
    except Exception as exc:  # pylint: disable=broad-except
        print_log(f"TM-score failed: {exc}")
        return None


def tm_pairwise_diversity(pdb_paths: List[str], chain_id: str, config: dict) -> Dict[str, Optional[float]]:
    tm_exec = resolve_tm_exec(config)
    if not tm_exec:
        return {
            'mean_1_minus_tm': None,
            'median_1_minus_tm': None,
            'min_tm': None,
            'max_tm': None,
            'coverage': 0.0,
        }
    scores = []
    total = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        ca_paths = []
        for idx, pdb_path in enumerate(pdb_paths):
            structure = pdb_utils.load_structure(pdb_path)
            chain = pdb_utils.get_chain(structure, chain_id)
            if chain is None:
                continue
            out_path = os.path.join(tmpdir, f"ca_{idx}.pdb")
            if pdb_utils.write_ca_pdb(chain, out_path) == 0:
                continue
            ca_paths.append(out_path)

        for i in range(len(ca_paths)):
            for j in range(i + 1, len(ca_paths)):
                total += 1
                score = tm_score(ca_paths[i], ca_paths[j], tm_exec)
                if score is not None:
                    scores.append(score)
    if not scores:
        return {
            'mean_1_minus_tm': None,
            'median_1_minus_tm': None,
            'min_tm': None,
            'max_tm': None,
            'coverage': 0.0,
        }
    arr = sorted(scores)
    n = len(arr)
    mean_1_minus = float(sum([1.0 - s for s in arr]) / n)
    median_1_minus = float(1.0 - arr[n // 2])
    return {
        'mean_1_minus_tm': mean_1_minus,
        'median_1_minus_tm': median_1_minus,
        'min_tm': float(min(arr)),
        'max_tm': float(max(arr)),
        'coverage': float(n / total) if total > 0 else 0.0,
    }
