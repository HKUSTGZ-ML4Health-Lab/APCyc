#!/usr/bin/python
# -*- coding:utf-8 -*-
import csv
import os
from typing import Dict, Optional, Tuple

from utils.logger import print_log

_SCORE_CACHE: Dict[str, Dict[str, Optional[float]]] = {}


def _normalize_pdb_id(value: str) -> str:
    base = os.path.basename(value.strip())
    if base.endswith(".pdb.gz"):
        return base[:-7]
    if base.endswith(".pdb"):
        return base[:-4]
    return os.path.splitext(base)[0]


def _load_score_csv(path: str) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return scores
        key_col = "input" if "input" in reader.fieldnames else reader.fieldnames[0]
        score_col = "affinity" if "affinity" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            raw_id = (row.get(key_col) or "").strip()
            if not raw_id:
                continue
            pid = _normalize_pdb_id(raw_id)
            val = (row.get(score_col) or "").strip()
            try:
                scores[pid] = float(val) if val != "" else None
            except ValueError:
                scores[pid] = None
    return scores


def vina_score(pdb_path: str, config: dict) -> Tuple[Optional[float], str]:
    if not config.get("enabled", False):
        return None, "disabled"
    score_csv = config.get("score_csv")
    if not score_csv or not os.path.exists(score_csv):
        return None, "missing_vina_csv"
    cache = _SCORE_CACHE.get(score_csv)
    if cache is None:
        try:
            cache = _load_score_csv(score_csv)
            _SCORE_CACHE[score_csv] = cache
        except Exception as exc:  # pylint: disable=broad-except
            print_log(f"Vina score CSV load failed: {exc}")
            return None, "vina_csv_failed"
    pid = _normalize_pdb_id(pdb_path)
    score = cache.get(pid)
    if score is None:
        return None, "missing_vina_score"
    return float(score), "ok"
