#!/usr/bin/python
# -*- coding:utf-8 -*-
import csv
import json
import os
from typing import Dict, Iterable, List

from utils.logger import print_log


def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rows:
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        if fieldnames:
            missing = [key for key in sorted(all_keys) if key not in fieldnames]
            fieldnames = list(fieldnames) + missing
        else:
            fieldnames = sorted(all_keys)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_failures(failures: Dict[str, int]):
    if not failures:
        return
    msg = ', '.join([f"{k}={v}" for k, v in failures.items()])
    print_log(f"Failures: {msg}")
