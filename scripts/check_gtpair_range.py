#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract records whose gt_pair is outside [0, peptide_length-1].
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check gt_pair range in jsonl.")
    parser.add_argument("--jsonl", required=True, help="Input jsonl file.")
    parser.add_argument("--out", default="bad_gtpair.jsonl", help="Output jsonl file.")
    parser.add_argument("--max", type=int, default=0, help="Max bad records to write (0 = all).")
    parser.add_argument(
        "--mode",
        choices=["range", "not_ends"],
        default="range",
        help="range: gt_pair outside [0, L-1]; not_ends: i!=0 or j!=L-1",
    )
    return parser.parse_args()


def _bad_pair(gt_pair, length: int, mode: str) -> Tuple[bool, str]:
    if gt_pair is None:
        return True, "missing_gt_pair"
    if not isinstance(gt_pair, (list, tuple)) or len(gt_pair) != 2:
        return True, "invalid_gt_pair"
    try:
        i, j = int(gt_pair[0]), int(gt_pair[1])
    except Exception:
        return True, "non_int_gt_pair"
    if length is None:
        return True, "missing_length"
    if mode == "not_ends":
        if i != 0 or j != length - 1:
            return True, "not_ends"
        return False, ""
    if i < 0 or j < 0:
        return True, "negative_index"
    if i >= length or j >= length:
        return True, "out_of_range"
    return False, ""


def main() -> None:
    args = parse_args()
    total = 0
    bad = 0
    reasons = {}
    written = 0

    with open(args.jsonl, "r") as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            length = record.get("peptide_length")
            gt_pair = record.get("gt_pair")
            is_bad, reason = _bad_pair(gt_pair, length, args.mode)
            if not is_bad:
                continue
            bad += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            record["_bad_reason"] = reason
            if args.max and written >= args.max:
                continue
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"total={total} bad={bad} written={written}")
    for key in sorted(reasons.keys()):
        print(f"{key}={reasons[key]}")


if __name__ == "__main__":
    main()
