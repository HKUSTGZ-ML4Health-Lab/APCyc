#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import Counter


def load_id_list(path: str):
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            token = os.path.basename(token)
            if token.lower().endswith(".pdb"):
                token = token[:-4]
            ids.append(token)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Count c_star for valid ids in jsonl")
    parser.add_argument("--jsonl", required=True, help="Path to jsonl file")
    parser.add_argument("--valid_ids", required=True, help="Path to valid ids txt")
    parser.add_argument("--output", default=None, help="Optional output json path")
    args = parser.parse_args()

    valid_list = load_id_list(args.valid_ids)
    valid_set = set(valid_list)
    dup = len(valid_list) - len(valid_set)

    counts = Counter()
    matched = 0
    with open(args.jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id", "")
            if sample_id in valid_set:
                counts[row.get("c_star", "UNKNOWN")] += 1
                matched += 1

    missing = len(valid_set) - matched

    print(f"valid_ids: {len(valid_set)} (dup={dup})")
    print(f"matched_in_jsonl: {matched}")
    print(f"missing_in_jsonl: {missing}")
    for key in sorted(counts.keys()):
        print(f"{key}: {counts[key]}")

    if args.output:
        payload = {
            "valid_ids": len(valid_set),
            "duplicate_ids": dup,
            "matched_in_jsonl": matched,
            "missing_in_jsonl": missing,
            "c_star_counts": dict(counts),
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
