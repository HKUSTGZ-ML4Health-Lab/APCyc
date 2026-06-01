#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Iterable, List, Set


def load_id_list(path: str, suffix: str) -> List[str]:
    ids: List[str] = []
    suffix_l = suffix.lower()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            token = os.path.basename(token)
            if token.lower().endswith(suffix_l):
                token = token[: -len(suffix)]
            ids.append(token)
    return ids


def iter_pdb_files(pdb_dir: str, suffix: str, recursive: bool) -> Iterable[str]:
    suffix_l = suffix.lower()
    if recursive:
        for root, _, files in os.walk(pdb_dir):
            for name in files:
                if name.lower().endswith(suffix_l):
                    yield os.path.join(root, name)
    else:
        for name in os.listdir(pdb_dir):
            if name.lower().endswith(suffix_l):
                yield os.path.join(pdb_dir, name)


def collect_pdb_ids(pdb_dir: str, suffix: str, recursive: bool) -> Set[str]:
    ids: Set[str] = set()
    for path in iter_pdb_files(pdb_dir, suffix, recursive):
        name = os.path.basename(path)
        ids.add(name[: -len(suffix)])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Check train/valid IDs against PDB directory")
    parser.add_argument("--train_ids", type=str, required=True)
    parser.add_argument("--valid_ids", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default=".pdb")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_list", type=int, default=50)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    train_ids = set(load_id_list(args.train_ids, args.suffix))
    valid_ids = set(load_id_list(args.valid_ids, args.suffix))
    pdb_ids = collect_pdb_ids(args.pdb_dir, args.suffix, args.recursive)

    missing_train = sorted(train_ids - pdb_ids)
    missing_valid = sorted(valid_ids - pdb_ids)
    overlap = sorted(train_ids & valid_ids)

    print(f"train ids: {len(train_ids)}")
    print(f"valid ids: {len(valid_ids)}")
    print(f"pdb ids: {len(pdb_ids)}")
    print(f"missing train ids: {len(missing_train)}")
    print(f"missing valid ids: {len(missing_valid)}")
    print(f"train/valid overlap: {len(overlap)}")

    if missing_train:
        print("missing train (sample):", missing_train[: args.max_list])
    if missing_valid:
        print("missing valid (sample):", missing_valid[: args.max_list])
    if overlap:
        print("overlap (sample):", overlap[: args.max_list])

    if args.output_json:
        payload = {
            "train_total": len(train_ids),
            "valid_total": len(valid_ids),
            "pdb_total": len(pdb_ids),
            "missing_train": missing_train,
            "missing_valid": missing_valid,
            "overlap": overlap,
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
