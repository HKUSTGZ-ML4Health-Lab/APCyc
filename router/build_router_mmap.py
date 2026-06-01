#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

# Ensure APCyc root is ahead of this script dir on sys.path (avoid data.py shadowing).
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from router.data import build_router_mmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Build router mmap cache")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--mmap_dir", type=str, required=True)
    parser.add_argument("--type_list", type=str, default="HT,SS,ISO")
    parser.add_argument("--chain_id", type=str, default="R")
    parser.add_argument("--backbone_only", action="store_true")
    args = parser.parse_args()

    type_list = [x.strip() for x in args.type_list.split(",") if x.strip()]
    build_router_mmap(
        args.train_jsonl,
        args.pdb_dir,
        args.mmap_dir,
        type_list,
        chain_id=args.chain_id,
        backbone_only=args.backbone_only,
    )


if __name__ == "__main__":
    main()
