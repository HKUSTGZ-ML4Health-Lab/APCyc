#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group flat PDB outputs into receptor subdirectories.
"""
import argparse
import os
import re
import shutil


_GLAD_RE = re.compile(r"^glad_([A-Za-z0-9]{4})_")
_FLOW_RE = re.compile(r"^flow_([A-Za-z0-9]{4})_")


def _get_receptor_id(filename: str, mode: str) -> str:
    name = os.path.basename(filename)
    base, _ = os.path.splitext(name)
    if mode == "glad":
        match = _GLAD_RE.match(base)
        if match:
            return match.group(1)
        return base[:4]
    if mode == "flow":
        match = _FLOW_RE.match(base)
        if match:
            return match.group(1)
        return base[:4]
    if mode == "pdb4":
        return base[:4]
    return base.split("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Group PDB files by receptor id.")
    parser.add_argument("--input_dir", required=True, help="Directory with flat PDB files.")
    parser.add_argument("--output_dir", default=None, help="Output directory (defaults to input_dir).")
    parser.add_argument(
        "--mode",
        default="glad",
        choices=["glad", "flow", "prefix", "pdb4"],
        help="How to parse receptor id from filename.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    moved = 0
    for name in sorted(os.listdir(input_dir)):
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src) or not name.lower().endswith(".pdb"):
            continue
        receptor_id = _get_receptor_id(name, args.mode)
        dst_dir = os.path.join(output_dir, receptor_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        base, ext = os.path.splitext(dst)
        idx = 1
        while os.path.exists(dst):
            dst = f"{base}_{idx}{ext}"
            idx += 1
        shutil.move(src, dst)
        moved += 1

    print(f"Moved {moved} files into receptor subdirectories under {output_dir}")


if __name__ == "__main__":
    main()
