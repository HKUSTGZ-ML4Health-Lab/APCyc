#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check peptide chirality and whether a simple mirror (x -> -x) would fix it.

Expected input layout (default):
generated_dir/
  target_id_1/
    sample1.pdb
    sample2.pdb
  target_id_2/
    ...

Use --flat if all PDBs are in a single directory.
"""
import argparse
import csv
import os
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from apcyc.evaluation.io import pdb_utils
from apcyc.evaluation.metrics import cpsea_geom


def parse_args():
    parser = argparse.ArgumentParser(description="Chirality check with mirror test.")
    parser.add_argument("--generated_dir", required=True, help="Root dir with target subfolders or flat PDBs.")
    parser.add_argument("--peptide_chain", default="L", help="Peptide chain id (default: L).")
    parser.add_argument("--flat", action="store_true", help="Treat generated_dir as a flat PDB folder.")
    parser.add_argument("--out", default="chirality_mirror.csv", help="CSV output path.")
    return parser.parse_args()


def _iter_pdbs(root: str, flat: bool) -> Iterator[Tuple[str, str]]:
    if flat:
        for name in sorted(os.listdir(root)):
            if name.lower().endswith(".pdb"):
                yield os.path.splitext(name)[0], os.path.join(root, name)
        return
    for target_id in sorted(os.listdir(root)):
        tdir = os.path.join(root, target_id)
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if name.lower().endswith(".pdb"):
                yield target_id, os.path.join(tdir, name)


def _chirality_counts(chain, mirror: bool = False) -> Dict[str, Optional[int]]:
    total = 0
    d_count = 0
    l_count = 0
    for residue in pdb_utils.iter_residues(chain):
        if not all(atom in residue for atom in ("N", "CA", "C", "CB")):
            continue
        coords = {}
        for atom in ("N", "CA", "C", "CB"):
            coord = residue[atom].get_coord().astype(np.float32)
            if mirror:
                coord[0] *= -1.0
            coords[atom] = coord
        chir = cpsea_geom.ca_chirality(coords["N"], coords["CA"], coords["C"], coords["CB"])
        total += 1
        if chir == "D":
            d_count += 1
        else:
            l_count += 1
    if total == 0:
        return {"total": 0, "d": None, "l": None}
    return {"total": total, "d": d_count, "l": l_count}


def _success_from_counts(counts: Dict[str, Optional[int]]) -> Optional[bool]:
    if not counts["total"]:
        return None
    return bool(counts["d"] == 0)


def main():
    args = parse_args()
    rows = []
    total_files = 0
    orig_fail = 0
    mirror_fix = 0
    missing_chain = 0

    for target_id, pdb_path in _iter_pdbs(args.generated_dir, args.flat):
        total_files += 1
        structure = pdb_utils.load_structure(pdb_path)
        chain = pdb_utils.get_chain(structure, args.peptide_chain)
        if chain is None:
            missing_chain += 1
            rows.append({
                "target_id": target_id,
                "pdb": pdb_path,
                "orig_ok": None,
                "mirror_ok": None,
                "orig_d": None,
                "orig_total": 0,
                "mirror_d": None,
                "mirror_total": 0,
                "note": "missing_chain",
            })
            continue

        orig_counts = _chirality_counts(chain, mirror=False)
        mirror_counts = _chirality_counts(chain, mirror=True)
        orig_ok = _success_from_counts(orig_counts)
        mirror_ok = _success_from_counts(mirror_counts)
        if orig_ok is False:
            orig_fail += 1
            if mirror_ok is True:
                mirror_fix += 1

        rows.append({
            "target_id": target_id,
            "pdb": pdb_path,
            "orig_ok": orig_ok,
            "mirror_ok": mirror_ok,
            "orig_d": orig_counts["d"],
            "orig_total": orig_counts["total"],
            "mirror_d": mirror_counts["d"],
            "mirror_total": mirror_counts["total"],
            "note": "",
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "target_id",
        "pdb",
        "orig_ok",
        "mirror_ok",
        "orig_d",
        "orig_total",
        "mirror_d",
        "mirror_total",
        "note",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "summary:",
        f"total={total_files}",
        f"missing_chain={missing_chain}",
        f"orig_fail={orig_fail}",
        f"mirror_fixable={mirror_fix}",
        f"csv={args.out}",
    )


if __name__ == "__main__":
    main()
