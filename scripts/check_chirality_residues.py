#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan PDBs and report residues with D chirality in a given chain.

Default expects:
  generated_dir/target_id/*.pdb
Use --flat if all PDBs are in one folder.
"""
import argparse
import csv
import os
from typing import Iterator, Tuple

from apcyc.evaluation.io import pdb_utils
from apcyc.evaluation.metrics import cpsea_geom


def parse_args():
    parser = argparse.ArgumentParser(description="Report D-chiral residues per PDB.")
    parser.add_argument("--generated_dir", required=True, help="Root dir with target subfolders or flat PDBs.")
    parser.add_argument("--peptide_chain", default="L", help="Peptide chain id (default: L).")
    parser.add_argument("--flat", action="store_true", help="Treat generated_dir as a flat PDB folder.")
    parser.add_argument("--out", default="chirality_residues.csv", help="CSV output path.")
    parser.add_argument("--max_per_pdb", type=int, default=10, help="Max D residues to record per PDB.")
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


def main():
    args = parse_args()
    rows = []
    total = 0
    missing_chain = 0
    d_pdbs = 0
    total_residues = 0
    total_chiral = 0
    total_d = 0

    for target_id, pdb_path in _iter_pdbs(args.generated_dir, args.flat):
        total += 1
        structure = pdb_utils.load_structure(pdb_path)
        chain = pdb_utils.get_chain(structure, args.peptide_chain)
        if chain is None:
            missing_chain += 1
            rows.append({
                "target_id": target_id,
                "pdb": pdb_path,
                "status": "missing_chain",
                "d_count": 0,
                "examples": "",
            })
            continue
        d_hits = []
        total_count = 0
        chiral_count = 0
        d_count = 0
        for residue in pdb_utils.iter_residues(chain):
            total_count += 1
            if not all(atom in residue for atom in ("N", "CA", "C", "CB")):
                continue
            chiral_count += 1
            if cpsea_geom.ca_chirality(
                residue["N"].get_coord(),
                residue["CA"].get_coord(),
                residue["C"].get_coord(),
                residue["CB"].get_coord(),
            ) == "D":
                d_count += 1
                d_hits.append(f"{residue.get_resname()}:{residue.id}")
                if len(d_hits) >= args.max_per_pdb:
                    break
        d_ratio = (d_count / chiral_count) if chiral_count > 0 else 0.0
        chiral_ratio = (chiral_count / total_count) if total_count > 0 else 0.0
        total_residues += total_count
        total_chiral += chiral_count
        total_d += d_count
        if d_hits:
            d_pdbs += 1
            rows.append({
                "target_id": target_id,
                "pdb": pdb_path,
                "status": "D_found",
                "d_count": d_count,
                "total_count": total_count,
                "chiral_count": chiral_count,
                "d_ratio": d_ratio,
                "chiral_ratio": chiral_ratio,
                "examples": ";".join(d_hits),
            })
        else:
            rows.append({
                "target_id": target_id,
                "pdb": pdb_path,
                "status": "all_L",
                "d_count": d_count,
                "total_count": total_count,
                "chiral_count": chiral_count,
                "d_ratio": d_ratio,
                "chiral_ratio": chiral_ratio,
                "examples": "",
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_id",
                "pdb",
                "status",
                "d_count",
                "total_count",
                "chiral_count",
                "d_ratio",
                "chiral_ratio",
                "examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    overall_d_ratio = (total_d / total_chiral) if total_chiral > 0 else 0.0
    overall_chiral_ratio = (total_chiral / total_residues) if total_residues > 0 else 0.0
    print(
        "summary:",
        f"total={total}",
        f"missing_chain={missing_chain}",
        f"pdbs_with_D={d_pdbs}",
        f"overall_d_ratio={overall_d_ratio:.6f}",
        f"overall_chiral_ratio={overall_chiral_ratio:.6f}",
        f"csv={args.out}",
    )


if __name__ == "__main__":
    main()
