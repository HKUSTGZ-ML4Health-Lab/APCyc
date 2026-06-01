#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CPSea LNR test set for APCyc sampling (no mmap):
1) merge receptor + peptide into a single complex PDB with chains R/L
2) write index.txt (sample_id per line)
3) write meta.jsonl ({"sample_id": ...} per line)
"""

import argparse
import json
import os
from typing import List, Optional

try:
    from Bio.PDB import PDBParser, PDBIO  # type: ignore
    from Bio.PDB.Chain import Chain  # type: ignore
    from Bio.PDB.Model import Model  # type: ignore
    from Bio.PDB.Structure import Structure  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise SystemExit("Biopython is required for this script. Please install biopython.") from exc


def _list_receptor_ids(rec_dir: str) -> List[str]:
    ids = []
    for fname in os.listdir(rec_dir):
        if not fname.endswith("_receptor.pdb"):
            continue
        ids.append(fname.replace("_receptor.pdb", ""))
    return sorted(set(ids))


def _first_chain(structure) -> Optional[Chain]:
    model = next(structure.get_models(), None)
    if model is None:
        return None
    for chain in model:
        return chain
    return None


def merge_complex(rec_path: str, pep_path: str, out_path: str, rec_chain_id: str, lig_chain_id: str) -> None:
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    rec_struct = parser.get_structure("rec", rec_path)
    pep_struct = parser.get_structure("pep", pep_path)

    # Merge all receptor chains into a single chain id.
    rec_chain = Chain(rec_chain_id)
    rec_model = next(rec_struct.get_models(), None)
    if rec_model is None:
        raise ValueError(f"No model found in receptor: {rec_path}")
    for chain in rec_model:
        for res in chain:
            rec_chain.add(res)

    pep_chain = _first_chain(pep_struct)
    if pep_chain is None:
        raise ValueError(f"No peptide chain found in {pep_path}")
    pep_chain.id = lig_chain_id

    model = Model(0)
    model.add(rec_chain)
    model.add(pep_chain)
    structure = Structure("complex")
    structure.add(model)

    io.set_structure(structure)
    io.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LNR test set for APCyc sampling (no mmap).")
    parser.add_argument("--receptor_dir", required=True, help="Directory with *_receptor.pdb files.")
    parser.add_argument("--separated_dir", required=True, help="Directory with <id>/peptide.pdb subfolders.")
    parser.add_argument("--out_complex_dir", required=True, help="Output dir for merged complex PDBs.")
    parser.add_argument("--out_index", required=True, help="Output index.txt path.")
    parser.add_argument("--out_meta", required=True, help="Output meta.jsonl path.")
    parser.add_argument("--rec_chain", default="R", help="Receptor chain id in merged PDB (default: R).")
    parser.add_argument("--lig_chain", default="L", help="Peptide chain id in merged PDB (default: L).")
    parser.add_argument("--skip_missing", action="store_true", help="Skip samples missing peptide/receptor.")
    args = parser.parse_args()

    os.makedirs(args.out_complex_dir, exist_ok=True)

    ids = _list_receptor_ids(args.receptor_dir)
    kept_ids: List[str] = []
    for sid in ids:
        rec_path = os.path.join(args.receptor_dir, f"{sid}_receptor.pdb")
        pep_path = os.path.join(args.separated_dir, sid, "peptide.pdb")
        if not os.path.exists(pep_path):
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing peptide: {pep_path}")
        if not os.path.exists(rec_path):
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing receptor: {rec_path}")
        out_path = os.path.join(args.out_complex_dir, f"{sid}.pdb")
        merge_complex(rec_path, pep_path, out_path, args.rec_chain, args.lig_chain)
        kept_ids.append(sid)

    with open(args.out_index, "w") as f:
        for sid in kept_ids:
            f.write(sid + "\n")

    with open(args.out_meta, "w") as f:
        for sid in kept_ids:
            f.write(json.dumps({"sample_id": sid}) + "\n")

    print(f"Done. complexes={len(kept_ids)}")
    print(f"index: {args.out_index}")
    print(f"meta:  {args.out_meta}")


if __name__ == "__main__":
    main()
