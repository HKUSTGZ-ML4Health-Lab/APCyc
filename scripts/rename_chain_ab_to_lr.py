#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename peptide/receptor chain IDs (default A->L, B->R) in PDB files.
"""
import argparse
import os
from Bio import PDB


def rename_chains(pdb_in: str, pdb_out: str, pep_chain: str, rec_chain: str, new_pep: str, new_rec: str, non_l_to_r: bool) -> None:
    parser = PDB.PDBParser(QUIET=True)
    writer = PDB.PDBIO()
    structure = parser.get_structure("pdb", pdb_in)
    for model in structure:
        if non_l_to_r:
            non_l = [chain for chain in model if chain.id != "L"]
            if len(non_l) != 1:
                # Skip if ambiguous (e.g., more than 2 chains)
                return
            non_l[0].id = "R"
        else:
            for chain in model:
                if chain.id == pep_chain:
                    chain.id = new_pep
                elif chain.id == rec_chain:
                    chain.id = new_rec
    os.makedirs(os.path.dirname(pdb_out), exist_ok=True)
    writer.set_structure(structure)
    writer.save(pdb_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename chain IDs in PDB files.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing PDB files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for renamed PDB files.")
    parser.add_argument("--pep_chain", default="A", help="Peptide chain ID in input (default: A).")
    parser.add_argument("--rec_chain", default="B", help="Receptor chain ID in input (default: B).")
    parser.add_argument("--new_pep", default="L", help="New peptide chain ID (default: L).")
    parser.add_argument("--new_rec", default="R", help="New receptor chain ID (default: R).")
    parser.add_argument("--non_l_to_r", action="store_true", help="Rename the only non-L chain to R.")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if args.recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if not name.lower().endswith(".pdb"):
                    continue
                src = os.path.join(root, name)
                rel = os.path.relpath(src, input_dir)
                dst = os.path.join(output_dir, rel)
                rename_chains(src, dst, args.pep_chain, args.rec_chain, args.new_pep, args.new_rec, args.non_l_to_r)
    else:
        for name in os.listdir(input_dir):
            if not name.lower().endswith(".pdb"):
                continue
            src = os.path.join(input_dir, name)
            dst = os.path.join(output_dir, name)
            rename_chains(src, dst, args.pep_chain, args.rec_chain, args.new_pep, args.new_rec, args.non_l_to_r)


if __name__ == "__main__":
    main()
