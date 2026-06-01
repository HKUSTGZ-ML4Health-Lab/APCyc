#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface, dist_matrix_from_blocks
from utils.logger import print_log

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # noqa: BLE001
    tqdm = None


class PocketSelect(Select):
    def __init__(self, target_chains, pocket_res_set):
        self.target_chains = set(target_chains)
        self.pocket_res_set = pocket_res_set

    def accept_chain(self, chain):
        return chain.id in self.target_chains

    def accept_residue(self, residue):
        if not Polypeptide.is_aa(residue, standard=False):
            return False
        chain_id = residue.get_parent().id
        resseq = residue.id[1]
        icode = residue.id[2].strip()
        return (chain_id, resseq, icode) in self.pocket_res_set


class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = set(chains)

    def accept_chain(self, chain):
        return chain.id in self.chains


def write_pdb(in_pdb, out_pdb, select_obj):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(in_pdb).stem, in_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb, select_obj)


def get_interface(pdb, receptor_chains, ligand_chains, pocket_th=10.0):
    """Match PepGLAD pocket definition (CB distance threshold)."""
    list_blocks, chain_ids = pdb_to_list_blocks(pdb, receptor_chains + ligand_chains, return_chain_ids=True)
    chain2blocks = {chain: block for chain, block in zip(chain_ids, list_blocks)}
    for c in receptor_chains:
        assert c in chain2blocks, f'Chain {c} not found for receptor'
    for c in ligand_chains:
        assert c in chain2blocks, f'Chain {c} not found for ligand'

    rec_blocks, rec_block_chains, lig_blocks, lig_block_chains = [], [], [], []
    for c in receptor_chains:
        for block in chain2blocks[c]:
            rec_blocks.append(block)
            rec_block_chains.append(c)
    for c in ligand_chains:
        for block in chain2blocks[c]:
            lig_blocks.append(block)
            lig_block_chains.append(c)

    _, (pocket_idx, lig_if_idx) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)
    epitope = []
    for i in pocket_idx:
        epitope.append((rec_blocks[i], rec_block_chains[i], i))

    dist_mat = dist_matrix_from_blocks([rec_blocks[i] for i in pocket_idx], [lig_blocks[i] for i in lig_if_idx])
    min_dists = np.min(dist_mat, axis=-1)
    lig_idxs = np.argmin(dist_mat, axis=-1)
    dists = []
    for i, d in zip(lig_idxs, min_dists):
        i = lig_if_idx[i]
        dists.append((lig_blocks[i], lig_block_chains[i], i, d))

    return epitope, dists


def process_one(pdb_path, target_chains, ligand_chains, pocket_th, out_json_dir, out_pocket_dir, out_peptide_dir, suffix):
    epitope, _ = get_interface(pdb_path, target_chains, ligand_chains, pocket_th)
    pocket_json = []
    pocket_res_set = set()
    for res, chain_name, _ in epitope:
        resseq, icode = res.id
        icode = icode.strip()
        pocket_json.append([chain_name, [resseq, icode if icode else " "]])
        pocket_res_set.add((chain_name, resseq, icode))

    pdb_id = Path(pdb_path).stem
    out_json_path = os.path.join(out_json_dir, f"{pdb_id}{suffix}")
    with open(out_json_path, 'w') as f:
        json.dump(pocket_json, f, separators=(',', ':'))

    if out_pocket_dir is not None:
        out_pocket_path = os.path.join(out_pocket_dir, f"{pdb_id}_pocket.pdb")
        write_pdb(pdb_path, out_pocket_path, PocketSelect(target_chains, pocket_res_set))

    if out_peptide_dir is not None:
        out_pep_path = os.path.join(out_peptide_dir, f"{pdb_id}_peptide.pdb")
        write_pdb(pdb_path, out_pep_path, ChainSelect(ligand_chains))


def parse():
    parser = argparse.ArgumentParser(description='Extract PepGLAD-compatible pocket definitions from complexes.')
    parser.add_argument('--pdb', type=str, default=None, help='Single complex PDB path')
    parser.add_argument('--pdb_dir', type=str, default=None, help='Directory of complex PDBs')
    parser.add_argument('--index', type=str, default=None, help='Optional ID list (one per line)')
    parser.add_argument('--target_chains', type=str, nargs='+', required=True, help='Receptor chain ids')
    parser.add_argument('--ligand_chains', type=str, nargs='+', required=True, help='Peptide chain ids')
    parser.add_argument('--pocket_th', type=float, default=10.0, help='CB distance threshold')
    parser.add_argument('--out_json_dir', type=str, required=True, help='Output directory for pocket JSONs')
    parser.add_argument('--out_pocket_dir', type=str, default=None, help='Optional output dir for pocket PDBs')
    parser.add_argument('--out_peptide_dir', type=str, default=None, help='Optional output dir for peptide PDBs')
    parser.add_argument('--json_suffix', type=str, default='_complex_pocket.json', help='Output JSON suffix')
    return parser.parse_args()


def main():
    args = parse()
    os.makedirs(args.out_json_dir, exist_ok=True)
    if args.out_pocket_dir is not None:
        os.makedirs(args.out_pocket_dir, exist_ok=True)
    if args.out_peptide_dir is not None:
        os.makedirs(args.out_peptide_dir, exist_ok=True)

    pdbs = []
    if args.pdb is not None:
        pdbs = [args.pdb]
    elif args.pdb_dir is not None:
        if args.index is not None:
            with open(args.index, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            pdbs = [os.path.join(args.pdb_dir, f"{pid}.pdb") for pid in ids]
        else:
            pdbs = sorted(str(p) for p in Path(args.pdb_dir).glob('*.pdb'))
    else:
        raise ValueError('Either --pdb or --pdb_dir must be specified')

    iterator = pdbs
    if tqdm is not None:
        iterator = tqdm(pdbs, desc='Extract pockets', unit='pdb')
    for pdb_path in iterator:
        if not os.path.exists(pdb_path):
            continue
        process_one(
            pdb_path,
            args.target_chains,
            args.ligand_chains,
            args.pocket_th,
            args.out_json_dir,
            args.out_pocket_dir,
            args.out_peptide_dir,
            args.json_suffix,
        )
    print_log(f'Processed {len(pdbs)} PDBs. Output: {args.out_json_dir}')


if __name__ == '__main__':
    main()
