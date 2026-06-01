#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmap_dataset import create_mmap
from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process CPCore complexes into mmap format')
    parser.add_argument('--index', type=str, default=None, help='Index file (one sample_id per line)')
    parser.add_argument('--train_index', type=str, default=None, help='Train index file (optional)')
    parser.add_argument('--valid_index', type=str, default=None, help='Valid index file (optional)')
    parser.add_argument('--out_index', type=str, default=None, help='Optional output path for merged index')
    parser.add_argument('--pdb_root', type=str, required=True, help='Directory containing PDBs')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for mmap')
    parser.add_argument('--rec_chain', type=str, default='R', help='Receptor chain id')
    parser.add_argument('--lig_chain', type=str, default='L', help='Peptide chain id')
    parser.add_argument('--pocket_th', type=float, default=10.0, help='CB distance threshold for pocket')
    return parser.parse_args()


def process_iterator(sample_ids, pdb_root, rec_chain, lig_chain, pocket_th):
    for cnt, pdb_id in enumerate(sample_ids):
        pdb_path = os.path.join(pdb_root, pdb_id + '.pdb')
        if not os.path.exists(pdb_path):
            print_log(f'{pdb_id} missing PDB: {pdb_path}', level='WARN')
            continue

        rec_blocks, lig_blocks = pdb_to_list_blocks(pdb_path, selected_chains=[rec_chain, lig_chain])
        if len(rec_blocks) == 0 or len(lig_blocks) == 0:
            print_log(f'{pdb_id} missing chains: {rec_chain}/{lig_chain}', level='WARN')
            continue

        try:
            _, (pocket_idx, _) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)
        except KeyError:
            print_log(f'{pdb_id} missing backbone atoms', level='WARN')
            continue

        rec_num_units = sum([len(block) for block in rec_blocks])
        lig_num_units = sum([len(block) for block in lig_blocks])
        data = ([block.to_tuple() for block in rec_blocks], [block.to_tuple() for block in lig_blocks])
        rec_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in rec_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])

        non_standard = 1 if '?' in lig_seq else 0

        yield pdb_id, data, [
            len(rec_blocks), len(lig_blocks), rec_num_units, lig_num_units,
            rec_chain, lig_chain, rec_seq, lig_seq, non_standard,
            ','.join([str(idx) for idx in pocket_idx]),
        ], cnt


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.index is None and (args.train_index is None or args.valid_index is None):
        raise ValueError('Provide --index or both --train_index and --valid_index')

    sample_ids = []
    seen = set()
    def _add_ids(path):
        if path is None:
            return
        with open(path, 'r') as fin:
            for line in fin:
                sid = line.strip()
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                sample_ids.append(sid)

    if args.index is not None:
        _add_ids(args.index)
    else:
        _add_ids(args.train_index)
        _add_ids(args.valid_index)
        if args.out_index is not None:
            os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
            with open(args.out_index, 'w') as fout:
                for sid in sample_ids:
                    fout.write(sid + '\n')
            print_log(f'Merged index saved to {args.out_index}')
    print_log(f'Total {len(sample_ids)} entries')

    create_mmap(
        process_iterator(sample_ids, args.pdb_root, args.rec_chain, args.lig_chain, args.pocket_th),
        args.out_dir, len(sample_ids))

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
