#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import shutil
import argparse

import numpy as np

from utils.logger import print_log
from utils.file_utils import get_filename, cnt_num_files
from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_to_data import blocks_to_data
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process PDB to monomers')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdb database')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()
    

def process_iterator(data_dir):

    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    file_cnt = 0
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        for pdb_file in os.listdir(category_dir):
            file_cnt += 1
            path = os.path.join(category_dir, pdb_file)
            tmp_file = os.path.join(tmp_dir, f'{pdb_file}.decompressed')

            try:
                # uncompress the file to the tmp file
                with gzip.open(path, 'rb') as fin:
                    with open(tmp_file, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
        
                list_blocks, chains = pdb_to_list_blocks(tmp_file, return_chain_ids=True)
            except Exception as e:
                print_log(f'Parsing {pdb_file} failed: {e}', level='WARN')
                continue

            for blocks, chain in zip(list_blocks, chains):

                # find broken chains: sequence starts from N end
                filter_blocks, NC_coords = [], []
                for block in blocks:
                    N_coord, C_coord, CA_coord = None, None, None
                    for atom in block:
                        if atom.name == 'N':
                            N_coord = atom.coordinate
                        elif atom.name == 'C':
                            C_coord = atom.coordinate
                        elif atom.name == 'CA':
                            CA_coord = atom.coordinate
                    if N_coord and C_coord and CA_coord:
                        filter_blocks.append(block)
                        NC_coords.append(N_coord)
                        NC_coords.append(C_coord)

                if len(filter_blocks) == 0:  # no valid residues
                    continue

                NC_coords = np.array(NC_coords)
                pep_bond_len = np.linalg.norm(NC_coords[1::2][:-1] - NC_coords[2::2], axis=-1)
                # broken = np.nonzero(pep_bond_len > 1.5)[0]

                if np.any(pep_bond_len > 1.5):
                    continue

                blocks = filter_blocks
                item_id = chain + '_' + pdb_file
                # data = blocks_to_data(blocks)
                num_blocks = len(blocks)
                num_units = sum([len(block.units) for block in blocks])
                data = [block.to_tuple() for block in blocks]

                seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in blocks])

                # id, data, properties, whether this entry is finished for producing data 
                yield item_id, data, [num_blocks, num_units, chain, seq], file_cnt
            
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    shutil.rmtree(tmp_dir)

def main(args):
    
    cnt = cnt_num_files(args.pdb_dir, recursive=True)

    print_log(f'Processing data from directory: {args.pdb_dir}.')
    print_log(f'Number of entries: {cnt}')
    create_mmap(
        process_iterator(args.pdb_dir),
        args.out_dir, cnt)
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())