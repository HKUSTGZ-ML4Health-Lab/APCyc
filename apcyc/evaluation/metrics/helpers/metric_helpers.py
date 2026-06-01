#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import Optional

from apcyc.evaluation.io import pdb_utils


def resolve_pdb_path(sample: dict) -> Optional[str]:
    pdb_path = sample.get('pdb')
    if pdb_path and not os.path.isabs(pdb_path):
        pdb_path = os.path.abspath(pdb_path)
    return pdb_path


def load_sample_chains(pdb_path: str, peptide_chain: str, rec_chain: str):
    structure = pdb_utils.load_structure(pdb_path)
    pep_chain_obj = pdb_utils.get_chain(structure, peptide_chain)
    rec_chain_obj = pdb_utils.get_chain(structure, rec_chain)
    return pep_chain_obj, rec_chain_obj


def extract_sequence_and_ca(chain):
    seq = pdb_utils.extract_sequence(chain)
    ca_coords, _ = pdb_utils.extract_ca_coords(chain)
    return seq, ca_coords


def resolve_pair_tuple(pair_hat, seq_len: int, assume_terminal_pair: bool):
    pair_tuple = None
    if isinstance(pair_hat, (list, tuple)) and len(pair_hat) >= 2:
        pair_tuple = (int(pair_hat[0]), int(pair_hat[1]))
    if pair_tuple is None and assume_terminal_pair and seq_len >= 2:
        pair_tuple = (0, seq_len - 1)
    return pair_tuple
