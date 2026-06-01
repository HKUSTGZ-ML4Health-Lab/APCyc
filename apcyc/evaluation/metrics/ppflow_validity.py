#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Tuple

import numpy as np

from apcyc.evaluation.io import pdb_utils

N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.53
C_N_LENGTH = 1.34


def bond_length_validity(pdb_path: str, chain_id: str, tol: float = 0.5) -> Tuple[Optional[bool], str]:
    structure = pdb_utils.load_structure(pdb_path)
    chain = pdb_utils.get_chain(structure, chain_id)
    if chain is None:
        return None, 'missing_chain'
    residues = pdb_utils.residues_as_list(chain)
    if len(residues) < 2:
        return None, 'too_short'

    nca = []
    cac = []
    cn = []

    for idx, residue in enumerate(residues):
        if not all(atom in residue for atom in ('N', 'CA', 'C')):
            return None, 'missing_atom'
        n_coord = residue['N'].get_coord()
        ca_coord = residue['CA'].get_coord()
        c_coord = residue['C'].get_coord()
        nca.append(np.linalg.norm(n_coord - ca_coord))
        cac.append(np.linalg.norm(ca_coord - c_coord))
        if idx < len(residues) - 1:
            next_res = residues[idx + 1]
            if 'N' not in next_res:
                return None, 'missing_atom'
            cn.append(np.linalg.norm(next_res['N'].get_coord() - c_coord))

    nca = np.asarray(nca)
    cac = np.asarray(cac)
    cn = np.asarray(cn)

    if np.any((nca > N_CA_LENGTH + tol) | (nca < N_CA_LENGTH - tol)):
        return False, 'bond_length_fail'
    if np.any((cac > CA_C_LENGTH + tol) | (cac < CA_C_LENGTH - tol)):
        return False, 'bond_length_fail'
    if np.any((cn > C_N_LENGTH + tol) | (cn < C_N_LENGTH - tol)):
        return False, 'bond_length_fail'
    return True, 'ok'
