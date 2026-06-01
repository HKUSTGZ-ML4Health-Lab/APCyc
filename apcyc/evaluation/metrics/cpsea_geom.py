#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
from Bio import PDB

from apcyc.evaluation.io import pdb_utils
from utils.logger import print_log

_RAMA_CACHE = {}

RAMA_PREFERENCES = {
    "General": {"file": "pref_general.data", "bounds": [0, 0.0005, 0.02, 1]},
    "GLY": {"file": "pref_glycine.data", "bounds": [0, 0.002, 0.02, 1]},
    "PRO": {"file": "pref_proline.data", "bounds": [0, 0.002, 0.02, 1]},
    "PRE-PRO": {"file": "pref_preproline.data", "bounds": [0, 0.002, 0.02, 1]},
}


def _rama_pref_values(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    if data_dir in _RAMA_CACHE:
        return _RAMA_CACHE[data_dir]
    prefs = {}
    for key, val in RAMA_PREFERENCES.items():
        pref = np.full((360, 360), 0, dtype=np.float64)
        path = os.path.join(data_dir, val['file'])
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                x = int(float(parts[1]))
                y = int(float(parts[0]))
                score = float(parts[2])
                pref[x + 180][y + 180] = score
                pref[x + 179][y + 179] = score
                pref[x + 179][y + 180] = score
                pref[x + 180][y + 179] = score
        prefs[key] = pref
    _RAMA_CACHE[data_dir] = prefs
    return prefs


def rama_allowed_favored(pdb_path: str, chain_id: str, data_dir: str) -> Dict[str, Optional[float]]:
    if not os.path.isdir(data_dir):
        return {'allowed': None, 'favored': None, 'total': 0}
    prefs = _rama_pref_values(data_dir)
    structure = pdb_utils.load_structure(pdb_path)
    chain = pdb_utils.get_chain(structure, chain_id)
    if chain is None:
        return {'allowed': None, 'favored': None, 'total': 0}

    accept_count = 0
    fav_count = 0
    total = 0

    ppb = PDB.PPBuilder()
    for poly in ppb.build_peptides(chain):
        phi_psi = poly.get_phi_psi_list()
        poly_len = len(poly)
        for idx, residue in enumerate(poly):
            phi, psi = phi_psi[idx]
            if phi is None or psi is None:
                continue
            res_name = residue.resname
            if idx < poly_len - 1 and poly[idx + 1].resname == "PRO":
                aa_type = "PRE-PRO"
            elif res_name == "PRO":
                aa_type = "PRO"
            elif res_name == "GLY":
                aa_type = "GLY"
            else:
                aa_type = "General"
            phi_deg = math.degrees(phi)
            psi_deg = math.degrees(psi)
            allowed = prefs[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= RAMA_PREFERENCES[aa_type]['bounds'][1]
            favored = prefs[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= RAMA_PREFERENCES[aa_type]['bounds'][2]
            total += 1
            if allowed:
                accept_count += 1
            if favored:
                fav_count += 1
    if total == 0:
        return {'allowed': None, 'favored': None, 'total': 0}
    return {
        'allowed': float(accept_count / total),
        'favored': float(fav_count / total),
        'total': total,
    }


def _subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def _cross(v1, v2):
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


def _dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def ca_chirality(N, CA, C, CB):
    n = _cross(_subtract(N, CA), _subtract(C, CA))
    return 'D' if _dot(n, _subtract(CB, CA)) < 0 else 'L'


def chirality_success(pdb_path: str, chain_id: str) -> Tuple[Optional[bool], str]:
    structure = pdb_utils.load_structure(pdb_path)
    chain = pdb_utils.get_chain(structure, chain_id)
    if chain is None:
        return None, 'missing_chain'
    for residue in pdb_utils.iter_residues(chain):
        if not all(atom in residue for atom in ('N', 'CA', 'C', 'CB')):
            continue
        if ca_chirality(residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord(), residue['CB'].get_coord()) == 'D':
            return False, 'chirality_fail'
    return True, 'ok'


def cyclization_success(pdb_path: str, chain_id: str, pair: Tuple[int, int], cb_low: float, cb_high: float, fallback_atom: str = 'CA') -> Tuple[Optional[bool], str]:
    structure = pdb_utils.load_structure(pdb_path)
    chain = pdb_utils.get_chain(structure, chain_id)
    if chain is None:
        return None, 'missing_chain'
    residues = pdb_utils.residues_as_list(chain)
    i, j = pair
    if i is None or j is None:
        return None, 'missing_pair'
    if i < 0 or j < 0 or i >= len(residues) or j >= len(residues):
        return None, 'pair_out_of_range'
    ai = residues[i]
    aj = residues[j]
    atom_name = 'CB' if 'CB' in ai and 'CB' in aj else fallback_atom
    if atom_name not in ai or atom_name not in aj:
        return None, 'missing_atom'
    dist = np.linalg.norm(ai[atom_name].get_coord() - aj[atom_name].get_coord())
    return (cb_low <= dist <= cb_high), 'ok'
