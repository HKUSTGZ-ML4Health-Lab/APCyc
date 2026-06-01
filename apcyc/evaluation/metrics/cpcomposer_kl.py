#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from Bio import PDB

from apcyc.evaluation.io import pdb_utils
from models.autoencoder.sidechain.constants import AA_GEOMETRY


AA20 = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]
AA20_SET = set(AA20)


@dataclass
class KLReference:
    aa_counts: np.ndarray
    phi_counts: np.ndarray
    psi_counts: np.ndarray
    chi_counts: np.ndarray


@dataclass
class KLMetrics:
    aa_kl: Optional[float]
    b_kl: Optional[float]
    s_kl: Optional[float]


def _dihedral(p0, p1, p2, p3) -> float:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1) + 1e-8
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def _chi_atom_sets(resname: str) -> List[List[str]]:
    if resname not in AA_GEOMETRY:
        return []
    aa_dict = AA_GEOMETRY[resname]
    atom_sets = []
    for parent_ix in aa_dict['chi_indices']:
        atom_quartet = aa_dict['parents'][parent_ix] + [aa_dict['atoms'][parent_ix]]
        atom_sets.append(atom_quartet)
    return atom_sets


def _counts_init(num_bins: int) -> np.ndarray:
    return np.zeros(num_bins, dtype=np.float64)


def _hist_index(angle: float, num_bins: int) -> int:
    # map [-180, 180) to bin
    angle = (angle + 180.0) % 360.0 - 180.0
    bin_size = 360.0 / num_bins
    return int((angle + 180.0) // bin_size)


def _collect_counts(pdb_path: str, chain_id: str, num_bins_phi: int, num_bins_chi: int,
                    masked_aas: Optional[set] = None):
    structure = pdb_utils.load_structure(pdb_path)
    chain = pdb_utils.get_chain(structure, chain_id)
    if chain is None:
        return None

    aa_counts = np.zeros(len(AA20), dtype=np.float64)
    phi_counts = _counts_init(num_bins_phi)
    psi_counts = _counts_init(num_bins_phi)
    chi_counts = np.zeros((4, num_bins_chi), dtype=np.float64)

    residues = pdb_utils.residues_as_list(chain)
    aa_seq = []
    for residue in residues:
        try:
            aa = PDB.Polypeptide.three_to_one(residue.get_resname())
        except KeyError:
            aa = None
        aa_seq.append(aa)

    mask_indices = set()
    if masked_aas:
        for idx, aa in enumerate(aa_seq):
            if aa in masked_aas:
                mask_indices.add(idx)

    ppb = PDB.PPBuilder()
    for poly in ppb.build_peptides(chain):
        phi_psi = poly.get_phi_psi_list()
        for res, (phi, psi) in zip(poly, phi_psi):
            if phi is None or psi is None:
                continue
            try:
                aa = PDB.Polypeptide.three_to_one(res.get_resname())
            except KeyError:
                continue
            if aa not in AA20_SET:
                continue
            if res in residues:
                idx = residues.index(res)
                if idx in mask_indices:
                    continue
            phi_counts[_hist_index(math.degrees(phi), num_bins_phi)] += 1
            psi_counts[_hist_index(math.degrees(psi), num_bins_phi)] += 1

    for idx, residue in enumerate(residues):
        if idx in mask_indices:
            continue
        try:
            aa = PDB.Polypeptide.three_to_one(residue.get_resname())
        except KeyError:
            continue
        if aa not in AA20_SET:
            continue
        aa_counts[AA20.index(aa)] += 1
        for chi_idx, atom_set in enumerate(_chi_atom_sets(residue.get_resname())):
            if chi_idx >= 4:
                break
            coords = []
            missing = False
            for atom_name in atom_set:
                if atom_name not in residue:
                    missing = True
                    break
                coords.append(residue[atom_name].get_coord())
            if missing:
                continue
            angle = _dihedral(np.array(coords[0]), np.array(coords[1]), np.array(coords[2]), np.array(coords[3]))
            chi_counts[chi_idx, _hist_index(angle, num_bins_chi)] += 1

    return aa_counts, phi_counts, psi_counts, chi_counts


def _normalize(counts: np.ndarray, eps: float) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return None
    probs = counts.astype(np.float64) + eps
    probs /= probs.sum()
    return probs


def _kl(p_ref: np.ndarray, p_gen: np.ndarray) -> Optional[float]:
    if p_ref is None or p_gen is None:
        return None
    return float(np.sum(p_ref * np.log(p_ref / p_gen)))


def build_reference(index_path: Optional[str], pdb_root: str, chain_id: Optional[str],
                    num_bins_phi: int, num_bins_chi: int, masked_aas: Optional[set] = None,
                    chain_source: str = 'fixed') -> KLReference:
    aa_counts = np.zeros(len(AA20), dtype=np.float64)
    phi_counts = _counts_init(num_bins_phi)
    psi_counts = _counts_init(num_bins_phi)
    chi_counts = np.zeros((4, num_bins_chi), dtype=np.float64)

    paths = []
    chain_map = {}
    if index_path:
        with open(index_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts or parts[0] == '':
                    continue
                pdb_id = parts[0]
                pdb_path = os.path.join(pdb_root, pdb_id + '.pdb')
                paths.append(pdb_path)
                if chain_source == 'index_peptide_id' and len(parts) >= 3:
                    chain_map[pdb_path] = parts[2]
    else:
        for fn in os.listdir(pdb_root):
            if fn.endswith('.pdb'):
                paths.append(os.path.join(pdb_root, fn))

    for pdb_path in paths:
        if not os.path.exists(pdb_path):
            continue
        chain = chain_map.get(pdb_path, chain_id)
        if chain is None:
            continue
        result = _collect_counts(pdb_path, chain, num_bins_phi, num_bins_chi, masked_aas=masked_aas)
        if result is None:
            continue
        aa_c, phi_c, psi_c, chi_c = result
        aa_counts += aa_c
        phi_counts += phi_c
        psi_counts += psi_c
        chi_counts += chi_c

    return KLReference(aa_counts, phi_counts, psi_counts, chi_counts)


def save_reference(ref: KLReference, out_path: str):
    data = {
        'aa_counts': ref.aa_counts.tolist(),
        'phi_counts': ref.phi_counts.tolist(),
        'psi_counts': ref.psi_counts.tolist(),
        'chi_counts': ref.chi_counts.tolist(),
    }
    with open(out_path, 'w') as f:
        json.dump(data, f)


def load_reference(path: str) -> KLReference:
    with open(path, 'r') as f:
        data = json.load(f)
    return KLReference(
        aa_counts=np.asarray(data['aa_counts'], dtype=np.float64),
        phi_counts=np.asarray(data['phi_counts'], dtype=np.float64),
        psi_counts=np.asarray(data['psi_counts'], dtype=np.float64),
        chi_counts=np.asarray(data['chi_counts'], dtype=np.float64),
    )


def compute_kl_for_target(pdb_paths: List[str], chain_id: str, ref: KLReference, config: dict,
                          sample_types: Optional[List[str]] = None,
                          masked_by_type: Optional[Dict[str, List[str]]] = None) -> KLMetrics:
    num_bins_phi = config.get('bins_phi_psi', 36)
    num_bins_chi = config.get('bins_chi', 36)
    eps = config.get('smoothing_eps', 1e-8)
    direction = config.get('direction', 'ref')

    gen_aa = np.zeros(len(AA20), dtype=np.float64)
    gen_phi = _counts_init(num_bins_phi)
    gen_psi = _counts_init(num_bins_phi)
    gen_chi = np.zeros((4, num_bins_chi), dtype=np.float64)

    for idx, pdb_path in enumerate(pdb_paths):
        masked = None
        if masked_by_type and sample_types and idx < len(sample_types):
            key = str(sample_types[idx]).upper()
            masked = set(masked_by_type.get(key, []))
        result = _collect_counts(pdb_path, chain_id, num_bins_phi, num_bins_chi, masked_aas=masked)
        if result is None:
            continue
        aa_c, phi_c, psi_c, chi_c = result
        gen_aa += aa_c
        gen_phi += phi_c
        gen_psi += psi_c
        gen_chi += chi_c

    p_ref_aa = _normalize(ref.aa_counts, eps)
    p_gen_aa = _normalize(gen_aa, eps)
    p_ref_phi = _normalize(ref.phi_counts, eps)
    p_gen_phi = _normalize(gen_phi, eps)
    p_ref_psi = _normalize(ref.psi_counts, eps)
    p_gen_psi = _normalize(gen_psi, eps)

    if direction == 'gen':
        aa_kl = _kl(p_gen_aa, p_ref_aa)
        phi_kl = _kl(p_gen_phi, p_ref_phi)
        psi_kl = _kl(p_gen_psi, p_ref_psi)
    else:
        aa_kl = _kl(p_ref_aa, p_gen_aa)
        phi_kl = _kl(p_ref_phi, p_gen_phi)
        psi_kl = _kl(p_ref_psi, p_gen_psi)

    b_kl = None
    if phi_kl is not None and psi_kl is not None:
        b_kl = float((phi_kl + psi_kl) / 2)

    chi_kls = []
    for i in range(4):
        p_ref_chi = _normalize(ref.chi_counts[i], eps)
        p_gen_chi = _normalize(gen_chi[i], eps)
        if direction == 'gen':
            chi_kl = _kl(p_gen_chi, p_ref_chi)
        else:
            chi_kl = _kl(p_ref_chi, p_gen_chi)
        if chi_kl is not None:
            chi_kls.append(chi_kl)
    s_kl = float(sum(chi_kls) / len(chi_kls)) if chi_kls else None

    return KLMetrics(aa_kl=aa_kl, b_kl=b_kl, s_kl=s_kl)
