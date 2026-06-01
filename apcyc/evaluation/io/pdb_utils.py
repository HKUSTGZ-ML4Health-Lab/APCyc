#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from Bio import PDB

PDBParser = PDB.PDBParser
PPBuilder = PDB.PPBuilder
is_aa = PDB.Polypeptide.is_aa
try:
    from Bio.PDB.Polypeptide import three_to_one as _three_to_one  # Biopython <1.83
    def three_to_one(resname: str) -> str:
        return _three_to_one(resname)
except Exception:
    # Biopython >=1.83 removed three_to_one; fall back to IUPAC mapping
    from Bio.Data.IUPACData import protein_letters_3to1_extended
    _AA3_TO_1 = {k.upper(): v for k, v in protein_letters_3to1_extended.items()}
    def three_to_one(resname: str) -> str:
        key = resname.upper()
        if key in _AA3_TO_1:
            return _AA3_TO_1[key]
        raise KeyError(key)


def load_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(pdb_path), pdb_path)


def get_chain(structure, chain_id: Optional[str] = None):
    model = next(structure.get_models())
    if chain_id is None:
        return next(model.get_chains(), None)
    return model.child_dict.get(chain_id)


def iter_residues(chain) -> Iterable[PDB.Residue.Residue]:
    for residue in chain.get_residues():
        if residue.id[0] != ' ':
            continue
        if not is_aa(residue, standard=True):
            continue
        yield residue


def residues_as_list(chain) -> List[PDB.Residue.Residue]:
    return list(iter_residues(chain))


def extract_sequence(chain) -> str:
    seq = []
    for residue in iter_residues(chain):
        resname = residue.get_resname()
        try:
            seq.append(three_to_one(resname))
        except KeyError:
            continue
    return ''.join(seq)


def extract_ca_coords(chain) -> Tuple[np.ndarray, List[int]]:
    coords = []
    idx_map = []
    for idx, residue in enumerate(iter_residues(chain)):
        if 'CA' not in residue:
            continue
        coords.append(residue['CA'].get_coord())
        idx_map.append(idx)
    if not coords:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.asarray(coords, dtype=np.float32), idx_map


def write_ca_pdb(chain, out_path: str, chain_id: str = 'A') -> int:
    residues = residues_as_list(chain)
    lines = []
    atom_idx = 1
    res_idx = 1
    for residue in residues:
        if 'CA' not in residue:
            res_idx += 1
            continue
        x, y, z = residue['CA'].get_coord()
        resname = residue.get_resname()
        line = (
            f"ATOM  {atom_idx:5d}  CA  {resname:>3s} {chain_id:1s}{res_idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
        lines.append(line)
        atom_idx += 1
        res_idx += 1
    with open(out_path, 'w') as f:
        f.writelines(lines)
        f.write('END\n')
    return atom_idx - 1


def get_atom_coord(residue, atom_name: str) -> Optional[np.ndarray]:
    if atom_name not in residue:
        return None
    return residue[atom_name].get_coord()


def iter_phi_psi(chain):
    ppb = PPBuilder()
    for poly in ppb.build_peptides(chain):
        phi_psi = poly.get_phi_psi_list()
        for residue, angles in zip(poly, phi_psi):
            if angles is None:
                continue
            phi, psi = angles
            yield residue, phi, psi


def split_pdb_by_chain(pdb_path: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    structure = load_structure(pdb_path)
    model = next(structure.get_models())
    outputs = {}
    for chain in model:
        out_path = os.path.join(out_dir, f"{os.path.basename(pdb_path)}.{chain.id}.pdb")
        io = PDB.PDBIO()
        io.set_structure(chain)
        io.save(out_path)
        outputs[chain.id] = out_path
    return outputs


def read_ca_trace(pdb_path: str, chain_id: Optional[str] = None) -> Optional[np.ndarray]:
    structure = load_structure(pdb_path)
    chain = get_chain(structure, chain_id)
    if chain is None:
        return None
    coords, _ = extract_ca_coords(chain)
    return coords
