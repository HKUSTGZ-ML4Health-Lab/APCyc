#!/usr/bin/env python3
"""Validate generated files and report raw geometry without implying chemical validity."""
import argparse
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apcyc.chemistry import is_iso_pair


def inspect(directory):
    rows = [json.loads(s) for s in (directory / 'predictions.jsonl').read_text().splitlines()]
    report = []
    for r in rows:
        path = directory / r['pdb']
        structure = PDBParser(QUIET=True).get_structure(r['pdb'], str(path))[0]
        peptide = list(structure[r['peptide_chain']])
        if len(peptide) != r['pep_len']:
            raise ValueError('Output length disagrees with prediction metadata.')
        if not all(all(k in res for k in ['N', 'CA', 'C', 'O']) for res in peptide):
            raise ValueError('Backbone atom identity lost in PDB.')
        coords = np.array([a.coord for a in structure.get_atoms()])
        if not np.isfinite(coords).all():
            raise ValueError('Nonfinite output coordinates.')
        i, j = r['pair_hat']
        if not (0 <= i < len(peptide) and 0 <= j < len(peptide)):
            raise ValueError('Pair index is not peptide local.')
        if [peptide[k].resname for k in [i, j]] != r['pair_hat_resnames']:
            raise ValueError('Pair residue names mismatch.')
        consecutive = [float(np.linalg.norm(a['C'].coord - b['N'].coord)) for a, b in zip(peptide, peptide[1:])]
        cycle = r['type_hat']
        if cycle == 'HT':
            distance = float(np.linalg.norm(peptide[-1]['C'].coord - peptide[0]['N'].coord))
            compatible = {i, j} == {0, len(peptide) - 1}
        elif cycle == 'SS':
            compatible = all(peptide[k].resname == 'CYS' and 'SG' in peptide[k] for k in [i, j])
            distance = float(np.linalg.norm(peptide[i]['SG'].coord - peptide[j]['SG'].coord)) if compatible else None
        elif cycle == 'ISO':
            compatible = i != j and is_iso_pair(peptide[i].resname, peptide[j].resname)
            distance = None  # No atom selection / chemical bond construction is invented here.
        else:
            raise ValueError('Unsupported cyclization type.')
        report.append(dict(pdb=r['pdb'], sequence=r['sequence'], finite_coordinates=True,
                           output_residues=len(peptide), type=cycle, pair=r['pair_hat'],
                           pair_residue_compatibility=compatible, endpoint_distance_angstrom=distance,
                           consecutive_CN_min=min(consecutive), consecutive_CN_max=max(consecutive),
                           chemical_closure_validated=False))
    return dict(samples=len(report), files=report, status='raw model output; no chemical relaxation or covalent bond validation')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    result=inspect(a.directory)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    else:
        print(json.dumps(result, indent=2))
