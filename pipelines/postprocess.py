"""Validate predicted topology and optionally relax it without changing sequence or sites."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree

from pipelines.process_utils import run_isolated
from utils.file_utils import sha256 as digest
from apcyc.chemistry import is_iso_pair


def peptide(path, chain):
    structure = PDBParser(QUIET=True).get_structure('prediction', str(path))
    if len(structure) != 1:
        raise ValueError('exactly_one_PDB_model_required')
    residues = [r for r in structure[0][chain] if r.id[0] == ' ']
    if not residues:
        raise ValueError('empty_peptide_chain')
    if any(not all(name in r for name in ['N', 'CA', 'C', 'O']) for r in residues):
        raise ValueError('incomplete_peptide_backbone')
    coords = np.array([a.coord for a in structure.get_atoms()])
    if not np.isfinite(coords).all():
        raise ValueError('nonfinite_coordinates')
    return structure, residues


def closure_spec(row, residues):
    if row.get('pair_index_base') != 0:
        raise ValueError('pair_index_base_must_be_explicitly_zero')
    pair = row.get('pair_hat')
    if not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair):
        raise ValueError('invalid_pair_indices')
    i, j = pair
    if not 0 <= i < j < len(residues):
        raise ValueError('pair_must_be_ordered_distinct_and_in_range')
    typ = row.get('type_hat')
    if typ == 'HT':
        if (i, j) != (0, len(residues) - 1):
            raise ValueError('head_tail_pair_must_be_terminal')
        return dict(type=typ, indices=[j, i], atoms=['C', 'N'], bounds=[1.20, 1.50])
    if typ == 'SS':
        if residues[i].resname != 'CYS' or residues[j].resname != 'CYS':
            raise ValueError('disulfide_requires_Cys_Cys')
        if j - i < 3:
            raise ValueError('disulfide_pair_too_close_in_sequence')
        return dict(type=typ, indices=[i, j], atoms=['SG', 'SG'], bounds=[1.85, 2.20])
    if typ == 'ISO':
        # Isopeptide endpoints must be Lys and Asp/Asn.
        if not is_iso_pair(residues[i].resname, residues[j].resname):
            raise ValueError('isopeptide_requires_Lys_Asp_or_Asn')
        if residues[i].resname == 'LYS':
            k, d = i, j
        else:
            k, d = j, i
        return dict(type=typ, indices=[k, d], atoms=['NZ', 'CG'], bounds=[1.20, 1.50])
    raise ValueError('unsupported_cyclization_type')


def handedness(residue):
    if residue.resname == 'GLY' or not all(a in residue for a in ['N', 'CA', 'C', 'CB']):
        return None
    ca = residue['CA'].coord
    return float(np.linalg.det(np.stack([residue[a].coord - ca for a in ['N', 'C', 'CB']])))


def validate_output(source, output, row, spec):
    before, source_residues = peptide(source, row.get('peptide_chain', 'L'))
    after, residues = peptide(output, row.get('peptide_chain', 'L'))
    if [(r.id, r.resname) for r in residues] != [(r.id, r.resname) for r in source_residues]:
        raise ValueError('relaxation_changed_residue_identity_or_numbering')
    # Receptor coordinates must remain fixed; newly added hydrogens are allowed.
    pep_chain = row.get('peptide_chain', 'L')
    for chain in before[0]:
        if chain.id == pep_chain:
            continue
        for residue in chain:
            for atom in residue:
                target = after[0][chain.id][residue.id][atom.name]
                if np.linalg.norm(target.coord - atom.coord) > 0.003:
                    raise ValueError('relaxation_moved_receptor')
    i, j = spec['indices']
    a, b = [residues[k][name] for k, name in zip([i, j], spec['atoms'])]
    distance = float(np.linalg.norm(a.coord - b.coord))
    if not spec['bounds'][0] <= distance <= spec['bounds'][1]:
        raise ValueError(f'closure_bond_out_of_range:{distance:.4f}')
    connections = set()
    for line in output.read_text().splitlines():
        if line.startswith('CONECT'):
            nums = [int(line[k:k+5]) for k in range(6, len(line), 5) if line[k:k+5].strip()]
            connections.update(tuple(sorted((nums[0], n))) for n in nums[1:])
    if tuple(sorted((a.serial_number, b.serial_number))) not in connections:
        raise ValueError('closure_bond_missing_from_topology')
    for old, new in zip(source_residues, residues):
        h0, h1 = handedness(old), handedness(new)
        if h0 is not None and (abs(h0) < 0.1 or h1 is None or abs(h1) < 0.1 or h0 * h1 < 0):
            raise ValueError('undefined_or_changed_CA_handedness')
    # Conservative severe-contact screen. It is not a full stereochemical proof.
    atoms = [a for a in after.get_atoms() if a.element not in {'H', 'D'}]
    coords = np.asarray([a.coord for a in atoms])
    severe = []
    for x, y in cKDTree(coords).query_pairs(1.0):
        if atoms[x].parent.parent.id != pep_chain and atoms[y].parent.parent.id != pep_chain:
            continue
        severe.append([atoms[x].serial_number, atoms[y].serial_number])
    if severe:
        raise ValueError(f'severe_heavy_atom_contact:{len(severe)}')
    return dict(closure_distance_angstrom=distance, explicit_bond=True,
                sequence_preserved=True, receptor_preserved=True, CA_handedness_preserved=True,
                severe_contacts=0, chemical_closure_validated=True,
                validation_scope='bond, identity, CA-handedness preservation and severe contacts; not exhaustive chemistry')


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Choose an empty postprocessing output directory.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.predictions.resolve()
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty prediction manifest')
    accepted, reports, seen = [], [], set()
    for row in rows:
        name = Path(row['pdb']).name
        if name in seen:
            raise ValueError('Duplicate prediction filename')
        seen.add(name)
        source = (predictions.parent / row['pdb']).resolve()
        if not source.is_relative_to(predictions.parent):
            raise ValueError('Prediction PDB must be inside the prediction directory')
        report = dict(pdb=name, type_hat=row.get('type_hat'),
                      pair_hat=row.get('pair_hat'), chemical_closure_validated=False)
        try:
            report['input_sha256'] = digest(source)
            _, residues = peptide(source, row.get('peptide_chain', 'L'))
            sequence = ''.join(seq1(r.resname) for r in residues)
            if sequence != row['sequence']:
                raise ValueError('manifest_sequence_differs_from_PDB')
            spec = closure_spec(row, residues)
            for k, atom in zip(spec['indices'], spec['atoms']):
                if atom not in residues[k]:
                    raise ValueError(f'missing_closure_atom:{k}:{atom}')
            report['closure'] = spec
            report['status'] = 'topology_eligible_not_relaxed'
            if args.relax_python:
                if not args.cpsea_relax_dir:
                    raise ValueError('--cpsea-relax-dir is required with --relax-python')
                output = args.out_dir / name
                request = args.out_dir / (Path(name).stem + '.request.json')
                request.write_text(json.dumps(dict(source=str(source), output=str(output),
                                                   chain=row.get('peptide_chain', 'L'), closure=spec)))
                adapter = Path(__file__).parent / 'adapters/relax.py'
                with (args.out_dir / (Path(name).stem + '.log')).open('w') as log:
                    run_isolated([str(args.relax_python), str(adapter), '--request', str(request),
                                    '--cpsea-relax-dir', str(args.cpsea_relax_dir)],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                report.update(validate_output(source, output, row, spec))
                report.update(status='relaxed_and_validated', output_sha256=digest(output))
                accepted_row = {**row, 'pdb': name, 'structure_status': 'relaxed_and_validated',
                                'chemical_closure_validated': True}
                if row.get('pocket_json'):
                    pocket = Path(row['pocket_json'])
                    accepted_row['pocket_json'] = str((pocket if pocket.is_absolute() else predictions.parent / pocket).resolve())
                accepted.append(accepted_row)
        except Exception as exc:
            report.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        reports.append(report)
    (args.out_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
    summary = dict(samples=len(rows), accepted=len(accepted), failed=sum(r['status']=='failed' for r in reports),
                   relaxation_requested=bool(args.relax_python), records=reports,
                   input_manifest_sha256=digest(predictions), sequence_mutation_allowed=False)
    (args.out_dir / 'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}))
    if args.strict and (summary['failed'] or len(accepted) != len(rows)):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--relax-python', type=Path)
    p.add_argument('--cpsea-relax-dir', type=Path)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--strict', action='store_true')
    run(p.parse_args())
