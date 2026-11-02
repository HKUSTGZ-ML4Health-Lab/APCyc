"""Measure cyclization and pairwise distances from a prediction manifest."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from utils.file_utils import require_distinct_paths


CONVENTIONS = {
    'protocol': 'apcyc.evaluation.v1',
    'cyclization_distance': 'Euclidean distance in angstroms between the two selected residue CA atoms; peptide-local zero-based pair_hat.',
    'sequence_distance': '1 - global BLOSUM62 alignment score / sqrt(self_score_A * self_score_B); all gap scores are explicitly zero.',
    'ca_rmsd': 'Peptide CA RMSD in the shared receptor coordinate frame, using residue order and equal peptide lengths; no peptide superposition or cyclic permutation.',
    'aggregation': 'Success rates over all candidates and all receptors; property means over receptors with at least one valid candidate; diversity and consistency over receptors with at least two valid candidates. Receptors have equal weight and coverage is reported.',
    'consistency': 'Uncorrected Cramers V; two single-cluster partitions score 1, exactly one single-cluster partition scores 0; fewer than two valid samples are ineligible.',
}


def sequence_distance(first, second):
    """Normalized global BLOSUM62 alignment distance."""
    alphabet = set('ACDEFGHIKLMNPQRSTVWY')
    if not first or not second or set(first + second) - alphabet:
        raise ValueError('Sequences must contain canonical amino acids.')
    aligner = PairwiseAligner(mode='global')
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    scale = math.sqrt(aligner.score(first, first) * aligner.score(second, second))
    value = 1.0 - aligner.score(first, second) / scale
    if not -1e-10 <= value <= 1.0 + 1e-10:
        raise ValueError('Normalized sequence distance outside [0, 1].')
    return float(np.clip(value, 0, 1))


def ca_rmsd(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0:
        raise ValueError('CA RMSD requires equal nonzero peptide lengths.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite CA coordinates.')
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def read_structure(path, row):
    structure = PDBParser(QUIET=True).get_structure('candidate', str(path))
    if len(structure) != 1:
        raise ValueError('Evaluation requires one PDB model.')
    pep_chain = row.get('peptide_chain', 'L')
    residues = [r for r in structure[0][pep_chain] if r.id[0] == ' ']
    sequence = ''.join(seq1(r.resname) for r in residues)
    if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
        raise ValueError('Peptide contains no residues or unsupported residue identities.')
    if row.get('sequence') is not None and row['sequence'] != sequence:
        raise ValueError('Manifest sequence differs from the measured PDB.')
    coords = np.asarray([r['CA'].coord for r in residues], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite peptide coordinates.')
    pair = row.get('pair_hat')
    if (row.get('pair_index_base') != 0 or not isinstance(pair, list) or len(pair) != 2
            or any(type(i) is not int for i in pair) or not 0 <= pair[0] < pair[1] < len(residues)):
        raise ValueError('Require ordered, distinct, zero-based peptide-local pair_hat.')
    receptor = {}
    for chain in structure[0]:
        if chain.id == pep_chain:
            continue
        for residue in chain:
            if residue.id[0] == ' ' and 'CA' in residue:
                receptor[(chain.id, residue.id, residue.resname)] = np.asarray(residue['CA'].coord, dtype=float)
    if not receptor or not all(np.isfinite(x).all() for x in receptor.values()):
        raise ValueError('Require finite receptor CA coordinates.')
    return sequence, coords, receptor, float(np.linalg.norm(coords[pair[0]] - coords[pair[1]]))


def measure(manifest, properties, rosetta_protocol=None):
    manifest, properties = Path(manifest), Path(properties)
    with properties.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate property CSV headers.')
        property_rows = list(reader)
    scores = {}
    for row in property_rows:
        sid = row.get('id')
        if not sid or sid in scores:
            raise ValueError('Property IDs must be unique and nonempty.')
        scores[sid] = row
    groups, rows, seen, source_hashes = {}, [], set(), {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row['pdb'])
        path = path if path.is_absolute() else manifest.parent / path
        sid = row['id'] if 'id' in row else path.stem
        receptor_id = row['receptor_id'] if 'receptor_id' in row else row.get('sample_id')
        if not isinstance(sid, str) or not sid.strip() or sid in seen or sid not in scores:
            raise ValueError('Candidate IDs must be unique and have a property row.')
        if not isinstance(receptor_id, str) or not receptor_id.strip():
            raise ValueError('Each candidate requires a receptor_id or sample_id.')
        seen.add(sid)
        sequence, coords, receptor, distance = read_structure(path, row)
        if 'sequence' in scores[sid] and scores[sid]['sequence'] != sequence:
            raise ValueError('Property CSV sequence differs from the measured PDB.')
        group = groups.setdefault(receptor_id, [])
        if group:
            reference = group[0]
            if len(coords) != len(reference['coords']):
                raise ValueError(f'{receptor_id}: use one peptide length per receptor evaluation group.')
            if set(receptor) != set(reference['receptor']) or any(
                    np.linalg.norm(receptor[k] - reference['receptor'][k]) > 0.003 for k in receptor):
                raise ValueError(f'{receptor_id}: receptor coordinates or residue identities differ.')
        group.append(dict(id=sid, sequence=sequence, coords=coords, receptor=receptor))
        rows.append({**scores[sid], 'id': sid, 'receptor_id': receptor_id,
                     'sequence': sequence, 'cyclization_distance': distance})
        source_hashes[sid] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows or set(scores) != seen:
        raise ValueError('Manifest and property CSV must contain exactly the same candidate IDs.')
    pairwise = dict(format='apcyc.evaluation_distances.v1', receptors={})
    for rid, group in groups.items():
        n = len(group)
        seq, structure = np.zeros((n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                seq[i, j] = seq[j, i] = sequence_distance(group[i]['sequence'], group[j]['sequence'])
                structure[i, j] = structure[j, i] = ca_rmsd(group[i]['coords'], group[j]['coords'])
        pairwise['receptors'][rid] = dict(ids=[x['id'] for x in group], sequence_distance=seq.tolist(), ca_rmsd=structure.tolist())
    conventions = dict(CONVENTIONS)
    if rosetta_protocol is not None:
        provenance = json.loads(Path(rosetta_protocol).read_text())
        if provenance.get('format') != 'apcyc.rosetta_protocol.v1' or not provenance.get('output_score_function'):
            raise ValueError('Require Rosetta adapter protocol metadata with an explicit output score function.')
        if provenance.get('candidate_ids') != sorted(seen):
            raise ValueError('Rosetta protocol metadata candidate IDs differ from the evaluation set.')
        if provenance.get('structures_sha256') != source_hashes:
            raise ValueError('Rosetta protocol metadata structure hashes differ from the measured PDBs.')
        for row in rows:
            for key in ['rosetta_dG', 'rosetta_total_score']:
                actual = float(row.get(key, 'nan'))
                expected = float(provenance.get('scores', {}).get(row['id'], {}).get(key, 'nan'))
                if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f'{row["id"]}: property CSV differs from Rosetta protocol scores.')
        conventions['rosetta_total_score'] = provenance
    return rows, conventions, pairwise, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--properties', required=True, type=Path)
    parser.add_argument('--rosetta-protocol', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    require_distinct_paths([args.out_dir / name for name in
                            ['measurements.csv', 'conventions.json', 'distances.json', 'inputs.json']],
                           [args.manifest, args.properties, args.rosetta_protocol])
    rows, conventions, pairwise, hashes = measure(args.manifest, args.properties, args.rosetta_protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out_dir / 'measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    for name, value in [('conventions.json', conventions), ('distances.json', pairwise),
                        ('inputs.json', dict(structures_sha256=hashes, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                            properties_sha256=hashlib.sha256(args.properties.read_bytes()).hexdigest()))]:
        (args.out_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
