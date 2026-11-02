"""Evaluate candidate success, quality and diversity with receptor-level coverage."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from utils.file_utils import require_distinct_paths


DEFAULT_METRICS = ['rosetta_total_score', 'vina_score', 'rosetta_dG',
                   'solubility_camsol_score', 'cycpep_proteolysis_sumscore', 'Pproxy', 'Sfinal']
DERIVED_METRICS = {'cyclization_success', 'energy_success', 'valid_fraction',
                   'D_seq', 'D_struct', 'D_co', 'cramers_v'}


def has_convention(value):
    return (isinstance(value, str) and bool(value.strip())) or (isinstance(value, dict) and bool(value))


def distance_matrix(values, n, name, upper=None):
    a = np.asarray(values, dtype=float)
    if a.shape != (n, n) or not np.isfinite(a).all():
        raise ValueError(f'{name}: require a finite {n} x {n} distance matrix.')
    if (a < 0).any() or (upper is not None and (a > upper).any()):
        raise ValueError(f'{name}: distance outside its allowed range.')
    if not np.allclose(a, a.T, atol=1e-8, rtol=0) or not np.allclose(np.diag(a), 0, atol=1e-8, rtol=0):
        raise ValueError(f'{name}: matrix must be symmetric with a zero diagonal.')
    return a


def single_linkage_labels(distances, threshold):
    """Connected components at the inclusive single-linkage distance cut."""
    n = len(distances)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            if distances[i, j] <= threshold:
                parent[root(i)] = root(j)
    mapping, labels = {}, []
    for i in range(n):
        group = root(i)
        mapping.setdefault(group, len(mapping))
        labels.append(mapping[group])
    return labels


def cramers_v(first, second):
    """Cramér's V with the PepGLAD single-cluster consistency convention."""
    if len(first) != len(second):
        raise ValueError('Cluster label lengths differ.')
    n = len(first)
    if n < 2:
        return None
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=float)
    np.add.at(table, (a, b), 1)
    denominator = min(table.shape) - 1
    if denominator == 0:
        return 1.0 if table.shape == (1, 1) else 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * denominator)))


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate(rows, conventions, pairwise=None, metrics=None):
    metrics = list(DEFAULT_METRICS if metrics is None else metrics)
    if not metrics or not all(isinstance(name, str) and name.strip() for name in metrics) or len(metrics) != len(set(metrics)):
        raise ValueError('Metric names must be nonempty and unique.')
    if set(metrics) & DERIVED_METRICS:
        raise ValueError('Select property columns in metrics; success and diversity metrics are included automatically.')
    if not isinstance(conventions, dict) or not has_convention(conventions.get('cyclization_distance')):
        raise ValueError('Record the supplied cyclization_distance measurement convention.')
    groups, seen = {}, set()
    for row in rows:
        sid, receptor = row.get('id'), row.get('receptor_id')
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError('Candidate IDs must be unique nonempty strings.')
        if not isinstance(receptor, str) or not receptor.strip():
            raise ValueError('Each candidate requires receptor_id.')
        seen.add(sid)
        distance, energy = finite_number(row.get('cyclization_distance')), finite_number(row.get('rosetta_dG'))
        if distance is None or distance < 0 or energy is None:
            raise ValueError(f'{sid}: missing/invalid cyclization distance or interface dG; do not omit failed candidates silently.')
        groups.setdefault(receptor, []).append(row)
    if not groups:
        raise ValueError('No candidates to evaluate.')
    if pairwise is not None:
        if pairwise.get('format') != 'apcyc.evaluation_distances.v1':
            raise ValueError('Unsupported pairwise distance format.')
        if not all(has_convention(conventions.get(k)) for k in ['sequence_distance', 'ca_rmsd']):
            raise ValueError('Record sequence-distance and CA-RMSD conventions before clustering.')
        if set(pairwise.get('receptors', {})) != set(groups):
            raise ValueError('Pairwise matrices must cover exactly the measured receptors.')

    receptor_reports, errors = {}, []
    diversity_names = ['D_seq', 'D_struct', 'D_co', 'cramers_v']
    for receptor, candidates in groups.items():
        cyc = [3.0 <= float(row['cyclization_distance']) <= 8.0 for row in candidates]
        energy = [float(row['rosetta_dG']) < 0 for row in candidates]
        valid = [row for row, c, e in zip(candidates, cyc, energy) if c and e]
        values = {name: None for name in metrics}
        values.update(cyclization_success=float(np.mean(cyc)), energy_success=float(np.mean(energy)),
                      valid_fraction=len(valid) / len(candidates), D_seq=None, D_struct=None, D_co=None, cramers_v=None)
        issues, notes = [], []
        if not valid:
            notes.append('no_valid_samples: included in success rates; ineligible for conditional quality metrics')
        else:
            for name in metrics:
                if name == 'rosetta_total_score' and not has_convention(conventions.get('rosetta_total_score')):
                    issues.append('rosetta_total_score: scoring protocol metadata required')
                    continue
                numbers = [finite_number(row.get(name)) for row in valid]
                if any(x is None for x in numbers):
                    issues.append(f'{name}: missing on a valid candidate; no partial mean reported')
                else:
                    values[name] = float(np.mean(numbers))

        if pairwise is not None:
            entry = pairwise['receptors'][receptor]
            ids = entry.get('ids', [])
            if len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in candidates}:
                raise ValueError(f'{receptor}: matrix IDs must match all candidate IDs exactly.')
            seq = distance_matrix(entry['sequence_distance'], len(ids), 'sequence_distance', upper=1)
            struct = distance_matrix(entry['ca_rmsd'], len(ids), 'ca_rmsd')
            if len(valid) >= 2:
                lookup = {sid: i for i, sid in enumerate(ids)}
                indices = [lookup[row['id']] for row in valid]
                seq_labels = single_linkage_labels(seq[np.ix_(indices, indices)], 0.4)
                structure_labels = single_linkage_labels(struct[np.ix_(indices, indices)], 4.0)
                values['D_seq'] = len(set(seq_labels)) / len(valid)
                values['D_struct'] = len(set(structure_labels)) / len(valid)
                values['D_co'] = math.sqrt(values['D_seq'] * values['D_struct'])
                values['cramers_v'] = cramers_v(seq_labels, structure_labels)
        elif len(valid) >= 2:
            issues.append('pairwise_distances_not_supplied')
        if len(valid) < 2:
            notes.append('fewer_than_two_valid_samples: ineligible for diversity and consistency')
        receptor_reports[receptor] = dict(candidates=len(candidates), valid=len(valid),
                                          valid_ids=[row['id'] for row in valid], metrics=values, issues=issues, notes=notes)
        errors.extend(f'{receptor}: {issue}' for issue in issues)

    all_names = list(next(iter(receptor_reports.values()))['metrics'])
    aggregate, coverage = {}, {}
    for name in all_names:
        minimum = 2 if name in diversity_names else (1 if name in metrics else 0)
        eligible = [r for r in receptor_reports.values() if r['valid'] >= minimum]
        per_receptor = [r['metrics'][name] for r in eligible]
        reported = sum(x is not None for x in per_receptor)
        aggregate[name] = None if not per_receptor or reported != len(per_receptor) else float(np.mean(per_receptor))
        coverage[name] = dict(eligible_receptors=len(eligible), reported_receptors=reported,
                              total_receptors=len(groups), eligible_fraction=len(eligible) / len(groups),
                              reported_fraction=reported / len(groups))
    return dict(format='apcyc.paper_evaluation.v2', status='complete' if not errors else 'incomplete',
                conventions=conventions, receptor_count=len(groups), candidate_count=len(rows),
                aggregation='equal-weight means over metric-eligible receptors; all receptors in success rates',
                receptor_reports=receptor_reports, aggregate=aggregate, coverage=coverage, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', required=True, type=Path)
    parser.add_argument('--conventions', required=True, type=Path, help='JSON describing supplied measurement/alignment definitions.')
    parser.add_argument('--distances', type=Path, help='Optional receptor-keyed precomputed distance matrices.')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    require_distinct_paths([args.output], [args.measurements, args.conventions, args.distances])
    with args.measurements.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Missing or duplicate CSV headers.')
        rows = list(reader)
    conventions = json.loads(args.conventions.read_text())
    pairwise = json.loads(args.distances.read_text()) if args.distances else None
    report = evaluate(rows, conventions, pairwise, args.metrics)
    report['inputs'] = {key: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for key, path in [('measurements', args.measurements), ('conventions', args.conventions),
                                          ('distances', args.distances)] if path is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 2 if args.require_complete and report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
