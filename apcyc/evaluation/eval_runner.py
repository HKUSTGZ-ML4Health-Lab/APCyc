#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import yaml
from tqdm import tqdm

from apcyc.evaluation.io import aggregate, report
from apcyc.evaluation.metrics.affinity.metric_rosetta_interface_dg import compute_rosetta_interface_dg
from apcyc.evaluation.metrics.affinity.metric_rosetta_total_score import compute_rosetta_total_score
from apcyc.evaluation.metrics.affinity.metric_vina_score import compute_vina_score
from apcyc.evaluation.metrics.aggregation.metric_aggregate_per_sample import aggregate_per_sample_metrics
from apcyc.evaluation.metrics.aggregation.metric_succ_cyclization import compute_succ_cyclization
from apcyc.evaluation.metrics.aggregation.metric_succ_energy_cyclization import compute_succ_energy_cyclization
from apcyc.evaluation.metrics.aggregation.metric_success_rows import select_success_rows
from apcyc.evaluation.metrics.aggregation.metric_valid_rate import compute_valid_rate
from apcyc.evaluation.metrics.consistency.metric_cramers_v import compute_cramers_v
from apcyc.evaluation.metrics.diversity.metric_co_diversity import compute_co_diversity
from apcyc.evaluation.metrics.diversity.metric_tm_mean import compute_tm_mean
from apcyc.evaluation.metrics.diversity.metric_tm_median import compute_tm_median
from apcyc.evaluation.metrics.geometry.metric_rama_allowed import compute_rama_allowed
from apcyc.evaluation.metrics.geometry.metric_rama_favored import compute_rama_favored
from apcyc.evaluation.metrics.helpers.metric_failure_counts import attach_failure_counts
from apcyc.evaluation.metrics.helpers.metric_helpers import (
    extract_sequence_and_ca,
    load_sample_chains,
    resolve_pair_tuple,
    resolve_pdb_path,
)
from apcyc.evaluation.metrics.stability.metric_bond_length_validity import compute_bond_length_validity
from apcyc.evaluation.metrics.stability.metric_cyclization_success import compute_cyclization_success
from apcyc.evaluation.metrics.stability.metric_energy_success import compute_energy_success


METRIC_INFO = {
    'rosetta_interface_dg': {'higher_better': False},
    'rosetta_total_score': {'higher_better': False},
    'vina_score': {'higher_better': False},
    'ramachandran_allowed': {'higher_better': True},
    'ramachandran_favored': {'higher_better': True},
    'cyclization_success': {'higher_better': True},
    'bond_length_validity': {'higher_better': True},
    'energy_success': {'higher_better': True},
    'succ_cyclization': {'higher_better': True},
    'succ_energy_cyclization': {'higher_better': True},
    'tm_mean': {'higher_better': True},
    'tm_median': {'higher_better': True},
    'cramers_v': {'higher_better': True},
    'co_diversity': {'higher_better': True},
}


def parse_args():
    parser = argparse.ArgumentParser(description='APCyc evaluation runner')
    parser.add_argument('--manifest', help='path to apcyc_predictions.jsonl')
    parser.add_argument('--generated_dir', help='directory with target_id/sample pdb layout')
    parser.add_argument('--flat_targets', action='store_true',
                        help='treat each pdb as its own target (ignore subfolders)')
    parser.add_argument('--reference_mode', action='store_true',
                        help='compute metrics without validity filtering (for reference sets)')
    parser.add_argument('--peptide_chain', default=None,
                        help='Override peptide chain ID (default: from config or L)')
    parser.add_argument('--receptor_chain', default=None,
                        help='Override receptor chain ID (default: from config or R)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: config parallel.num_workers or 1)')
    parser.add_argument('--outdir', required=True, help='output directory')
    parser.add_argument('--config', required=True, help='evaluation config yaml')
    return parser.parse_args()


def load_manifest(path: str) -> List[dict]:
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    return items


def build_manifest_from_dir(generated_dir: str, flat_targets: bool = False) -> List[dict]:
    generated_dir = os.path.abspath(generated_dir)
    items = []
    for root, _, files in os.walk(generated_dir):
        for name in files:
            if not name.lower().endswith(('.pdb', '.cif')):
                continue
            pdb_path = os.path.join(root, name)
            sample_id = os.path.splitext(name)[0]
            if flat_targets:
                target_id = sample_id
            else:
                parent = os.path.basename(os.path.dirname(pdb_path))
                if parent and parent != os.path.basename(generated_dir):
                    target_id = parent
                else:
                    target_id = sample_id
            items.append({
                'target_id': target_id,
                'sample_id': sample_id,
                'pdb': pdb_path,
            })
    return items


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _metric_direction(metric_name: str) -> bool:
    base_name = metric_name
    for prefix in ('mean_over_valid.', 'best_of_k.'):
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break
    if base_name.startswith('failure.'):
        return False
    if base_name in METRIC_INFO:
        return METRIC_INFO[base_name].get('higher_better', True)
    return True


def eval_target(target_id: str, samples: List[dict], cfg: dict):
    failures = defaultdict(int)
    metrics_per_sample = []

    peptide_chain = cfg.get('peptide_chain', 'L')
    rec_chain = cfg.get('receptor_chain', 'R')
    validity_filter = cfg.get('validity_filter', True)
    force_valid_all = cfg.get('force_valid_all', False)

    rama_cfg = cfg.get('rama', {})
    rama_data_dir = rama_cfg.get('data_dir', os.path.join('CPSea', 'Model_Generation_and_Evaluation', 'Evaluation', 'data'))

    cycl_cfg = cfg.get('cyclization', {})
    cb_low = cycl_cfg.get('cb_cb_low', 3.0)
    cb_high = cycl_cfg.get('cb_cb_high', 8.0)
    fallback_atom = cycl_cfg.get('fallback_atom', 'CA')
    assume_terminal_pair = cycl_cfg.get('assume_terminal_pair_if_missing', False)
    ppflow_cfg = cfg.get('ppflow_validity', {})
    ppflow_enabled = ppflow_cfg.get('enabled', True)
    ppflow_tol = ppflow_cfg.get('tol', 0.5)

    rosetta_cfg = cfg.get('rosetta', {})
    vina_cfg = cfg.get('vina', {}) or {'enabled': False}
    skip_energy_metrics = cfg.get('skip_energy_metrics', False)

    seqs = []
    cas = []
    valid_pdbs = []
    valid_types = []
    rama_cache = {}

    for sample in samples:
        pdb_path = resolve_pdb_path(sample)
        if not pdb_path or not os.path.exists(pdb_path):
            failures['missing_pdb'] += 1
            continue

        pep_chain_obj, rec_chain_obj = load_sample_chains(pdb_path, peptide_chain, rec_chain)
        if pep_chain_obj is None:
            failures['missing_chain'] += 1
            continue
        seq, ca_coords = extract_sequence_and_ca(pep_chain_obj)
        pair_tuple = resolve_pair_tuple(sample.get('pair_hat'), len(seq), assume_terminal_pair)

        rama_allowed = compute_rama_allowed(
            pdb_path,
            peptide_chain,
            rama_data_dir,
            failures,
            cache=rama_cache,
        )
        rama_favored = compute_rama_favored(
            pdb_path,
            peptide_chain,
            rama_data_dir,
            failures,
            cache=rama_cache,
        )
        cycl_ok = compute_cyclization_success(
            pdb_path,
            peptide_chain,
            pair_tuple,
            cb_low,
            cb_high,
            fallback_atom,
            failures,
        )
        bond_ok = compute_bond_length_validity(pdb_path, peptide_chain, ppflow_enabled, ppflow_tol, failures)
        valid = True if force_valid_all else bool(cycl_ok)
        rosetta_score_val = None
        rosetta_total_score_val = None
        vina_score_val = None

        if skip_energy_metrics:
            energy_success = True
        else:
            if not validity_filter or valid:
                rosetta_score_val = compute_rosetta_interface_dg(
                    pdb_path,
                    peptide_chain,
                    rec_chain,
                    rec_chain_obj,
                    rosetta_cfg,
                    failures,
                )
                if rosetta_cfg.get('also_total_score', False):
                    rosetta_total_score_val = compute_rosetta_total_score(
                        pdb_path,
                        peptide_chain,
                        rec_chain,
                        rec_chain_obj,
                        rosetta_cfg,
                        failures,
                    )
                vina_score_val = compute_vina_score(pdb_path, vina_cfg, failures)

            energy_success = compute_energy_success(rosetta_score_val)
        metrics_per_sample.append({
            'target_id': target_id,
            'sample_id': sample.get('sample_id'),
            'sample_idx': sample.get('sample_idx'),
            'pdb': pdb_path,
            'type_hat': sample.get('type_hat'),
            'pair_hat': pair_tuple,
            'valid': valid,
            'ramachandran_allowed': rama_allowed,
            'ramachandran_favored': rama_favored,
            'cyclization_success': cycl_ok,
            'bond_length_validity': bond_ok,
            'energy_success': energy_success,
            'rosetta_interface_dg': rosetta_score_val,
            'rosetta_total_score': rosetta_total_score_val,
            'vina_score': vina_score_val,
        })

        if valid:
            seqs.append(seq)
            cas.append(ca_coords)
            valid_pdbs.append(pdb_path)
            valid_types.append(sample.get('type_hat'))

    target_metrics = {
        'target_id': target_id,
        'valid_count': len(valid_pdbs),
        'sample_count': len(samples),
    }
    target_metrics['valid_rate'] = compute_valid_rate(metrics_per_sample)

    # Success rate (per target): at least one peptide satisfies constraints
    succ_rows = select_success_rows(metrics_per_sample, samples, cfg.get('K', None))
    target_metrics['succ_cyclization'] = compute_succ_cyclization(succ_rows)
    target_metrics['succ_energy_cyclization'] = compute_succ_energy_cyclization(succ_rows)

    # Aggregate per-sample metrics
    aggregation_modes = cfg.get('aggregation', ['mean_over_valid'])
    target_metrics.update(
        aggregate_per_sample_metrics(metrics_per_sample, aggregation_modes, validity_filter, _metric_direction)
    )

    diversity_cache = {}
    tm_cache = {}
    target_metrics.update({
        'co_diversity': compute_co_diversity(valid_pdbs, seqs, cas, cfg, cache=diversity_cache),
        'cramers_v': compute_cramers_v(valid_pdbs, seqs, cas, cfg, cache=diversity_cache),
        'tm_mean': compute_tm_mean(valid_pdbs, peptide_chain, cfg, cache=tm_cache),
        'tm_median': compute_tm_median(valid_pdbs, peptide_chain, cfg, cache=tm_cache),
    })
    attach_failure_counts(target_metrics, failures)

    return target_metrics, metrics_per_sample, failures


def _eval_target_job(payload):
    target_id, samples, cfg = payload
    t_metrics, s_rows, failures = eval_target(target_id, samples, cfg)
    return target_id, t_metrics, s_rows, failures


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg['manifest'] = args.manifest
    cfg['generated_dir'] = args.generated_dir
    cfg['outdir'] = args.outdir
    if args.peptide_chain:
        cfg['peptide_chain'] = args.peptide_chain
    if args.receptor_chain:
        cfg['receptor_chain'] = args.receptor_chain
    if args.reference_mode:
        cfg['validity_filter'] = False
    cfg.setdefault('cyclization', {}).setdefault('assume_terminal_pair_if_missing', True)

    if not args.manifest and not args.generated_dir:
        raise ValueError('Either --manifest or --generated_dir must be provided.')
    if args.generated_dir:
        manifest = build_manifest_from_dir(args.generated_dir, flat_targets=args.flat_targets)
    else:
        manifest = load_manifest(args.manifest)

    targets = defaultdict(list)
    for item in manifest:
        target_id = item.get('receptor_id') or item.get('target_id') or item.get('sample_id')
        targets[str(target_id)].append(item)

    ref_failures = {}

    metrics_target_rows = []
    metrics_sample_rows = []
    failure_counts = defaultdict(int)

    show_pbar = cfg.get('pbar', True)
    target_items = list(targets.items())
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cfg.get('parallel', {}).get('num_workers', 1)
    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError):
        num_workers = 1
    num_workers = max(1, num_workers)

    if num_workers == 1:
        for target_id, samples in tqdm(
            target_items,
            total=len(target_items),
            desc='Eval targets',
            unit='target',
            disable=not show_pbar,
        ):
            t_metrics, s_rows, failures = eval_target(target_id, samples, cfg)
            metrics_target_rows.append(t_metrics)
            metrics_sample_rows.extend(s_rows)
            for key, val in failures.items():
                failure_counts[key] += val
    else:
        tasks = [(target_id, samples, cfg) for target_id, samples in target_items]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_eval_target_job, task): task[0] for task in tasks}
            iterator = as_completed(futures)
            if tqdm is not None and show_pbar:
                iterator = tqdm(iterator, total=len(futures), desc='Eval targets', unit='target')
            for future in iterator:
                target_id = futures[future]
                try:
                    _, t_metrics, s_rows, failures = future.result()
                except Exception:  # pylint: disable=broad-except
                    failure_counts['eval_target_failed'] += 1
                    continue
                metrics_target_rows.append(t_metrics)
                metrics_sample_rows.extend(s_rows)
                for key, val in failures.items():
                    failure_counts[key] += val
    for key, val in ref_failures.items():
        failure_counts[key] += val

    # Aggregation per target
    aggregation_modes = cfg.get('aggregation', ['mean_over_valid'])
    summary = {
        'meta': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'manifest': args.manifest,
            'generated_dir': args.generated_dir,
        },
        'metrics': {},
        'directions': {},
    }

    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.getcwd()).decode().strip()
        summary['meta']['git_commit'] = commit
    except Exception:  # pylint: disable=broad-except
        pass

    metrics_by_name = defaultdict(list)
    for row in metrics_target_rows:
        for key, val in row.items():
            if key in ('target_id', 'valid_count', 'sample_count'):
                continue
            metrics_by_name[key].append(val)

    total_samples = sum(row.get('sample_count', 0) or 0 for row in metrics_target_rows)
    total_success = sum(row.get('valid_count', 0) or 0 for row in metrics_target_rows)
    overall_success_rate = None
    if total_samples > 0:
        overall_success_rate = total_success / total_samples

    for metric_name, values in metrics_by_name.items():
        higher_better = _metric_direction(metric_name)
        agg = aggregate.summarize(values)
        summary['metrics'][metric_name] = {
            'mean': agg.get('mean'),
            'med': agg.get('med'),
            'best': aggregate.best_of_k(values, higher_better),
        }
        summary['directions'][metric_name] = 'higher' if higher_better else 'lower'
    if overall_success_rate is not None:
        summary['metrics']['overall_success_rate'] = {
            'mean': overall_success_rate,
            'med': overall_success_rate,
            'best': overall_success_rate,
        }
        summary['directions']['overall_success_rate'] = 'higher'

    # Write outputs
    report.write_csv(
        os.path.join(args.outdir, 'metrics_per_sample.csv'),
        metrics_sample_rows,
        sorted(metrics_sample_rows[0].keys()) if metrics_sample_rows else []
    )
    report.write_csv(
        os.path.join(args.outdir, 'metrics_per_target.csv'),
        metrics_target_rows,
        sorted(metrics_target_rows[0].keys()) if metrics_target_rows else []
    )
    report.write_json(os.path.join(args.outdir, 'summary.json'), summary)
    report.write_json(os.path.join(args.outdir, 'failures.json'), dict(failure_counts))
    report.write_json(os.path.join(args.outdir, 'config_snapshot.json'), cfg)

    report.log_failures(failure_counts)


if __name__ == '__main__':
    main()
