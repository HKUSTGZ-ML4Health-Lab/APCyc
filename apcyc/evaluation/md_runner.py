#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import yaml
from tqdm import tqdm

from utils.logger import print_log
from apcyc.evaluation.io import aggregate, pdb_utils, report
from apcyc.evaluation.metrics import md_sim


def parse_args():
    parser = argparse.ArgumentParser(description='APCyc MD runner')
    parser.add_argument('--manifest', help='path to apcyc_predictions.jsonl')
    parser.add_argument('--generated_dir', help='directory with target_id/sample pdb layout')
    parser.add_argument('--flat_targets', action='store_true',
                        help='treat each pdb as its own target (ignore subfolders)')
    parser.add_argument('--target_id', help='override target_id for all generated_dir samples')
    parser.add_argument('--outdir', required=True, help='output directory')
    parser.add_argument('--config', required=True, help='md config yaml')
    parser.add_argument('--max_samples_per_target', type=int,
                        help='override md.max_samples_per_target from config (<=0 means no limit)')
    parser.add_argument('--report_outdir', help='generate MD report to this directory')
    parser.add_argument('--report_config', help='md report config yaml (default: apcyc/evaluation/configs/md_report.yaml)')
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


def build_manifest_from_dir(
    generated_dir: str,
    flat_targets: bool = False,
    target_id_override: str = None,
) -> List[dict]:
    generated_dir = os.path.abspath(generated_dir)
    items = []
    for root, _, files in os.walk(generated_dir):
        for name in files:
            if not name.lower().endswith(('.pdb', '.cif')):
                continue
            pdb_path = os.path.join(root, name)
            sample_id = os.path.splitext(name)[0]
            if target_id_override:
                target_id = str(target_id_override)
            elif flat_targets:
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


def summarize_metrics(rows: List[dict]) -> Dict[str, dict]:
    metrics = defaultdict(list)
    for row in rows:
        for key in ('md_rmsd_mean', 'md_rmsd_std', 'md_mmpbsa_dg'):
            metrics[key].append(row.get(key))
    summary = {}
    for key, vals in metrics.items():
        summary[key] = aggregate.summarize(vals)
    return summary


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    cfg = yaml.safe_load(open(args.config, 'r'))
    md_cfg = cfg.get('md', cfg)
    md_cfg.setdefault('work_dir', os.path.join(args.outdir, 'md_runs'))
    if args.max_samples_per_target is not None:
        md_cfg['max_samples_per_target'] = int(args.max_samples_per_target)

    pep_chain = cfg.get('peptide_chain', md_cfg.get('peptide_chain', 'L'))
    rec_chain = cfg.get('receptor_chain', md_cfg.get('receptor_chain', 'R'))

    if not args.manifest and not args.generated_dir:
        raise ValueError('Either --manifest or --generated_dir must be provided.')
    if args.generated_dir:
        manifest = build_manifest_from_dir(
            args.generated_dir,
            flat_targets=args.flat_targets,
            target_id_override=args.target_id,
        )
    else:
        manifest = load_manifest(args.manifest)

    targets = defaultdict(list)
    for item in manifest:
        target_id = item.get('receptor_id') or item.get('target_id') or item.get('sample_id')
        targets[str(target_id)].append(item)

    metrics_rows = []
    failures = defaultdict(int)

    show_pbar = md_cfg.get('pbar', True)
    for target_id, samples in tqdm(
        targets.items(),
        total=len(targets),
        desc='MD targets',
        unit='target',
        disable=not show_pbar,
    ):
        md_state = {'count': 0}
        for sample in samples:
            pdb_path = sample.get('pdb')
            if pdb_path and not os.path.isabs(pdb_path):
                pdb_path = os.path.abspath(pdb_path)
            if not pdb_path or not os.path.exists(pdb_path):
                failures['missing_pdb'] += 1
                continue

            structure = pdb_utils.load_structure(pdb_path)
            chain = pdb_utils.get_chain(structure, pep_chain)
            if chain is None:
                failures['missing_chain'] += 1
                continue
            rec_chain_obj = pdb_utils.get_chain(structure, rec_chain)
            if rec_chain_obj is None:
                failures['missing_receptor'] += 1
                continue

            if not md_sim.should_run_md(target_id, sample, md_cfg, md_state):
                continue

            run_dir = md_sim.build_run_dir(md_cfg.get('work_dir'), target_id, sample.get('sample_id'))
            md_metrics, md_status = md_sim.run_md(pdb_path, pep_chain, rec_chain, md_cfg, run_dir)
            md_state['count'] += 1

            metrics_rows.append({
                'target_id': target_id,
                'sample_id': sample.get('sample_id'),
                'sample_idx': sample.get('sample_idx'),
                'pdb': pdb_path,
                'type_hat': sample.get('type_hat'),
                'run_dir': run_dir,
                'md_status': md_status,
                'md_rmsd_mean': md_metrics.get('md_rmsd_mean'),
                'md_rmsd_std': md_metrics.get('md_rmsd_std'),
                'md_mmpbsa_dg': md_metrics.get('md_mmpbsa_dg'),
            })

    summary = {
        'meta': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'manifest': args.manifest,
            'generated_dir': args.generated_dir,
            'outdir': args.outdir,
        },
        'summary': summarize_metrics(metrics_rows),
        'failures': dict(failures),
    }

    report.write_csv(os.path.join(args.outdir, 'md_metrics_per_sample.csv'), metrics_rows, fieldnames=[
        'target_id', 'sample_id', 'sample_idx', 'type_hat', 'pdb', 'run_dir',
        'md_status', 'md_rmsd_mean', 'md_rmsd_std', 'md_mmpbsa_dg',
    ])
    report.write_json(os.path.join(args.outdir, 'md_summary.json'), summary)
    report.write_json(os.path.join(args.outdir, 'md_config_snapshot.json'), md_cfg)
    report.log_failures(failures)

    if args.report_outdir:
        from apcyc.evaluation import md_report as md_report_mod
        report_cfg = args.report_config or os.path.join(
            os.path.dirname(__file__),
            'configs',
            'md_report.yaml',
        )
        md_report_mod.generate_report(args.outdir, args.report_outdir, report_cfg)


if __name__ == '__main__':
    main()
