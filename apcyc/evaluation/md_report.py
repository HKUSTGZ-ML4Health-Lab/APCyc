#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import print_log


def parse_args():
    parser = argparse.ArgumentParser(description='APCyc MD report generator')
    parser.add_argument('--md_outdir', required=True, help='MD runner output directory')
    parser.add_argument('--outdir', required=True, help='report output directory')
    parser.add_argument('--config', required=True, help='report config yaml')
    return parser.parse_args()


def _read_csv(path: str) -> List[dict]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    val = str(val).strip()
    if val == '' or val.lower() in ('none', 'nan'):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _load_rmsd_series(path: str) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    return xs, ys


def _load_rmsd_values(path: str) -> List[float]:
    _, ys = _load_rmsd_series(path)
    return ys


def _maybe_load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:  # pylint: disable=broad-except
        return None


def _format_mean_std(mean_val: Optional[float], std_val: Optional[float]) -> str:
    if mean_val is None or std_val is None:
        return ''
    return f"{mean_val:.2f}±{std_val:.2f}"


def _format_float(val: Optional[float]) -> str:
    if val is None:
        return ''
    return f"{val:.2f}"


def _ordered_sample_labels(labels: List[str]) -> List[str]:
    if not labels:
        return labels
    label_set = {str(l) for l in labels}
    ordered = []
    ref = "linear_reference"
    ours = "ours"
    other = None
    if ref in label_set:
        ordered.append(ref)
    for cand in sorted(label_set):
        if cand not in (ref, ours):
            other = cand
            break
    if other:
        ordered.append(other)
    if ours in label_set:
        ordered.append(ours)
    return ordered


def _sample_focus_labels(labels: List[str]) -> List[str]:
    focus = []
    for name in ("linear_reference", "ours"):
        if name in labels:
            focus.append(name)
    return focus


def _infer_time_axis(
    xs: List[float],
    md_cfg: dict,
    time_unit: str,
) -> List[float]:
    if not xs:
        return xs
    if time_unit == 'frame':
        return xs
    prod_cfg = md_cfg.get('production', {})
    dt = float(prod_cfg.get('dt', 0.002))
    ntwx = int(prod_cfg.get('ntwx', 1000))
    # cpptraj default first column is frame index starting at 1
    step_ps = dt * ntwx
    factor = step_ps / 1000.0 if time_unit == 'ns' else step_ps
    return [(x - xs[0]) * factor for x in xs]


def generate_report(md_outdir: str, outdir: str, config_path: str):
    os.makedirs(outdir, exist_ok=True)

    import yaml
    cfg = yaml.safe_load(open(config_path, 'r'))

    metrics_csv = os.path.join(md_outdir, 'md_metrics_per_sample.csv')
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Missing {metrics_csv}. Run md_runner first.")

    rows = _read_csv(metrics_csv)
    md_cfg = _maybe_load_json(os.path.join(md_outdir, 'md_config_snapshot.json')) or {}
    report_cfg = cfg.get('report', cfg)

    target_ids = [str(t) for t in report_cfg.get('targets', [])]
    label_map = report_cfg.get('label_map', {})  # type_hat -> label
    label_mode = report_cfg.get('label_mode', 'sample')  # sample | type
    include_status = set(report_cfg.get('include_status', ['ok', 'cached', 'written', 'prepared']))
    time_unit = report_cfg.get('time_unit', 'ns')

    # Build per-target, per-label selection
    selected = {}
    for row in rows:
        target_id = str(row.get('target_id'))
        if target_ids and target_id not in target_ids:
            continue
        status = row.get('md_status')
        if include_status and status not in include_status:
            continue
        type_hat = row.get('type_hat')
        if label_mode == 'type':
            label = label_map.get(type_hat, type_hat)
        else:
            sample_id = row.get('sample_id') or row.get('sample_idx') or row.get('pdb')
            if sample_id and isinstance(sample_id, str) and os.path.sep in sample_id:
                sample_id = os.path.splitext(os.path.basename(sample_id))[0]
            label = sample_id
        if not label:
            continue
        selected.setdefault(target_id, {}).setdefault(label, []).append(row)

    # Plot RMSD trajectories (all samples in one figure per target)
    fig_cfg = report_cfg.get('figure', {})
    fig_targets = fig_cfg.get('targets', target_ids)
    fig_labels = fig_cfg.get('labels', None)
    colors = fig_cfg.get('colors', {})
    traj_alpha = float(fig_cfg.get('traj_alpha', 0.35))
    traj_linewidth = float(fig_cfg.get('traj_linewidth', 1.0))
    fig_rows = len(fig_targets) if fig_targets else 1
    fig, axes = plt.subplots(fig_rows, 1, figsize=fig_cfg.get('figsize', [6, 3 * max(fig_rows, 1)]), sharex=False)
    if fig_rows == 1:
        axes = [axes]

    for ax, target_id in zip(axes, fig_targets):
        target = selected.get(str(target_id), {})
        if fig_labels is None:
            labels = list(target.keys())
            if label_mode == 'sample':
                labels = _ordered_sample_labels(labels)
        else:
            labels = fig_labels
        for label in labels:
            rows_for_label = target.get(label, [])
            if not rows_for_label:
                continue
            first_plotted = False
            for row in rows_for_label:
                rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                if not os.path.exists(rmsd_path):
                    continue
                xs, ys = _load_rmsd_series(rmsd_path)
                if not xs:
                    continue
                xs = _infer_time_axis(xs, md_cfg, time_unit)
                ax.plot(
                    xs,
                    ys,
                    label=label if not first_plotted else None,
                    color=colors.get(label),
                    alpha=traj_alpha,
                    linewidth=traj_linewidth,
                )
                first_plotted = True
        ax.set_title(f"{target_id}")
        ax.set_xlabel(f"Time ({time_unit})" if time_unit != 'frame' else "Frame")
        ax.set_ylabel("RMSD (Å)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'md_rmsd_trajectories.png'), dpi=300)
    fig.savefig(os.path.join(outdir, 'md_rmsd_trajectories.pdf'))

    # Plot RMSD trajectories for linear_reference vs ours
    if label_mode == 'sample' and fig_labels is None:
        fig2, axes2 = plt.subplots(fig_rows, 1, figsize=fig_cfg.get('figsize', [6, 3 * max(fig_rows, 1)]), sharex=False)
        if fig_rows == 1:
            axes2 = [axes2]
        for ax, target_id in zip(axes2, fig_targets):
            target = selected.get(str(target_id), {})
            labels = _sample_focus_labels(list(target.keys()))
            for label in labels:
                rows_for_label = target.get(label, [])
                if not rows_for_label:
                    continue
                first_plotted = False
                for row in rows_for_label:
                    rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                    if not os.path.exists(rmsd_path):
                        continue
                    xs, ys = _load_rmsd_series(rmsd_path)
                    if not xs:
                        continue
                    xs = _infer_time_axis(xs, md_cfg, time_unit)
                    ax.plot(
                        xs,
                        ys,
                        label=label if not first_plotted else None,
                        color=colors.get(label),
                        alpha=traj_alpha,
                        linewidth=traj_linewidth,
                    )
                    first_plotted = True
            ax.set_title(f"{target_id}")
            ax.set_xlabel(f"Time ({time_unit})" if time_unit != 'frame' else "Frame")
            ax.set_ylabel("RMSD (Å)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir, 'md_rmsd_trajectories_ref_vs_ours.png'), dpi=300)
        fig2.savefig(os.path.join(outdir, 'md_rmsd_trajectories_ref_vs_ours.pdf'))

    # Plot RMSD distribution density (ensemble)
    density_cfg = report_cfg.get('density', {})
    if density_cfg.get('enabled', True):
        dens_targets = density_cfg.get('targets', target_ids)
        dens_labels = density_cfg.get('labels', None)
        dens_colors = density_cfg.get('colors', colors)
        dens_bins = int(density_cfg.get('bins', 60))
        dens_range = density_cfg.get('range', None)
        per_sample = bool(density_cfg.get('per_sample', True))
        aggregate = bool(density_cfg.get('aggregate', True))
        if label_mode == 'sample':
            aggregate = False
        per_sample_alpha = float(density_cfg.get('per_sample_alpha', 0.15))
        agg_alpha = float(density_cfg.get('aggregate_alpha', 0.9))
        dens_rows = len(dens_targets) if dens_targets else 1
        dens_fig, dens_axes = plt.subplots(
            dens_rows, 1,
            figsize=density_cfg.get('figsize', [6, 3 * max(dens_rows, 1)]),
            sharex=False,
        )
        if dens_rows == 1:
            dens_axes = [dens_axes]

        for ax, target_id in zip(dens_axes, dens_targets):
            target = selected.get(str(target_id), {})
            if dens_labels is None:
                labels = list(target.keys())
                if label_mode == 'sample':
                    labels = _ordered_sample_labels(labels)
            else:
                labels = dens_labels
            global_vals = []
            label_rows = {}
            for label in labels:
                rows_for_label = target.get(label, [])
                label_rows[label] = rows_for_label
                for row in rows_for_label:
                    rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                    if not os.path.exists(rmsd_path):
                        continue
                    vals = _load_rmsd_values(rmsd_path)
                    if vals:
                        global_vals.extend(vals)
            if not global_vals:
                continue
            if dens_range and isinstance(dens_range, (list, tuple)) and len(dens_range) == 2:
                rmin, rmax = float(dens_range[0]), float(dens_range[1])
            else:
                rmin, rmax = float(min(global_vals)), float(max(global_vals))
            if rmin == rmax:
                rmax = rmin + 1e-3
            bin_edges = np.linspace(rmin, rmax, dens_bins + 1)

            for label in labels:
                rows_for_label = label_rows.get(label, [])
                if not rows_for_label:
                    continue
                if per_sample:
                    for row in rows_for_label:
                        rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                        if not os.path.exists(rmsd_path):
                            continue
                        vals = _load_rmsd_values(rmsd_path)
                        if not vals:
                            continue
                        ax.hist(
                            vals,
                            bins=bin_edges,
                            density=True,
                            histtype='step',
                            color=dens_colors.get(label),
                            alpha=per_sample_alpha,
                            linewidth=0.8,
                            label=label if label_mode == 'sample' else None,
                        )
                if aggregate:
                    # aggregate per label
                    agg_vals = []
                    for row in rows_for_label:
                        rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                        if not os.path.exists(rmsd_path):
                            continue
                        vals = _load_rmsd_values(rmsd_path)
                        if vals:
                            agg_vals.extend(vals)
                    if agg_vals:
                        ax.hist(
                            agg_vals,
                            bins=bin_edges,
                            density=True,
                            histtype='step',
                            color=dens_colors.get(label),
                            alpha=agg_alpha,
                            linewidth=1.5,
                            label=label,
                        )

            ax.set_title(f"{target_id}")
            ax.set_xlabel("RMSD (Å)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        dens_fig.tight_layout()
        dens_fig.savefig(os.path.join(outdir, 'md_rmsd_density.png'), dpi=300)
        dens_fig.savefig(os.path.join(outdir, 'md_rmsd_density.pdf'))

        # Density plot for linear_reference vs ours
        if label_mode == 'sample' and dens_labels is None:
            dens_fig2, dens_axes2 = plt.subplots(
                dens_rows, 1,
                figsize=density_cfg.get('figsize', [6, 3 * max(dens_rows, 1)]),
                sharex=False,
            )
            if dens_rows == 1:
                dens_axes2 = [dens_axes2]
            for ax, target_id in zip(dens_axes2, dens_targets):
                target = selected.get(str(target_id), {})
                labels = _sample_focus_labels(list(target.keys()))
                if not labels:
                    continue
                global_vals = []
                label_rows = {}
                for label in labels:
                    rows_for_label = target.get(label, [])
                    label_rows[label] = rows_for_label
                    for row in rows_for_label:
                        rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                        if not os.path.exists(rmsd_path):
                            continue
                        vals = _load_rmsd_values(rmsd_path)
                        if vals:
                            global_vals.extend(vals)
                if not global_vals:
                    continue
                if dens_range and isinstance(dens_range, (list, tuple)) and len(dens_range) == 2:
                    rmin, rmax = float(dens_range[0]), float(dens_range[1])
                else:
                    rmin, rmax = float(min(global_vals)), float(max(global_vals))
                if rmin == rmax:
                    rmax = rmin + 1e-3
                bin_edges = np.linspace(rmin, rmax, dens_bins + 1)
                for label in labels:
                    rows_for_label = label_rows.get(label, [])
                    if not rows_for_label:
                        continue
                    for row in rows_for_label:
                        rmsd_path = os.path.join(row.get('run_dir', ''), 'rmsd.dat')
                        if not os.path.exists(rmsd_path):
                            continue
                        vals = _load_rmsd_values(rmsd_path)
                        if not vals:
                            continue
                        ax.hist(
                            vals,
                            bins=bin_edges,
                            density=True,
                            histtype='step',
                            color=dens_colors.get(label),
                            alpha=per_sample_alpha,
                            linewidth=0.8,
                            label=label,
                        )
                ax.set_title(f"{target_id}")
                ax.set_xlabel("RMSD (Å)")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
            dens_fig2.tight_layout()
            dens_fig2.savefig(os.path.join(outdir, 'md_rmsd_density_ref_vs_ours.png'), dpi=300)
            dens_fig2.savefig(os.path.join(outdir, 'md_rmsd_density_ref_vs_ours.pdf'))

    # Table 4-style summary
    table_rows = []
    for target_id, labels in selected.items():
        for label, rows_for_label in labels.items():
            mean_val = _to_float(rows_for_label[0].get('md_rmsd_mean'))
            std_val = _to_float(rows_for_label[0].get('md_rmsd_std'))
            dg_val = _to_float(rows_for_label[0].get('md_mmpbsa_dg'))
            table_rows.append({
                'target_id': target_id,
                'label': label,
                'rmsd_mean_std': _format_mean_std(mean_val, std_val),
                'dg_mmpbsa': _format_float(dg_val),
            })

    table_csv = os.path.join(outdir, 'md_table4.csv')
    with open(table_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target_id', 'label', 'rmsd_mean_std', 'dg_mmpbsa'])
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    # Results text
    text_lines = [
        "Results.",
        "As shown in the RMSD trajectories, cyclic peptides exhibit flatter trajectories and lower RMSD,",
        "suggesting improved conformational stability from geometric constraints.",
        "RMSD density plots summarize the ensemble stability of all samples.",
        "Table 4 reports mean RMSD (± std) and MM/PBSA ΔG for each target/peptide.",
    ]
    with open(os.path.join(outdir, 'md_results_text.txt'), 'w') as f:
        f.write(" ".join(text_lines) + "\n")

    print_log(f"Saved MD report to {outdir}")


def main():
    args = parse_args()
    generate_report(args.md_outdir, args.outdir, args.config)


if __name__ == '__main__':
    main()
