#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import re
import shutil
import shlex
import subprocess
import tempfile
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np
import yaml
try:
    from tqdm import tqdm
except Exception:  # pylint: disable=broad-except
    tqdm = None

from apcyc.evaluation.io import aggregate, pdb_utils, report
from apcyc.evaluation.metrics import rosetta_score, vina_score


def _iter_pdbs(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith((".pdb", ".cif")):
                yield os.path.join(root, name)


def _safe_copy_or_move(src: str, dst: str, move: bool):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        idx = 1
        while os.path.exists(dst):
            dst = f"{base}_{idx}{ext}"
            idx += 1
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def _resolve_output_path(src: str, input_dir: str, out_dir: str, preserve_structure: bool):
    if preserve_structure:
        rel = os.path.relpath(src, input_dir)
        return os.path.join(out_dir, rel)
    return os.path.join(out_dir, os.path.basename(src))


def _resolve_target_id(pdb_path: str, input_dir: str) -> str:
    try:
        rel = os.path.relpath(pdb_path, input_dir)
        parts = rel.split(os.sep)
        if len(parts) > 1:
            return parts[0]
    except Exception:  # pylint: disable=broad-except
        pass
    return os.path.basename(os.path.dirname(pdb_path)) or os.path.splitext(os.path.basename(pdb_path))[0]


def _normalize_chain_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    value = str(value).strip()
    if not value:
        return []
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def _parse_vina_score(text: str):
    if not text:
        return None
    patterns = [
        r"Affinity:\s*([-+]?\d*\.?\d+)",
        r"Estimated Free Energy of Binding\s*:\s*([-+]?\d*\.?\d+)",
        r"REMARK VINA RESULT:\s*([-+]?\d*\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _get_tool_path(env_key: str, default_path: str) -> str:
    value = os.environ.get(env_key, "").strip()
    return value if value else default_path


def _heavy_atom_coords(pdb_path: str):
    try:
        structure = pdb_utils.load_structure(pdb_path)
    except Exception:  # pylint: disable=broad-except
        return None
    coords = []
    for atom in structure.get_atoms():
        element = getattr(atom, "element", "") or ""
        element = element.strip().upper()
        if not element:
            name = atom.get_name().strip()
            element = re.sub(r"[^A-Za-z]", "", name)[:1].upper()
        if element in {"H", "D"}:
            continue
        coords.append(atom.get_coord())
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def _compute_box_from_pdb(pdb_path: str, padding: float, min_size: float):
    coords = _heavy_atom_coords(pdb_path)
    if coords is None:
        return None
    arr = coords
    min_c = arr.min(axis=0)
    max_c = arr.max(axis=0)
    center = arr.mean(axis=0)
    size = (max_c - min_c) + (2.0 * float(padding))
    size = np.maximum(size, float(min_size))
    return center, size


def _energy_metric_direction(metric_name: str) -> bool:
    # higher_better? For energies and vina, lower is better.
    if metric_name.startswith(("rosetta_interface_dg", "vina_score")) or metric_name.startswith("rosetta_total_score"):
        return False
    return True


def _filter_values_for_aggregation(metric_name: str, values):
    # Only keep negative values for total_score and vina aggregation.
    if metric_name.startswith("vina_score") or metric_name.startswith("rosetta_total_score"):
        return [v for v in values if v is not None and v < 0]
    return [v for v in values if v is not None]


def _run_vina_score(
    pdb_path: str,
    ligand_chain: str,
    receptor_chain: str,
    vina_cfg: dict,
    debug_log: Optional[str] = None,
):
    mgl_pythonsh = vina_cfg.get("mgl_pythonsh") or _get_tool_path(
        "MGLTOOLS_PYTHONSH",
        "/data_hdd/home/yangziyi/Tools/mgltools_x86_64Linux2_1.5.6/bin/pythonsh",
    )
    prepare_ligand = vina_cfg.get("prepare_ligand") or _get_tool_path(
        "MGLTOOLS_PREPARE_LIGAND",
        "/data_hdd/home/yangziyi/Tools/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
    )
    prepare_receptor = vina_cfg.get("prepare_receptor") or _get_tool_path(
        "MGLTOOLS_PREPARE_RECEPTOR",
        "/data_hdd/home/yangziyi/Tools/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
    )
    vina_bin = vina_cfg.get("vina_bin") or _get_tool_path(
        "VINA_BIN",
        "/data_hdd/home/yangziyi/Tools/autodock_vina_1_1_2_linux_x86/bin/vina",
    )
    conda_env = vina_cfg.get("conda_env")
    conda_bin = vina_cfg.get("conda_bin") or "conda"
    keep_temp = bool(vina_cfg.get("keep_temp", False))
    tmp_root = vina_cfg.get("tmp_dir")
    auto_box = vina_cfg.get("auto_box", True)
    mode = (vina_cfg.get("mode") or "score_only").strip().lower()
    default_padding = 8.0 if mode == "score_only" else 10.0
    default_min_size = 22.0 if mode == "score_only" else 26.0
    padding_by_mode = vina_cfg.get("box_padding_by_mode") or {}
    min_size_by_mode = vina_cfg.get("box_min_size_by_mode") or {}
    box_padding = padding_by_mode.get(mode, vina_cfg.get("box_padding", default_padding))
    box_min_size = min_size_by_mode.get(mode, vina_cfg.get("box_min_size", default_min_size))
    box_padding = float(box_padding)
    box_min_size = float(box_min_size)
    debug_enabled = bool(debug_log)

    def _debug(message: str):
        if not debug_enabled:
            return
        log_dir = os.path.dirname(debug_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(debug_log, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    ligand_ids = _normalize_chain_ids(ligand_chain)
    receptor_ids = _normalize_chain_ids(receptor_chain)
    if len(ligand_ids) != 1:
        _debug(f"[vina] invalid_ligand_chain={ligand_chain}")
        return None, "invalid_ligand_chain"
    if len(receptor_ids) != 1:
        _debug(f"[vina] invalid_receptor_chain={receptor_chain}")
        return None, "invalid_receptor_chain"

    tmpdir = None
    tmp_path = None
    if keep_temp:
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
            tmp_path = tempfile.mkdtemp(dir=tmp_root)
        else:
            tmp_path = tempfile.mkdtemp()
    else:
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
            tmpdir = tempfile.TemporaryDirectory(dir=tmp_root)
        else:
            tmpdir = tempfile.TemporaryDirectory()
        tmp_path = tmpdir.name
    _debug(f"[vina] pdb={pdb_path}")
    _debug(f"[vina] ligand_chain={ligand_ids[0]} receptor_chain={receptor_ids[0]}")
    _debug(f"[vina] tmp_path={tmp_path} keep_temp={keep_temp}")
    try:
        split = pdb_utils.split_pdb_by_chain(pdb_path, tmp_path)
        ligand_pdb = split.get(ligand_ids[0])
        receptor_pdb = split.get(receptor_ids[0])
        if ligand_pdb is None:
            _debug("[vina] missing_ligand_chain after split")
            return None, "missing_ligand_chain"
        if receptor_pdb is None:
            _debug("[vina] missing_receptor_chain after split")
            return None, "missing_receptor_chain"

        ligand_basename = os.path.basename(ligand_pdb)
        receptor_basename = os.path.basename(receptor_pdb)
        _debug(f"[vina] ligand_pdb={ligand_pdb} exists={os.path.exists(ligand_pdb)}")
        _debug(f"[vina] receptor_pdb={receptor_pdb} exists={os.path.exists(receptor_pdb)}")
        ligand_pdbqt = os.path.join(tmp_path, "ligand.pdbqt")
        receptor_pdbqt = os.path.join(tmp_path, "receptor.pdbqt")

        ligand_cmd = [
            mgl_pythonsh,
            prepare_ligand,
            "-l",
            ligand_basename,
            "-o",
            "ligand.pdbqt",
        ]
        receptor_cmd = [
            mgl_pythonsh,
            prepare_receptor,
            "-r",
            receptor_basename,
            "-o",
            "receptor.pdbqt",
        ]
        try:
            if debug_enabled:
                _debug(f"[vina] ligand_cmd={shlex.join(ligand_cmd)} cwd={tmp_path}")
                result = subprocess.run(
                    ligand_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path
                )
                if result.stdout:
                    _debug(f"[vina] ligand_stdout={result.stdout.strip()}")
                if result.stderr:
                    _debug(f"[vina] ligand_stderr={result.stderr.strip()}")
                _debug(f"[vina] receptor_cmd={shlex.join(receptor_cmd)} cwd={tmp_path}")
                result = subprocess.run(
                    receptor_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path
                )
                if result.stdout:
                    _debug(f"[vina] receptor_stdout={result.stdout.strip()}")
                if result.stderr:
                    _debug(f"[vina] receptor_stderr={result.stderr.strip()}")
            else:
                subprocess.run(ligand_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path)
                subprocess.run(receptor_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_path)
        except Exception as exc:  # pylint: disable=broad-except
            _debug(f"[vina] vina_prepare_failed err={exc}")
            return None, "vina_prepare_failed"

        vina_cmd = [
            vina_bin,
            "--receptor",
            receptor_pdbqt,
            "--ligand",
            ligand_pdbqt,
        ]
        if mode == "local_only":
            vina_cmd.append("--local_only")
        elif mode == "minimize":
            vina_cmd.append("--minimize")
        else:
            vina_cmd.append("--score_only")
        if auto_box:
            box = _compute_box_from_pdb(ligand_pdb, box_padding, box_min_size)
            if box is None:
                _debug("[vina] missing_vina_box")
                return None, "missing_vina_box"
            center, size = box
            _debug(f"[vina] vina_mode={mode} box_padding={box_padding} box_min_size={box_min_size}")
            _debug(f"[vina] box_center={center.tolist()} box_size={size.tolist()}")
            vina_cmd += [
                "--center_x",
                f"{center[0]:.3f}",
                "--center_y",
                f"{center[1]:.3f}",
                "--center_z",
                f"{center[2]:.3f}",
                "--size_x",
                f"{size[0]:.3f}",
                "--size_y",
                f"{size[1]:.3f}",
                "--size_z",
                f"{size[2]:.3f}",
            ]
        else:
            required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
            if any(vina_cfg.get(key) is None for key in required):
                _debug("[vina] missing_vina_box")
                return None, "missing_vina_box"
            vina_cmd += [
                "--center_x",
                str(vina_cfg.get("center_x")),
                "--center_y",
                str(vina_cfg.get("center_y")),
                "--center_z",
                str(vina_cfg.get("center_z")),
                "--size_x",
                str(vina_cfg.get("size_x")),
                "--size_y",
                str(vina_cfg.get("size_y")),
                "--size_z",
                str(vina_cfg.get("size_z")),
            ]
        if vina_cfg.get("cpu") is not None:
            vina_cmd += ["--cpu", str(vina_cfg.get("cpu"))]
        if vina_cfg.get("seed") is not None:
            vina_cmd += ["--seed", str(vina_cfg.get("seed"))]
        if conda_env:
            vina_cmd = [conda_bin, "run", "-n", conda_env] + vina_cmd

        _debug(f"[vina] vina_cmd={shlex.join(vina_cmd)}")
        result = subprocess.run(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            _debug(f"[vina] vina_returncode={result.returncode}")
            _debug(f"[vina] vina_stdout={result.stdout.strip() if result.stdout else ''}")
            _debug(f"[vina] vina_stderr={result.stderr.strip() if result.stderr else ''}")
            return None, "vina_failed"

        text = result.stdout or ""
        if not text and result.stderr:
            text = result.stderr
        _debug(f"[vina] vina_stdout={result.stdout.strip() if result.stdout else ''}")
        _debug(f"[vina] vina_stderr={result.stderr.strip() if result.stderr else ''}")
        score = _parse_vina_score(text)
        if score is None:
            _debug("[vina] missing_vina_score")
            return None, "missing_vina_score"
        return score, "ok"
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
            _debug(f"[vina] tmp_path_removed={tmp_path}")


def main():
    parser = argparse.ArgumentParser(description="Energy-only postprocess: compute dG/Vina and filter final_success.")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDBs to score")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="Evaluation config yaml (for chains/rosetta/vina)")
    parser.add_argument("--peptide_chain", default=None, help="Override peptide chain ID from config")
    parser.add_argument("--receptor_chain", default=None, help="Override receptor chain ID from config")
    parser.add_argument("--final_dir", default=None, help="Output directory for energy-success structures")
    parser.add_argument("--preserve_structure", action="store_true", default=True,
                        help="Preserve input subdirectory structure when copying")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--vina_debug", action="store_true", help="Enable verbose vina debug log")
    parser.add_argument("--vina_debug_log", default=None, help="Path for vina debug log file")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    peptide_chain = args.peptide_chain or cfg.get("peptide_chain", "L")
    rec_chain = args.receptor_chain or cfg.get("receptor_chain", "R")
    rosetta_cfg = cfg.get("rosetta", {})
    calc_cpsea = rosetta_cfg.get("calc_cpsea_dg", True)
    calc_pepglad = rosetta_cfg.get("calc_pepglad_dg", True)
    calc_total_score = rosetta_cfg.get("also_total_score", False)
    vina_cfg = cfg.get("vina", {}) or {"enabled": False}
    vina_auto = bool(vina_cfg.get("auto", False))
    vina_modes_cfg = vina_cfg.get("modes")
    if isinstance(vina_modes_cfg, str):
        vina_modes = [m.strip() for m in vina_modes_cfg.split(",") if m.strip()]
    elif isinstance(vina_modes_cfg, (list, tuple)):
        vina_modes = [str(m).strip() for m in vina_modes_cfg if str(m).strip()]
    else:
        vina_modes = []
    if not vina_modes:
        vina_modes = [(vina_cfg.get("mode") or "score_only").strip()]
    vina_modes = [m.lower() for m in vina_modes]
    primary_vina_mode = "score_only" if "score_only" in vina_modes else vina_modes[0]

    out_dir = os.path.abspath(args.out_dir)
    final_dir = os.path.abspath(args.final_dir or os.path.join(out_dir, "final_success"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    vina_debug_log = None
    if args.vina_debug or vina_cfg.get("debug"):
        vina_debug_log = args.vina_debug_log or vina_cfg.get("debug_log") or os.path.join(out_dir, "vina_debug.log")
        debug_dir = os.path.dirname(vina_debug_log)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    rows = []
    target_rows = []
    rosetta_vals = []
    rosetta_pepglad_vals = []
    rosetta_total_vals = []
    vina_vals = []
    vina_vals_by_mode = {mode: [] for mode in vina_modes}
    rosetta_status_counts = defaultdict(int)
    rosetta_pepglad_status_counts = defaultdict(int)
    rosetta_total_status_counts = defaultdict(int)
    vina_status_counts = defaultdict(int)
    vina_status_counts_by_mode = {mode: defaultdict(int) for mode in vina_modes}
    failures = defaultdict(int)
    total = 0
    success = 0
    targets = defaultdict(list)
    aggregation_modes = cfg.get("aggregation") or ["mean_over_valid", "best_of_k"]
    if isinstance(aggregation_modes, str):
        aggregation_modes = [m.strip() for m in aggregation_modes.split(",") if m.strip()]

    pdb_paths = list(_iter_pdbs(args.input_dir))
    show_pbar = cfg.get("pbar", True)
    iterator = pdb_paths
    if tqdm is not None and show_pbar:
        iterator = tqdm(pdb_paths, total=len(pdb_paths), desc="Energy samples", unit="pdb")

    for pdb_path in iterator:
        total += 1
        target_id = _resolve_target_id(pdb_path, args.input_dir)
        structure = None
        try:
            structure = pdb_utils.load_structure(pdb_path)
        except Exception:  # pylint: disable=broad-except
            failures["load_structure_failed"] += 1
        pep_chain_obj = pdb_utils.get_chain(structure, peptide_chain) if structure else None
        rec_chain_obj = pdb_utils.get_chain(structure, rec_chain) if structure else None
        if pep_chain_obj is None:
            failures["missing_peptide_chain"] += 1

        vina_scores_by_mode = {}
        vina_status_by_mode = {}
        if vina_cfg.get("enabled", False):
            if vina_auto:
                for mode in vina_modes:
                    cfg_mode = deepcopy(vina_cfg)
                    cfg_mode["mode"] = mode
                    score, status = _run_vina_score(
                        pdb_path,
                        ligand_chain=peptide_chain,
                        receptor_chain=rec_chain,
                        vina_cfg=cfg_mode,
                        debug_log=vina_debug_log,
                    )
                    vina_scores_by_mode[mode] = score
                    vina_status_by_mode[mode] = status
            else:
                score, status = vina_score.vina_score(pdb_path, vina_cfg)
                vina_scores_by_mode[primary_vina_mode] = score
                vina_status_by_mode[primary_vina_mode] = status
                for mode in vina_modes:
                    if mode == primary_vina_mode:
                        continue
                    vina_scores_by_mode[mode] = None
                    vina_status_by_mode[mode] = "skipped"
        else:
            for mode in vina_modes:
                vina_scores_by_mode[mode] = None
                vina_status_by_mode[mode] = "disabled"
        for mode in vina_modes:
            vina_status_counts_by_mode[mode][vina_status_by_mode[mode]] += 1
        vina_score_val = vina_scores_by_mode.get(primary_vina_mode)
        vina_status = vina_status_by_mode.get(primary_vina_mode, "disabled")
        vina_status_counts[vina_status] += 1

        rosetta_score_val = None
        rosetta_status = "skipped"
        rosetta_pepglad_val = None
        rosetta_pepglad_status = "skipped"
        rosetta_total_score_val = None
        rosetta_total_status = "skipped"
        if rosetta_cfg.get("enabled", True) and (calc_cpsea or calc_pepglad):
            if pep_chain_obj is None:
                rosetta_status = "missing_peptide_chain"
                rosetta_pepglad_status = "missing_peptide_chain"
            elif rosetta_cfg.get("require_receptor", False) and rec_chain_obj is None:
                rosetta_status = "missing_receptor"
                rosetta_pepglad_status = "missing_receptor"
            else:
                if calc_pepglad:
                    pepglad_cfg = deepcopy(rosetta_cfg)
                    pepglad_cfg["energy_mode"] = "interface_dg"
                    pepglad_cfg["engine"] = "pyrosetta"
                    rosetta_pepglad_val, rosetta_pepglad_status = rosetta_score.rosetta_total_score(
                        pdb_path,
                        peptide_chain,
                        pepglad_cfg,
                        rec_chain=rec_chain,
                    )
                if calc_cpsea:
                    cpsea_cfg = deepcopy(rosetta_cfg)
                    cpsea_cfg["energy_mode"] = "interface_dg"
                    cpsea_cfg["engine"] = "rosetta_scripts"
                    rosetta_score_val, rosetta_status = rosetta_score.rosetta_total_score(
                        pdb_path,
                        peptide_chain,
                        cpsea_cfg,
                        rec_chain=rec_chain,
                    )
        if rosetta_cfg.get("enabled", True) and calc_total_score:
            if pep_chain_obj is None:
                rosetta_total_status = "missing_peptide_chain"
            elif rosetta_cfg.get("require_receptor", False) and rec_chain_obj is None:
                rosetta_total_status = "missing_receptor"
            else:
                total_cfg = deepcopy(rosetta_cfg)
                total_cfg["energy_mode"] = "total_score"
                total_cfg["engine"] = "pyrosetta"
                rosetta_total_score_val, rosetta_total_status = rosetta_score.rosetta_total_score(
                    pdb_path,
                    peptide_chain,
                    total_cfg,
                    rec_chain=rec_chain,
                )
        rosetta_status_counts[rosetta_status] += 1
        rosetta_pepglad_status_counts[rosetta_pepglad_status] += 1
        rosetta_total_status_counts[rosetta_total_status] += 1

        energy_success_source = None
        if rosetta_score_val is not None:
            energy_success = rosetta_score_val < 0
            energy_success_source = "cpsea"
        else:
            energy_success = False
        if energy_success:
            success += 1
            if rosetta_score_val is not None:
                rosetta_vals.append(rosetta_score_val)
            if rosetta_pepglad_val is not None:
                rosetta_pepglad_vals.append(rosetta_pepglad_val)
            if rosetta_total_score_val is not None and rosetta_total_score_val < 0:
                rosetta_total_vals.append(rosetta_total_score_val)
            if vina_score_val is not None and vina_score_val < 0:
                vina_vals.append(vina_score_val)
            for mode in vina_modes:
                mode_val = vina_scores_by_mode.get(mode)
                if mode_val is not None and mode_val < 0:
                    vina_vals_by_mode[mode].append(mode_val)
            dst = _resolve_output_path(pdb_path, args.input_dir, final_dir, args.preserve_structure)
            _safe_copy_or_move(pdb_path, dst, args.move)

        row = {
            "target_id": target_id,
            "pdb": pdb_path,
            "rosetta_interface_dg": rosetta_score_val,
            "rosetta_status": rosetta_status,
            "rosetta_interface_dg_pepglad": rosetta_pepglad_val,
            "rosetta_status_pepglad": rosetta_pepglad_status,
            "rosetta_total_score": rosetta_total_score_val,
            "rosetta_total_status": rosetta_total_status,
            "vina_score": vina_score_val,
            "vina_status": vina_status,
            "energy_success": energy_success,
            "energy_success_source": energy_success_source,
        }
        for mode in vina_modes:
            row[f"vina_score_{mode}"] = vina_scores_by_mode.get(mode)
            row[f"vina_status_{mode}"] = vina_status_by_mode.get(mode)
        rows.append(row)
        targets[target_id].append(row)

    energy_metric_keys = [
        "rosetta_interface_dg",
        "rosetta_interface_dg_pepglad",
        "rosetta_total_score",
        "vina_score",
    ]
    for mode in vina_modes:
        energy_metric_keys.append(f"vina_score_{mode}")

    # Per-target aggregation on successful samples (cpsea dG success)
    for target_id, t_rows in targets.items():
        success_rows = [r for r in t_rows if r.get("energy_success")]
        target_metrics = {
            "target_id": target_id,
            "sample_count": len(t_rows),
            "energy_success_count": len(success_rows),
            "energy_success_rate": (len(success_rows) / len(t_rows)) if t_rows else None,
        }
        for key in energy_metric_keys:
            values = _filter_values_for_aggregation(key, [r.get(key) for r in success_rows])
            higher_better = _energy_metric_direction(key)
            for mode in aggregation_modes:
                if mode == "best_of_k":
                    agg_val = aggregate.best_of_k(values, higher_better)
                else:
                    agg_val = aggregate.mean_over_valid(values)
                target_metrics[f"{mode}.{key}"] = agg_val
        target_rows.append(target_metrics)

    cross_target = {}
    if target_rows:
        for key in energy_metric_keys:
            higher_better = _energy_metric_direction(key)
            for mode in aggregation_modes:
                values = _filter_values_for_aggregation(key, [row.get(f"{mode}.{key}") for row in target_rows])
                if mode == "best_of_k":
                    agg_val = aggregate.best_of_k(values, higher_better)
                else:
                    agg_val = aggregate.mean_over_valid(values)
                cross_target[f"{mode}.{key}"] = agg_val

    summary = {
        "total": total,
        "energy_success": success,
        "energy_success_rate": (success / total) if total > 0 else None,
        "rosetta_interface_dg": aggregate.summarize(rosetta_vals),
        "rosetta_interface_dg_pepglad": aggregate.summarize(rosetta_pepglad_vals),
        "rosetta_total_score": aggregate.summarize(rosetta_total_vals),
        "vina_score": aggregate.summarize(vina_vals),
        "rosetta_status_counts": dict(rosetta_status_counts),
        "rosetta_pepglad_status_counts": dict(rosetta_pepglad_status_counts),
        "rosetta_total_status_counts": dict(rosetta_total_status_counts),
        "vina_status_counts": dict(vina_status_counts),
        "failures": dict(failures),
        "final_success_dir": final_dir,
        "target_aggregation_modes": aggregation_modes,
        "target_aggregation": cross_target,
    }
    if vina_modes:
        for mode in vina_modes:
            summary[f"vina_score_{mode}"] = aggregate.summarize(vina_vals_by_mode.get(mode, []))
            summary[f"vina_status_counts_{mode}"] = dict(vina_status_counts_by_mode.get(mode, {}))

    report.write_csv(os.path.join(out_dir, "energy_per_sample.csv"), rows, [])
    report.write_csv(os.path.join(out_dir, "energy_per_target.csv"), target_rows, [])
    report.write_json(os.path.join(out_dir, "energy_summary.json"), summary)


if __name__ == "__main__":
    main()
