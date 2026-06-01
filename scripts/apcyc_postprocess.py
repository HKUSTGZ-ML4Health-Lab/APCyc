#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
import json
from typing import Dict

try:
    from tqdm import tqdm
except Exception:  # pylint: disable=broad-except
    tqdm = None

_RELAX_CFG = {}


def _set_relax_cfg(cfg):
    global _RELAX_CFG
    _RELAX_CFG = cfg


def _determine_cyclization_type(distance):
    if 3 <= distance < 4.5:
        return "DISULFIDE"
    if 4.5 <= distance < 6:
        return "HEADTAIL"
    if 6 <= distance <= 8:
        return "ISOPEPTIDE"
    return "UNSUPPORTED"


def _relax_process_row(row):
    if not _RELAX_CFG:
        raise RuntimeError("Relax config is not initialized.")
    cpsea_root = _RELAX_CFG["cpsea_root"]
    out_dir = _RELAX_CFG["out_dir"]
    preserve_structure = _RELAX_CFG.get("preserve_structure", False)
    structure_root = _RELAX_CFG.get("structure_root")
    ligand_chain = _RELAX_CFG.get("ligand_chain", "L")

    relax_dir = os.path.join(cpsea_root, "PostProcess", "Relax")
    if relax_dir not in sys.path:
        sys.path.insert(0, relax_dir)
    from cys_to_cys_model import ForceFieldMinimizerCys  # noqa: E402
    from head_tail_model import ForceFieldMinimizerHeadTail  # noqa: E402
    from k_to_de_model import ForceFieldMinimizerKtoDE  # noqa: E402

    pdb_path, dist_str, length = row
    dist = float(dist_str)
    cyc_type = _determine_cyclization_type(dist)
    if cyc_type == "UNSUPPORTED":
        return (pdb_path, dist_str, length, cyc_type, None)
    length = int(length)
    out_name = os.path.basename(pdb_path)[:-4] + "_relaxed.pdb"
    if preserve_structure and structure_root:
        rel = os.path.relpath(pdb_path, structure_root)
        subdir = os.path.dirname(rel)
        out_path = os.path.join(out_dir, subdir, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = os.path.join(out_dir, out_name)
    opts = [((ligand_chain, 0), (ligand_chain, length - 1))]
    if cyc_type == "DISULFIDE":
        ff = ForceFieldMinimizerCys(ligand_chain=ligand_chain)
    elif cyc_type == "ISOPEPTIDE":
        ff = ForceFieldMinimizerKtoDE(ligand_chain=ligand_chain)
    else:
        ff = ForceFieldMinimizerHeadTail(ligand_chain=ligand_chain)
    energy = ff(pdb_path, out_path, cyclic_chains=[ligand_chain], cyclic_opts=opts)
    return (pdb_path, dist_str, length, cyc_type, energy)


def _run_cmd(cmd):
    print(f"[postprocess] run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_predictions(pred_path: str) -> Dict[str, dict]:
    pred_map = {}
    if not pred_path:
        return pred_map
    with open(pred_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pdb_path = obj.get("pdb")
            if pdb_path:
                pred_map[os.path.basename(pdb_path)] = obj
    return pred_map


def _residue_by_index(chain, idx: int):
    residues = [res for res in chain.get_residues()]
    if idx < 0 or idx >= len(residues):
        return None
    return residues[idx]


def _cb_coord(residue):
    if residue.get_resname() == "GLY":
        coords = [
            residue["N"].get_coord(),
            residue["CA"].get_coord(),
            residue["C"].get_coord(),
            residue["O"].get_coord(),
        ]
        import numpy as np
        b = coords[1] - coords[0]
        c = coords[2] - coords[1]
        a = np.cross(b, c)
        return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + coords[1]
    return residue["CB"].get_coord()


def _pair_cb_filter(
    predictions,
    source_dir,
    fail_site_dir,
    fail_distance_dir,
    min_sep,
    min_cb,
    max_cb,
    preserve_structure=False,
    ligand_chain="L",
):
    from Bio.PDB import PDBParser
    import numpy as np

    os.makedirs(fail_site_dir, exist_ok=True)
    os.makedirs(fail_distance_dir, exist_ok=True)
    parser = PDBParser(QUIET=True)

    total = 0
    fail_site = 0
    fail_distance = 0
    missing_pair = 0
    failed_rel_paths = set()
    failed_basenames = set()

    pdb_paths = []
    for root, _, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(".pdb"):
                pdb_paths.append(os.path.join(root, name))
    iterator = pdb_paths
    if tqdm is not None:
        iterator = tqdm(pdb_paths, total=len(pdb_paths), desc="Pair-CB filter", unit="pdb")

    for src in iterator:
        name = os.path.basename(src)
        total += 1
        pred = predictions.get(name)
        if pred is None:
            missing_pair += 1
            fail_site += 1
            dst_dir = fail_site_dir
        else:
            pair = pred.get("pair_hat")
            if not pair or len(pair) != 2:
                missing_pair += 1
                fail_site += 1
                dst_dir = fail_site_dir
            else:
                i, j = int(pair[0]), int(pair[1])
                if abs(i - j) < min_sep:
                    fail_site += 1
                    dst_dir = fail_site_dir
                else:
                    try:
                        struct = parser.get_structure("pdb", src)
                        chain = None
                        for ch in struct.get_chains():
                            if ch.get_id() == ligand_chain:
                                chain = ch
                                break
                        if chain is None:
                            missing_pair += 1
                            fail_site += 1
                            dst_dir = fail_site_dir
                        else:
                            res_i = _residue_by_index(chain, i)
                            res_j = _residue_by_index(chain, j)
                            if res_i is None or res_j is None:
                                missing_pair += 1
                                fail_site += 1
                                dst_dir = fail_site_dir
                            else:
                                dist = float(np.linalg.norm(_cb_coord(res_i) - _cb_coord(res_j)))
                                if not (min_cb <= dist <= max_cb):
                                    fail_distance += 1
                                    dst_dir = fail_distance_dir
                                else:
                                    dst_dir = None
                    except Exception:
                        missing_pair += 1
                        fail_site += 1
                        dst_dir = fail_site_dir

        if dst_dir:
            if preserve_structure:
                rel = os.path.relpath(src, source_dir)
                dst = os.path.join(dst_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            else:
                dst = os.path.join(dst_dir, name)
            base, ext = os.path.splitext(dst)
            idx = 1
            while os.path.exists(dst):
                dst = f"{base}_{idx}{ext}"
                idx += 1
            shutil.copy2(src, dst)
            rel = os.path.relpath(src, source_dir)
            failed_rel_paths.add(rel)
            failed_basenames.add(os.path.basename(src))

    denom = max(total - fail_site, 1)
    success = (total - fail_site - fail_distance) / denom
    print(f"[postprocess] pair-cb filter: total={total} fail_site={fail_site} fail_distance={fail_distance} missing_pair={missing_pair} success={success:.4f}")
    return {
        "total": total,
        "fail_site": fail_site,
        "fail_distance": fail_distance,
        "missing_pair": missing_pair,
        "success": success,
        "pair_site_min_sep": min_sep,
        "cb_min": min_cb,
        "cb_max": max_cb,
        "failed_rel_paths": sorted(failed_rel_paths),
        "failed_basenames": sorted(failed_basenames),
    }


def _filter_cb_csv(csv_path, source_dir, target_dir, min_cb, max_cb, preserve_structure=False):
    os.makedirs(target_dir, exist_ok=True)
    files_to_move = set()
    total = 0
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0] == "Filename":
                continue
            try:
                dist = float(row[1])
            except ValueError:
                continue
            total += 1
            if not (min_cb <= dist <= max_cb):
                files_to_move.add(row[0])

    copied = 0
    failed_rel_paths = set()
    failed_basenames = set()
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for name in files:
            file_paths.append(os.path.join(root, name))
    iterator = file_paths
    if tqdm is not None:
        iterator = tqdm(file_paths, total=len(file_paths), desc="CB filter move", unit="file")
    for src in iterator:
        name = os.path.basename(src)
        if name in files_to_move:
            if preserve_structure:
                rel = os.path.relpath(src, source_dir)
                dst = os.path.join(target_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            else:
                dst = os.path.join(target_dir, name)
            base, ext = os.path.splitext(dst)
            idx = 1
            while os.path.exists(dst):
                dst = f"{base}_{idx}{ext}"
                idx += 1
            shutil.copy2(src, dst)
            copied += 1
            rel = os.path.relpath(src, source_dir)
            failed_rel_paths.add(rel)
            failed_basenames.add(os.path.basename(src))
    success = (total - len(files_to_move)) / max(total, 1)
    print(f"[postprocess] CB filter copied {copied} files to {target_dir}")
    return {
        "total": total,
        "fail_distance": len(files_to_move),
        "success": success,
        "cb_min": min_cb,
        "cb_max": max_cb,
        "failed_rel_paths": sorted(failed_rel_paths),
        "failed_basenames": sorted(failed_basenames),
    }


def _copy_passed_files(source_dir, passed_dir, failed_rel_paths, failed_basenames, preserve_structure=False):
    os.makedirs(passed_dir, exist_ok=True)
    failed_rel_paths = set(failed_rel_paths or [])
    failed_basenames = set(failed_basenames or [])
    for root, _, files in os.walk(source_dir):
        for name in files:
            if not name.lower().endswith(".pdb"):
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, source_dir)
            if preserve_structure:
                if rel in failed_rel_paths:
                    continue
                dst = os.path.join(passed_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            else:
                if name in failed_basenames:
                    continue
                dst = os.path.join(passed_dir, name)
            base, ext = os.path.splitext(dst)
            idx = 1
            while os.path.exists(dst):
                dst = f"{base}_{idx}{ext}"
                idx += 1
            shutil.copy2(src, dst)


def _count_pdbs(root_dir: str) -> int:
    total = 0
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".pdb"):
                total += 1
    return total


def _relax_from_cb_length(
    cpsea_root,
    cb_length_csv,
    out_dir,
    num_workers,
    structure_root=None,
    preserve_structure=False,
    ligand_chain="L",
):
    relax_dir = os.path.join(cpsea_root, "PostProcess", "Relax")
    if not os.path.isdir(relax_dir):
        raise RuntimeError(f"Relax dir not found: {relax_dir}")

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(cb_length_csv, "r") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if not row or row[0] == "Filename":
                continue
            rows.append(row)

    import multiprocessing as mp
    from tqdm import tqdm

    cfg = {
        "cpsea_root": cpsea_root,
        "out_dir": out_dir,
        "preserve_structure": preserve_structure,
        "structure_root": structure_root,
        "ligand_chain": ligand_chain,
    }
    _set_relax_cfg(cfg)

    result_csv = os.path.join(out_dir, "energy_data.csv")
    with open(result_csv, "w") as fout:
        fout.write("pdb\tout\tcyc_type\tlength\tenergy\n")
        if num_workers <= 1:
            for pdb_path, dist, length, cyc_type, energy in tqdm(
                map(_relax_process_row, rows), total=len(rows)
            ):
                out_name = os.path.basename(pdb_path)[:-4] + "_relaxed.pdb"
                if energy is None:
                    energy_str = "none"
                else:
                    energy_str = ",".join([f"{v:.4f}" for v in energy])
                fout.write(f"{os.path.basename(pdb_path)}\t{out_name}\t{cyc_type}\t{length}\t{energy_str}\n")
                fout.flush()
        else:
            with mp.Pool(processes=num_workers, initializer=_set_relax_cfg, initargs=(cfg,)) as pool:
                for pdb_path, dist, length, cyc_type, energy in tqdm(
                    pool.imap_unordered(_relax_process_row, rows), total=len(rows)
                ):
                    out_name = os.path.basename(pdb_path)[:-4] + "_relaxed.pdb"
                    if energy is None:
                        energy_str = "none"
                    else:
                        energy_str = ",".join([f"{v:.4f}" for v in energy])
                    fout.write(f"{os.path.basename(pdb_path)}\t{out_name}\t{cyc_type}\t{length}\t{energy_str}\n")
                    fout.flush()
    print(f"[postprocess] Relaxed structures saved to {out_dir}")


def _group_outputs_by_receptor(root_dir, mode="prefix"):
    for current_root, _, files in os.walk(root_dir):
        for name in files:
            if not name.lower().endswith(".pdb"):
                continue
            if mode == "pdb4":
                receptor_id = name[:4]
            elif mode == "sample":
                receptor_id = os.path.splitext(name)[0]
            else:
                receptor_id = name.split("_", 1)[0]
            src = os.path.join(current_root, name)
            dst_dir = os.path.join(root_dir, receptor_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, name)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            shutil.move(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Integrated CPSea-style postprocess pipeline")
    parser.add_argument("--generated", required=True, help="Generated PDB directory")
    parser.add_argument("--out_root", required=True, help="Output root directory")
    parser.add_argument("--cpsea_root", default=None, help="Path to CPSea/Model_Generation_and_Evaluation")
    parser.add_argument("--rename_mode", default="none", choices=["none", "glad", "flow", "diff"])
    parser.add_argument("--ligand_chain", default="L", help="Ligand/peptide chain ID (default: L)")
    parser.add_argument("--receptors", default=None, help="Full receptor PDB directory")
    parser.add_argument("--epitopes_json", default=None, help="Epitope JSON file")
    parser.add_argument("--pocket_json_dir", default=None, help="Pocket JSON dir (for glad_cut_pocket)")
    parser.add_argument("--do_cut_pocket", action="store_true", help="Run glad_cut_pocket step")
    parser.add_argument("--do_cb_filter", action="store_true", help="Run CB distance filter")
    parser.add_argument("--use_pair_cb_filter", action="store_true",
                        help="Use pair_hat from predictions.jsonl for CB filter")
    parser.add_argument("--predictions", default=None, help="apcyc_predictions.jsonl for pair-based CB filter")
    parser.add_argument("--pair_site_min_sep", type=int, default=3, help="Min residue index separation for pair_hat")
    parser.add_argument("--cb_min", type=float, default=3.0)
    parser.add_argument("--cb_max", type=float, default=8.0)
    parser.add_argument("--cb_filter_summary", default=None, help="Write CB filter summary JSON")
    parser.add_argument("--do_combine_epitope", action="store_true")
    parser.add_argument("--do_relax", action="store_true")
    parser.add_argument("--relax_workers", type=int, default=8)
    parser.add_argument("--do_combine_receptor", action="store_true")
    parser.add_argument("--process_cb_failed", action="store_true",
                        help="Also run downstream steps on CB-filter failed structures")
    parser.add_argument("--only_cb_failed_postprocess", action="store_true",
                        help="Only run downstream steps on CB-filter failed structures")
    parser.add_argument("--process_all_after_filter", action="store_true",
                        help="Process all samples after CB filter (including failed)")
    parser.add_argument("--preserve_structure", action="store_true", default=True)
    parser.add_argument("--receptor_id_mode", default="prefix", choices=["prefix", "sample", "pdb4"])
    args = parser.parse_args()

    apcyc_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(apcyc_root)
    cpsea_root = args.cpsea_root or os.path.join(repo_root, "CPSea", "Model_Generation_and_Evaluation")
    post_dir = os.path.join(cpsea_root, "PostProcess")
    if not os.path.isdir(post_dir):
        raise RuntimeError(f"CPSea PostProcess not found: {post_dir}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    step_dir = os.path.abspath(args.generated)

    if args.rename_mode != "none":
        rename_script = os.path.join(post_dir, f"rename_{args.rename_mode}.py")
        rename_out = out_root / f"{args.rename_mode}_renamed"
        _run_cmd([sys.executable, rename_script, "-i", step_dir, "-o", str(rename_out)])
        step_dir = str(rename_out)

    if args.do_cut_pocket:
        if not args.receptors or not args.pocket_json_dir:
            raise RuntimeError("--do_cut_pocket requires --receptors and --pocket_json_dir")
        cut_out = out_root / "cut_pocket"
        cut_script = os.path.join(post_dir, "glad_cut_pocket.py")
        _run_cmd([
            sys.executable,
            cut_script,
            "--renamed",
            step_dir,
            "--receptor",
            args.receptors,
            "--json",
            args.pocket_json_dir,
            "--output",
            str(cut_out),
        ])
        step_dir = str(cut_out)

    failed_sets = []
    if args.only_cb_failed_postprocess and not args.do_cb_filter:
        raise RuntimeError("--only_cb_failed_postprocess requires --do_cb_filter")
    if args.only_cb_failed_postprocess:
        args.process_cb_failed = True

    if args.do_cb_filter:
        cb_csv = out_root / "cb_distance.csv"
        summary = None
        failed_rel_paths = []
        failed_basenames = []
        if args.use_pair_cb_filter:
            total_all = _count_pdbs(step_dir)
            pred_map = _load_predictions(args.predictions)
            cb_failed_site = out_root / "cb_failed_site"
            cb_failed_distance = out_root / "cb_failed_distance"
            summary = _pair_cb_filter(
                pred_map,
                step_dir,
                str(cb_failed_site),
                str(cb_failed_distance),
                args.pair_site_min_sep,
                args.cb_min,
                args.cb_max,
                args.preserve_structure,
                ligand_chain=args.ligand_chain,
            )
            if summary is not None:
                failed_rel_paths = summary.get("failed_rel_paths", [])
                failed_basenames = summary.get("failed_basenames", [])
                summary["mode"] = "pair_hat"
                denom = max(total_all, 1)
                total_valid = summary.get("total", 0)
                missing = max(total_all - total_valid, 0)
                cb_fail = summary.get("fail_site", 0) + summary.get("fail_distance", 0) + missing
                summary["cb_fail"] = cb_fail
                summary["total"] = total_all
                summary["success"] = (
                    total_all
                    - cb_fail
                ) / denom
            failed_sets = [
                ("cb_failed_site", str(cb_failed_site)),
                ("cb_failed_distance", str(cb_failed_distance)),
            ]
        else:
            total_all = _count_pdbs(step_dir)
            cb_script = os.path.join(post_dir, "cb_distance_calculater.py")
            _run_cmd([sys.executable, cb_script, step_dir, str(cb_csv), "--chain", args.ligand_chain])
            cb_failed = out_root / "cb_failed"
            summary = _filter_cb_csv(
                str(cb_csv),
                step_dir,
                str(cb_failed),
                args.cb_min,
                args.cb_max,
                preserve_structure=args.preserve_structure,
            )
            if summary is not None:
                failed_rel_paths = summary.get("failed_rel_paths", [])
                failed_basenames = summary.get("failed_basenames", [])
                summary["mode"] = "distance_csv"
                denom = max(total_all, 1)
                total_valid = summary.get("total", 0)
                missing = max(total_all - total_valid, 0)
                cb_fail = summary.get("fail_distance", 0) + missing
                summary["cb_fail"] = cb_fail
                summary["total"] = total_all
                summary["success"] = (total_all - cb_fail) / denom
            failed_sets = [("cb_failed", str(cb_failed))]
        if summary is not None:
            summary_path = Path(args.cb_filter_summary) if args.cb_filter_summary else (out_root / "cb_filter_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as fout:
                json.dump(summary, fout, indent=2)
            print(f"[postprocess] CB filter summary written to {summary_path}")
        if not args.process_all_after_filter and not args.only_cb_failed_postprocess and summary is not None:
            passed_dir = out_root / "cb_passed"
            _copy_passed_files(step_dir, str(passed_dir), failed_rel_paths, failed_basenames, args.preserve_structure)
            step_dir = str(passed_dir)

    if args.do_combine_epitope and not args.only_cb_failed_postprocess:
        if not args.receptors or not args.epitopes_json:
            raise RuntimeError("--do_combine_epitope requires --receptors and --epitopes_json")
        recon_out = out_root / "reconstructed"
        combine_script = os.path.join(post_dir, "combine_epitope.py")
        _run_cmd([
            sys.executable,
            combine_script,
            "--generated",
            step_dir,
            "--receptors",
            args.receptors,
            "--epitopes",
            args.epitopes_json,
            "--output",
            str(recon_out),
            "--ligand_chain",
            args.ligand_chain,
        ])
        step_dir = str(recon_out)
    if args.do_combine_epitope and args.process_cb_failed:
        if not args.receptors or not args.epitopes_json:
            raise RuntimeError("--do_combine_epitope requires --receptors and --epitopes_json")
        combine_script = os.path.join(post_dir, "combine_epitope.py")
        updated_failed = []
        for label, failed_dir in failed_sets:
            if not os.path.isdir(failed_dir):
                continue
            failed_out = Path(failed_dir)
            _run_cmd([
                sys.executable,
                combine_script,
                "--generated",
                failed_dir,
                "--receptors",
                args.receptors,
                "--epitopes",
                args.epitopes_json,
                "--output",
                str(failed_out),
                "--ligand_chain",
                args.ligand_chain,
            ])
            updated_failed.append((label, str(failed_out)))
        failed_sets = updated_failed

    if args.do_relax and not args.only_cb_failed_postprocess:
        cb_len_csv = out_root / "cb_length.csv"
        cb_len_script = os.path.join(post_dir, "cb_distance_and_length.py")
        _run_cmd([
            sys.executable,
            cb_len_script,
            os.path.abspath(step_dir),
            str(cb_len_csv),
            "--chain",
            args.ligand_chain,
        ])
        relax_out = out_root / "relaxed"
        _relax_from_cb_length(
            cpsea_root,
            str(cb_len_csv),
            str(relax_out),
            args.relax_workers,
            structure_root=os.path.abspath(step_dir),
            preserve_structure=args.preserve_structure,
            ligand_chain=args.ligand_chain,
        )
        step_dir = str(relax_out)
    if args.do_relax and args.process_cb_failed:
        cb_len_script = os.path.join(post_dir, "cb_distance_and_length.py")
        updated_failed = []
        for label, failed_dir in failed_sets:
            if not os.path.isdir(failed_dir):
                continue
            failed_cb_len = out_root / f"{label}_cb_length.csv"
            _run_cmd([
                sys.executable,
                cb_len_script,
                os.path.abspath(failed_dir),
                str(failed_cb_len),
                "--chain",
                args.ligand_chain,
            ])
            failed_relax = Path(failed_dir)
            _relax_from_cb_length(
                cpsea_root,
                str(failed_cb_len),
                str(failed_relax),
                args.relax_workers,
                structure_root=os.path.abspath(failed_dir),
                preserve_structure=args.preserve_structure,
                ligand_chain=args.ligand_chain,
            )
            updated_failed.append((label, str(failed_relax)))
        failed_sets = updated_failed

    if args.do_combine_receptor and not args.only_cb_failed_postprocess:
        if not args.receptors:
            raise RuntimeError("--do_combine_receptor requires --receptors")
        full_out = out_root / "full_complex"
        combine_script = os.path.join(post_dir, "combine_receptor.py")
        _run_cmd([
            sys.executable,
            combine_script,
            "--relaxed",
            step_dir,
            "--receptors",
            args.receptors,
            "--output",
            str(full_out),
            "--ligand_chain",
            args.ligand_chain,
        ])
        if args.preserve_structure:
            _group_outputs_by_receptor(str(full_out), mode=args.receptor_id_mode)
        step_dir = str(full_out)
    if args.do_combine_receptor and args.process_cb_failed:
        if not args.receptors:
            raise RuntimeError("--do_combine_receptor requires --receptors")
        combine_script = os.path.join(post_dir, "combine_receptor.py")
        for label, failed_dir in failed_sets:
            if not os.path.isdir(failed_dir):
                continue
            failed_full = Path(failed_dir)
            _run_cmd([
                sys.executable,
                combine_script,
                "--relaxed",
                failed_dir,
                "--receptors",
                args.receptors,
                "--output",
                str(failed_full),
                "--ligand_chain",
                args.ligand_chain,
            ])
            if args.preserve_structure:
                _group_outputs_by_receptor(str(failed_full), mode=args.receptor_id_mode)

    print(f"[postprocess] Done. Final output: {step_dir}")


if __name__ == "__main__":
    main()
