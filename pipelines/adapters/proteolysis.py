#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


DEFAULT_PROTEASES: list[str] = [
    "S01.001",
    "A01.001",
    "C01.060",
    "M10.003",
    "M10.008",
]


def _parse_proteases(value: str | None) -> list[str]:
    if value is None:
        return DEFAULT_PROTEASES
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _ensure_models_exist(prosperous_dir: Path, proteases: list[str]) -> None:
    missing: list[str] = []
    for p in proteases:
        d = prosperous_dir / "finalModels" / p
        if not d.is_dir():
            missing.append(p)
    if missing:
        raise SystemExit(
            "ERROR: Missing built-in model folders under 'finalModels/' for: "
            + ", ".join(missing)
        )


def _render_progress(current: int, total: int, label: str) -> str:
    width = 28
    total = max(total, 1)
    current = min(max(current, 0), total)
    filled = int(round(width * (current / total)))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} {label}"


def _iter_fasta_records(path: Path) -> tuple[str, str]:
    seq_id = None
    seq_chunks: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_chunks)
                seq_id = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if seq_id is not None:
        yield seq_id, "".join(seq_chunks)


def _count_fasta_sequences(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return max(count, 0)


def _clean_sequence(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY-]", "-", seq.upper())


def _generate_windows(seq: str, window: int, stride: int, max_windows: int | None) -> list[tuple[int, str]]:
    if len(seq) < window:
        return [(window // 2, seq + "-" * (window - len(seq)))]
    if len(seq) == window:
        return [(window // 2, seq)]

    last_start = len(seq) - window
    windows: list[tuple[int, str]] = []
    for start in range(0, last_start + 1, stride):
        windows.append((start + window // 2, seq[start : start + window]))
    if windows and windows[-1][0] != last_start + window // 2:
        windows.append((last_start + window // 2, seq[last_start : last_start + window]))

    if max_windows is not None and max_windows > 0 and len(windows) > max_windows:
        if max_windows == 1:
            return [windows[len(windows) // 2]]
        step = (len(windows) - 1) / (max_windows - 1)
        idxs = {int(round(i * step)) for i in range(max_windows)}
        windows = [windows[i] for i in sorted(idxs)]

    return windows


def _write_windows_fasta(path: Path, seq_id: str, windows: list[tuple[int, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pos, window in windows:
            f.write(f">{seq_id}|{pos}\n{window}\n")


def _load_existing_scores(path: Path) -> dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "proteolysis_sumscore"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"ERROR: Existing output has unexpected header in {path}. "
                f"Need columns: {sorted(required)}; got: {reader.fieldnames}"
            )
        data: dict[str, float] = {}
        for row in reader:
            seq_id = (row.get("id") or "").strip()
            if not seq_id:
                continue
            try:
                score = float(row.get("proteolysis_sumscore", ""))
            except ValueError:
                continue
            data[seq_id] = score
        return data


def _sum_scores_single(results_csv: Path) -> float:
    sum_by_prot: dict[str, float] = {}
    cnt_by_prot: dict[str, int] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"protease", "pro"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"ERROR: Unexpected results header in {results_csv}. "
                f"Need columns: {sorted(required)}; got: {reader.fieldnames}"
            )
        for row in reader:
            prot = (row.get("protease") or "").strip()
            if not prot:
                continue
            try:
                p = float(row.get("pro", ""))
            except ValueError:
                continue

            sum_by_prot[prot] = sum_by_prot.get(prot, 0.0) + float(p)
            cnt_by_prot[prot] = cnt_by_prot.get(prot, 0) + 1

    total_score = 0.0
    for prot, s in sum_by_prot.items():
        n = cnt_by_prot.get(prot, 0)
        if n <= 0:
            continue
        total_score += s / n
    return total_score


def _run_prosperous_prediction(
    prosperous_dir: Path,
    fasta: Path,
    run_dir: Path,
    proteases: list[str],
    process_num: int,
) -> Path:
    prosperous_py = prosperous_dir / "Prosperousplus.py"
    if not prosperous_py.exists():
        raise SystemExit(f"ERROR: Entry script not found: {prosperous_py}")

    # Preflight: ProsperousPlus requires pycaret even for prediction mode.
    probe = subprocess.run([sys.executable, "-c", "import pycaret"], cwd=str(prosperous_dir))
    if probe.returncode != 0:
        raise SystemExit(
            "ERROR: Missing dependency 'pycaret' in the current Python environment.\n"
            "Fix: activate the ProsperousPlus conda env (python=3.7) and install requirements, e.g.\n"
            "  conda create -n prosperousplus python=3.7 -y\n"
            "  conda activate prosperousplus\n"
            "  pip install -r requirements.txt\n"
            "Then rerun this script with that environment's python."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(prosperous_py),
        "--predictfile",
        str(fasta),
        "--outputpath",
        str(run_dir),
        "--inputType",
        "fasta",
        "--protease",
        ",".join(proteases),
        "--mode",
        "prediction",
        "--PLOT",
        "No",
        "--processNum",
        str(process_num),
    ]
    proc = subprocess.run(cmd, cwd=str(prosperous_dir))
    if proc.returncode != 0:
        raise SystemExit(f"ERROR: ProsperousPlus prediction failed (exit {proc.returncode}).")

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"ERROR: ProsperousPlus did not produce {results_csv}")
    return results_csv


def _process_sequence(
    seq_id: str,
    seq: str,
    prosperous_dir: Path,
    run_dir: Path,
    proteases: list[str],
    stride: int,
    max_windows: int | None,
    process_num: int,
) -> tuple[str, float]:
    cleaned = _clean_sequence(seq)
    windows = _generate_windows(cleaned, window=8, stride=stride, max_windows=max_windows)
    if not windows:
        return seq_id, 0.0

    with tempfile.TemporaryDirectory(dir=run_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_fasta = tmpdir_path / "windows.fasta"
        _write_windows_fasta(tmp_fasta, seq_id, windows)
        seq_out = tmpdir_path / "out"
        seq_out.mkdir()

        results_csv = _run_prosperous_prediction(
            prosperous_dir, tmp_fasta, seq_out, proteases, process_num
        )
        score = _sum_scores_single(results_csv)

    return seq_id, score


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a single 'proteolysis sum score' per peptide from a FASTA file using "
            "ProsperousPlus built-in models: sum cleavage probabilities across positions for each protease, "
            "then sum across selected proteases."
        )
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Input FASTA file path (e.g., sequence/seqs.aa)",
    )
    parser.add_argument(
        "--out",
        default=Path("cycpep_proteolysis_sumscore.csv"),
        type=Path,
        help="Output CSV path (default: cycpep_proteolysis_sumscore.csv)",
    )
    parser.add_argument(
        "--prosperous-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="ProsperousPlus folder containing Prosperousplus.py (default: this script's folder)",
    )
    parser.add_argument(
        "--proteases",
        default=None,
        type=str,
        help="Comma-separated protease IDs (default: built-in 20-protease list hardcoded in this script)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        type=Path,
        help="Working directory for ProsperousPlus outputs (default: <out parent>/prosperousplus_run_<timestamp>)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride for windowing when sequence length > 8 (default: 2)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Max number of windows per sequence (default: no limit)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of sequences to process in parallel (default: min(8, CPU cores))",
    )
    parser.add_argument(
        "--process-num",
        type=int,
        default=2,
        help="ProsperousPlus --processNum value per job (default: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV (skip already-computed IDs and append new rows).",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Only process the first FASTA record (for quick smoke tests).",
    )
    args = parser.parse_args()

    prosperous_dir = args.prosperous_dir.resolve()
    fasta = args.fasta.expanduser().resolve()
    out_csv = args.out.expanduser().resolve()
    proteases = _parse_proteases(args.proteases)

    if not fasta.exists():
        print(f"ERROR: FASTA not found: {fasta}", file=sys.stderr)
        return 2
    if not fasta.is_file():
        print(f"ERROR: Not a file: {fasta}", file=sys.stderr)
        return 2

    _ensure_models_exist(prosperous_dir, proteases)

    if args.run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out_csv.parent / f"prosperousplus_run_{ts}"
    else:
        run_dir = args.run_dir.expanduser().resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_scores: dict[str, float] = {}
    existing_count = 0
    if args.resume:
        try:
            existing_scores = _load_existing_scores(out_csv)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
        existing_count = len(existing_scores)
        if existing_scores:
            print(f"[resume] Found {existing_count} completed IDs in {out_csv}", flush=True)

    total_seqs = _count_fasta_sequences(fasta)
    if args.first_only:
        total_seqs = min(total_seqs, 1)
    if total_seqs == 0:
        print("ERROR: No sequences found in FASTA.", file=sys.stderr)
        return 2

    print("Selected proteases ({}): {}".format(len(proteases), ", ".join(proteases)))
    print(
        "Running ProsperousPlus prediction with per-sequence input (jobs={}, processNum={})...".format(
            args.jobs, args.process_num
        ),
        flush=True,
    )

    processed = 0
    written = 0
    in_flight: set = set()
    max_in_flight = max(args.jobs * 2, 1)

    if args.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        stream_file = out_csv.open("a", encoding="utf-8", newline="")
        stream_writer = csv.writer(stream_file)
    else:
        stream_file = out_csv.open("w", encoding="utf-8", newline="")
        stream_writer = csv.writer(stream_file)
        stream_writer.writerow(["id", "proteolysis_sumscore"])
        stream_file.flush()

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for idx, (seq_id, seq) in enumerate(_iter_fasta_records(fasta)):
            if args.first_only and idx > 0:
                break
            if args.resume and seq_id in existing_scores:
                processed += 1
                sys.stdout.write(
                    "\r" + _render_progress(processed, total_seqs, "Step 1/2: Running prediction")
                )
                sys.stdout.flush()
                continue
            in_flight.add(
                executor.submit(
                    _process_sequence,
                    seq_id,
                    seq,
                    prosperous_dir,
                    run_dir,
                    proteases,
                    args.stride,
                    args.max_windows,
                    args.process_num,
                )
            )
            if len(in_flight) >= max_in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    sid, score = fut.result()
                    stream_writer.writerow([sid, f"{score:.6f}"])
                    stream_file.flush()
                    written += 1
                    processed += 1
                    sys.stdout.write(
                        "\r" + _render_progress(processed, total_seqs, "Step 1/2: Running prediction")
                    )
                    sys.stdout.flush()

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                sid, score = fut.result()
                stream_writer.writerow([sid, f"{score:.6f}"])
                stream_file.flush()
                written += 1
                processed += 1
                sys.stdout.write(
                    "\r" + _render_progress(processed, total_seqs, "Step 1/2: Running prediction")
                )
                sys.stdout.flush()

    sys.stdout.write("\r" + _render_progress(total_seqs, total_seqs, "Step 1/2: Running prediction") + "\n")
    sys.stdout.flush()

    stream_file.close()
    total_rows = existing_count + written
    if args.resume:
        print(f"OK: wrote {written} new rows to {out_csv} (total {total_rows})")
    else:
        print(f"OK: wrote {written} rows to {out_csv}")
    print(f"ProsperousPlus raw results kept in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
