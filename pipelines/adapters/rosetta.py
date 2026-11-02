#!/usr/bin/env python3
"""
Run CPSea Rosetta dG on a flat PDB directory and output a CSV.

Example:
  python run_cpsea_dg.py --pdb-dir /path/to/pdbs --out rosetta_dg.csv --cores 32
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET


def collect_pdbs(pdb_dir: Path) -> List[Path]:
    pdbs = sorted(pdb_dir.rglob("*.pdb"))
    pdbs_gz = sorted(pdb_dir.rglob("*.pdb.gz"))
    return pdbs + pdbs_gz


def normalize_id(decoy: str, input_ids=None) -> str:
    name = os.path.basename(decoy.strip())
    if name.endswith(".pdb.gz"):
        name = name[:-7]
    elif name.endswith(".pdb"):
        name = name[:-4]
    else:
        name = Path(name).stem
    if input_ids is None or name in input_ids:
        return name
    # Rosetta appends an output-decoy index. Strip it only when this resolves
    # to an actual input, so identifiers that naturally end in digits survive.
    original = re.sub(r"_\d{4}$", "", name)
    if original in input_ids:
        return original
    raise ValueError(f"Rosetta decoy does not match an input: {name}")


def read_score_rows(path, input_ids):
    """Retain interface dg and the independent total-score stability metric."""
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        sid = normalize_id(entry['decoy'], input_ids)
        if sid in seen:
            raise ValueError(f'Duplicate Rosetta result: {sid}')
        seen.add(sid)
        row = {'id': sid}
        for source, target in [('dg', 'rosetta_dG'), ('total_score', 'rosetta_total_score')]:
            value = entry.get(source)
            # Missing/nonfinite results stay missing for collector strict checks.
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            row[target] = value if value is not None and math.isfinite(value) else None
        rows.append(row)
    return rows


def protocol_metadata(xml, pdbs, rosetta_bin):
    root = ET.parse(xml).getroot()
    output = root.find('OUTPUT')
    name = output.get('scorefxn') if output is not None else None
    scores = {s.get('name'): dict(s.attrib) for s in root.findall('./SCOREFXNS/ScoreFunction')}
    if not name or name not in scores:
        raise ValueError('Rosetta XML requires OUTPUT with a named ScoreFunction.')
    def digest(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()
    return dict(format='apcyc.rosetta_protocol.v1', xml_sha256=digest(xml),
                output_score_function=scores[name], scope='full input complex', relaxation='none in scoring adapter',
                candidate_ids=sorted(normalize_id(str(p)) for p in pdbs),
                structures_sha256={normalize_id(str(p)): digest(p) for p in pdbs},
                binary_sha256=digest(rosetta_bin))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Rosetta dG from a PDB directory.")
    parser.add_argument("--pdb-dir", required=True, help="Directory containing PDB files.")
    parser.add_argument("--out", default="rosetta_dg.csv", help="Output CSV filename.")
    parser.add_argument("--cores", type=int, default=32, help="CPU cores to use.")
    parser.add_argument(
        "--rosetta-bin",
        default="",
        help="Path to rosetta_scripts.static.linuxgccrelease (or set ROSETTA_BIN).",
    )
    parser.add_argument(
        "--rosetta-xml",
        default="",
        help="RosettaScripts XML; defaults to configs/evaluation/rosetta.xml.",
    )
    parser.add_argument(
        "--eval-dir",
        default="",
        help="Path to CPSea Evaluation dir (contains rosetta_analysis.py).",
    )
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    if not pdb_dir.is_dir():
        print(f"ERROR: pdb-dir not found: {pdb_dir}", file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent
    eval_dir = Path(args.eval_dir).expanduser().resolve() if args.eval_dir else None
    if eval_dir is None:
        default_eval = script_dir / "CPSea" / "Model_Generation_and_Evaluation" / "Evaluation"
        if default_eval.exists():
            eval_dir = default_eval

    if eval_dir is None or not eval_dir.exists():
        print("ERROR: Evaluation dir not found. Use --eval-dir.", file=sys.stderr)
        return 2

    rosetta_script = eval_dir / "rosetta_analysis.py"
    rosetta_xml = (
        Path(args.rosetta_xml).expanduser().resolve()
        if args.rosetta_xml
        else script_dir.parents[1] / "configs" / "evaluation" / "rosetta.xml"
    )

    if not rosetta_script.exists():
        print(f"ERROR: Missing {rosetta_script}", file=sys.stderr)
        return 2
    if not rosetta_xml.exists():
        print(f"ERROR: Missing {rosetta_xml}", file=sys.stderr)
        return 2

    rosetta_bin = args.rosetta_bin or os.environ.get("ROSETTA_BIN", "")
    if not rosetta_bin:
        print("ERROR: rosetta binary not set. Use --rosetta-bin or ROSETTA_BIN.", file=sys.stderr)
        return 2

    pdbs = collect_pdbs(pdb_dir)
    if not pdbs:
        print(f"ERROR: No PDB files found in {pdb_dir}", file=sys.stderr)
        return 2
    input_ids = {normalize_id(str(p)) for p in pdbs}
    if len(input_ids) != len(pdbs):
        raise ValueError('Duplicate PDB identifiers in Rosetta input directory')

    provenance = protocol_metadata(rosetta_xml, pdbs, rosetta_bin)

    out_path = Path(args.out).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="cpsea_dg_") as tmp:
        tmp_dir = Path(tmp)
        list_path = tmp_dir / "pdb_list.txt"
        rosetta_sc = tmp_dir / "rosetta.sc"

        with list_path.open("w", encoding="utf-8") as handle:
            for p in pdbs:
                handle.write(f"{p}\n")

        env = os.environ.copy()
        env["ROSETTA_BIN"] = rosetta_bin
        env["ROSETTA_XML"] = str(rosetta_xml)

        subprocess.run(
            [sys.executable, str(rosetta_script), "-i", str(list_path), "-o", str(rosetta_sc), "-c", str(args.cores)],
            check=True,
            env=env,
        )

        rows = read_score_rows(rosetta_sc, input_ids)
        if {row['id'] for row in rows} != input_ids or any(
                row['rosetta_dG'] is None or row['rosetta_total_score'] is None for row in rows):
            raise ValueError('Rosetta must return finite interface and total scores for every input.')
        provenance['scores'] = {row['id']: {k: row[k] for k in ['rosetta_dG', 'rosetta_total_score']} for row in rows}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=["id", "rosetta_dG", "rosetta_total_score"])
            writer.writeheader()
            writer.writerows(rows)
        out_path.with_suffix('.protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
