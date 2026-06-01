#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Build batch CSV for PharmPapp preprocessing.

This helper reads an index file describing protein–peptide complexes and emits a
CSV with the columns expected by ``pharm_papp_prepare.py``:

```
pdb,ligand_chain,sample_id[,receptor_chain]
```

Typical usage with the APCyc dataset structure:

```bash
python -m scripts.permeability.build_pharm_csv \
  --index APCyc/datasets/train_valid/all.txt \
  --pdb-root APCyc/datasets/train_valid/pdbs \
  --output APCyc/datasets/train_valid/hydrogenated_list.csv
```

Each row of ``all.txt`` is expected to contain at least three tokens:
``sample_id\treceptor_chain\tligand_chain[\textra_fields...]``.  The script
locates ``<sample_id>.pdb`` under ``--pdb-root`` and writes a CSV suitable for
``pharm_papp_prepare.py --batch-csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate PharmPapp batch CSV from dataset index.')
    parser.add_argument('--index', required=True, type=Path, help='Path to the dataset index file (e.g. all.txt).')
    parser.add_argument('--pdb-root', required=True, type=Path, help='Directory containing the complex PDB files.')
    parser.add_argument('--output', required=True, type=Path, help='Destination CSV path.')
    parser.add_argument('--strict', action='store_true', help='Fail if any PDB file is missing. Otherwise the row is skipped with a warning.')
    return parser.parse_args()


def iter_index_rows(index_path: Path) -> Iterable[Tuple[str, str, str]]:
    with index_path.open('r') as fin:
        for lineno, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f'Line {lineno} in {index_path} has fewer than 3 columns: {line!r}')
            sample_id, rec_chain, lig_chain = parts[:3]
            yield sample_id, rec_chain, lig_chain


def main() -> int:
    args = parse_args()
    index_path = args.index.resolve()
    pdb_root = args.pdb_root.resolve()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for sample_id, rec_chain, lig_chain in iter_index_rows(index_path):
        pdb_path = pdb_root / f'{sample_id}.pdb'
        if not pdb_path.exists():
            msg = f'Warning: PDB not found for sample {sample_id}: {pdb_path}'
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr)
            missing.append(sample_id)
            continue
        rows.append({
            'pdb': str(pdb_path),
            'ligand_chain': lig_chain,
            'sample_id': sample_id,
            'receptor_chain': rec_chain,
        })

    if not rows:
        print('No valid entries found. Nothing written.', file=sys.stderr)
        return 1

    fieldnames = ['pdb', 'ligand_chain', 'sample_id', 'receptor_chain']
    with output_path.open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {output_path}')
    if missing:
        print(f'Skipped {len(missing)} samples with missing PDB files. See stderr warnings.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
