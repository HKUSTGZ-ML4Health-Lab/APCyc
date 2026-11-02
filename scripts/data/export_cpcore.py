#!/usr/bin/env python3
"""Export selected stored CPCore structures and pocket annotations without changing mmap data."""
import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.format import Block
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mmap-dir', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--ids', type=Path, help='One stored sample ID per line; default index order.')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError('Choose a new or empty output directory.')
    rows = [line.split('\t') for line in (a.mmap_dir / 'index.txt').read_text().splitlines()]
    by_id = {r[0]: r for r in rows}
    ids = list(dict.fromkeys(a.ids.read_text().splitlines())) if a.ids else list(by_id)
    if set(ids) - set(by_id):
        raise ValueError('Requested IDs absent from mmap index.')
    if a.limit:
        ids = ids[:a.limit]
    for part in ['complexes', 'pockets']:
        (a.out_dir / part).mkdir(parents=True, exist_ok=True)
    manifest = []
    with (a.mmap_dir / 'data.bin').open('rb') as f:
        for sid in ids:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid):
                raise ValueError('Sample ID is not a safe filename.')
            row = by_id[sid]
            start, end = map(int, row[1:3])
            f.seek(start)
            payload = f.read(end - start)
            chains = [[Block.from_tuple(b) for b in blocks] for blocks in json.loads(gzip.decompress(payload))]
            pdb_rel, pocket_rel = f'complexes/{sid}.pdb', f'pockets/{sid}.json'
            list_blocks_to_pdb(chains, ['R', 'L'], str(a.out_dir / pdb_rel))
            pocket = [['R', chains[0][int(i)].id] for i in row[-1].split(',')]
            (a.out_dir / pocket_rel).write_text(json.dumps(pocket) + '\n')
            manifest.append(dict(sample_id=sid, complex_pdb=pdb_rel, pocket_json=pocket_rel,
                                 receptor_chain='R', peptide_chain='L', peptide_length=len(chains[1]),
                                 source_record_sha256=hashlib.sha256(payload).hexdigest(),
                                 source='stored CPCore mmap; PDB formatting regenerated'))
    (a.out_dir / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    print(json.dumps({'exported': len(manifest), 'out': str(a.out_dir)}))


if __name__ == '__main__':
    main()
