"""Predict frozen surrogate scores for receptor-peptide structures."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from apcyc.checkpoint import sha256
from apcyc.property_scorer import load_scorer
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.format import VOCAB
from utils.const import sidechain_atoms


def structure_batch(record, base, device='cpu'):
    def resolve(name):
        path = Path(record[name])
        return path if path.is_absolute() else Path(base) / path
    receptor_chain, peptide_chain = record.get('receptor_chain', 'R'), record.get('peptide_chain', 'L')
    if (len(receptor_chain) != 1 or len(peptide_chain) != 1 or receptor_chain == peptide_chain):
        raise ValueError('Require distinct one-character receptor and peptide chains.')
    pdb = resolve('pdb')
    receptor, peptide = pdb_to_list_blocks(str(pdb), [receptor_chain, peptide_chain], strict=True)
    kept = list(enumerate(receptor))
    sources = {'pdb': sha256(pdb)}
    if record.get('pocket_json'):
        pocket_path = resolve('pocket_json')
        pocket = set()
        for chain, residue in json.loads(pocket_path.read_text()):
            if chain == receptor_chain:
                number, code = (residue[0], residue[1] if len(residue) > 1 else ' ') if isinstance(residue, (list, tuple)) else (residue, ' ')
                pocket.add((int(number), code or ' '))
        kept = [(i, block) for i, block in kept if (int(block.id[0]), block.id[1] or ' ') in pocket]
        sources['pocket_json'] = sha256(pocket_path)
    if len(kept) < 3 or not peptide:
        raise ValueError('Require at least three receptor residues and a nonempty peptide.')
    blocks = [block for _, block in kept] + peptide
    coordinates, tokens, atom_masks = [], [], []
    for block in blocks:
        symbol = VOCAB.abrv_to_symbol(block.abrv)
        if symbol not in sidechain_atoms:
            raise ValueError(f'Unsupported structure residue: {block.abrv}')
        atoms = {a.name: a.coordinate for a in block.units}
        if 'CA' not in atoms:
            raise ValueError('Every scoring residue must have a CA atom.')
        mean = np.mean(list(atoms.values()), axis=0).tolist()
        names = VOCAB.backbone_atoms + sidechain_atoms[symbol]
        coordinates.append([atoms.get(name, mean) for name in names] + [mean] * (14 - len(names)))
        atom_masks.append([name in atoms for name in names] + [False] * (14 - len(names)))
        tokens.append(VOCAB.symbol_to_idx(symbol))
    explicit_tokens = record.get('residue_tokens')
    sequence = ''.join(VOCAB.idx_to_symbol(token) for token in tokens[len(kept):])
    if 'sequence' in record and record['sequence'] != sequence:
        raise ValueError('Manifest sequence differs from the scoring PDB.')
    if explicit_tokens is not None:
        if not isinstance(explicit_tokens, list) or len(explicit_tokens) != len(peptide):
            raise ValueError('residue_tokens must contain one token per peptide residue.')
        for i, token in enumerate(explicit_tokens):
            actual = VOCAB.idx_to_symbol(tokens[len(kept) + i])
            if VOCAB.resolve_symbol(token) != actual or VOCAB.symbol_to_idx(token) == VOCAB.symbol_to_idx(VOCAB.UNK):
                raise ValueError('Peptide tokens differ from the structure or checkpoint vocabulary.')
            tokens[len(kept) + i] = VOCAB.symbol_to_idx(token)
    X = torch.tensor(coordinates, dtype=torch.float32, device=device)
    if not torch.isfinite(X).all():
        raise ValueError('Nonfinite input coordinates.')
    ca = X[:len(kept), 1].cpu().numpy()
    covariance = np.cov(ca, rowvar=False) + np.eye(3) * 1e-4
    batch = dict(X=X, S=torch.tensor(tokens, dtype=torch.long, device=device),
                 mask=torch.tensor([False] * len(kept) + [True] * len(peptide), device=device),
                 position_ids=torch.tensor([i + 1 for i, _ in kept] + list(range(1, len(peptide) + 1)), device=device),
                 atom_mask=torch.tensor(atom_masks, dtype=torch.bool, device=device),
                 lengths=torch.tensor([len(blocks)], dtype=torch.long, device=device),
                 L=torch.tensor(np.linalg.cholesky(covariance), dtype=torch.float32, device=device).unsqueeze(0))
    return batch, sources


def read_scoring_manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty scoring manifest.')
    seen = set()
    for row in rows:
        sid = row['id'] if 'id' in row else Path(row['pdb']).stem
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError('Scoring requires unique nonempty candidate IDs.')
        seen.add(sid)
        row['id'] = sid
    return rows


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory.')
    scorer = load_scorer(args.scorer, args.device)
    rows = read_scoring_manifest(args.manifest)
    records, sources = [], {}
    for row in rows:
        sid = row['id']
        batch, hashes = structure_batch(row, args.manifest.parent, args.device)
        with torch.no_grad():
            normalized = scorer(batch)
            raw = scorer.raw_scores(normalized)
        records.append({'id': sid, **{key: float(raw[0, i]) for i, key in enumerate(scorer.property_keys)},
                        **{key + '__normalized': float(normalized[0, i]) for i, key in enumerate(scorer.property_keys)}})
        sources[sid] = hashes
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / 'surrogate_scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    metadata = dict(format='apcyc.surrogate_scores.v1', scorer_sha256=sha256(args.scorer),
                    input_manifest_sha256=sha256(args.manifest), inputs=sources,
                    targets=scorer.property_keys, encoding='deterministic_AE_mean',
                    scorer_time=0, measured_properties=False, reconstruction=scorer.provenance,
                    samples=len(records), optimizer_steps=0)
    (args.out_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'samples': len(records), 'scores': str(args.out_dir / 'surrogate_scores.csv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
