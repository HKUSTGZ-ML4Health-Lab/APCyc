"""Receptor-conditioned inference with explicit inputs and portable output manifests."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from apcyc.checkpoint import load_bundle, sha256
from apcyc.constants import IDX2TYPE
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.format import Atom, Block, VOCAB
from utils.const import sidechain_atoms
from utils.random_seed import setup_seed

OPTIONS = {'pbar', 'energy_func', 'energy_lambda', 'autoencoder_n_iter', 't_hard',
           'return_pair_trajectory', 'use_cg_guidance', 'property_ckpt', 'property_ckpts',
           'guidance_w', 'guidance_w_map', 'guidance_w_min', 'guidance_w_max', 'lambda_max',
           't_guidance_start', 't_guidance_end', 'guidance_grad_norm_balance',
           'guidance_grad_norm_eps', 'guidance_grad_norm_ref'}
OPTIONS.update({'property_guidance_mode', 'property_scorer'})


def sampling_options(config, base):
    opt = dict(config.get('sample_opt', {}))
    unknown = set(opt) - OPTIONS
    if unknown:
        raise ValueError(f'Unknown sampling options: {sorted(unknown)}')
    def resolve(p):
        p = Path(p).expanduser()
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            raise FileNotFoundError(f'Property checkpoint not found: {p}')
        return str(p.resolve())
    for k in ['property_ckpt', 'property_ckpts', 'property_scorer']:
        v = opt.get(k)
        if isinstance(v, dict):
            opt[k] = {name: resolve(p) for name, p in v.items()}
        elif isinstance(v, list):
            opt[k] = [resolve(p) for p in v]
        elif v:
            opt[k] = resolve(v)
    if opt.get('use_cg_guidance'):
        if not (opt.get('property_ckpt') or opt.get('property_ckpts') or opt.get('property_scorer')):
            raise ValueError('Classifier guidance requires property checkpoint(s).')
        from apcyc.guidance import validate_guidance_options
        validate_guidance_options(opt)
    return opt


def read_manifest(path):
    records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    seen = set()
    for record in records:
        sid = record['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid) or sid in seen:
            raise ValueError(f'Invalid or duplicate sample_id: {sid}')
        seen.add(sid)
        for key in ['complex_pdb', 'pocket_json']:
            p = Path(record[key])
            record[key] = p if p.is_absolute() else path.parent / p
            if not record[key].is_file():
                raise FileNotFoundError(record[key])
    if not records:
        raise ValueError('Empty input manifest')
    return records


def prepare_input(record, length, device):
    rec_chain = record.get('receptor_chain', 'R')
    pep_chain = record.get('peptide_chain', 'L')
    if len(rec_chain) != 1 or len(pep_chain) != 1 or rec_chain == pep_chain:
        raise ValueError('PDB receptor and output peptide chains must be distinct one-character IDs.')
    receptor, = pdb_to_list_blocks(str(record['complex_pdb']), [rec_chain])
    pocket = set()
    for chain, position in json.loads(record['pocket_json'].read_text()):
        if chain == rec_chain:
            if isinstance(position, (list, tuple)):
                pocket.add((int(position[0]), str(position[1] or ' ') if len(position) > 1 else ' '))
            else:
                pocket.add((int(position), ' '))
    kept = [(i, b) for i, b in enumerate(receptor) if (int(b.id[0]), b.id[1] or ' ') in pocket]
    if not kept:
        raise ValueError(f"No matching pocket residues for {record['sample_id']}")
    X, S, masks, positions = [], [], [], []
    for i, block in kept:
        symbol = VOCAB.abrv_to_symbol(block.abrv)
        coords = {a.name: a.coordinate for a in block.units}
        mean = np.mean(list(coords.values()), axis=0).tolist()
        names = VOCAB.backbone_atoms + sidechain_atoms.get(symbol, [])
        atom_x = [coords.get(a, mean) for a in names]
        atom_mask = [a in coords for a in names]
        X.append(atom_x + [mean] * (14 - len(atom_x)))
        masks.append(atom_mask + [False] * (14 - len(atom_mask)))
        S.append(VOCAB.symbol_to_idx(symbol))
        positions.append(i + 1)
    n = len(X)
    ca = np.asarray(X, dtype=np.float64)[:, 1][np.asarray(masks)[:, 1]]
    if len(ca) < 3:
        raise ValueError('Pocket requires at least three C-alpha atoms for covariance normalization.')
    covariance = np.cov(ca, rowvar=False) + np.eye(3) * 1e-4
    X += [[[0., 0., 0.]] * 14 for _ in range(length)]
    masks += [[True] * 4 + [False] * 10 for _ in range(length)]
    S += [VOCAB.symbol_to_idx(VOCAB.LAT)] * length
    positions += list(range(1, length + 1))
    batch = dict(X=torch.tensor(X, dtype=torch.float32, device=device),
                 S=torch.tensor(S, dtype=torch.long, device=device),
                 mask=torch.tensor([False] * n + [True] * length, device=device),
                 atom_mask=torch.tensor(masks, dtype=torch.bool, device=device),
                 position_ids=torch.tensor(positions, dtype=torch.long, device=device),
                 lengths=torch.tensor([n + length], dtype=torch.long, device=device),
                 L=torch.tensor(np.linalg.cholesky(covariance), dtype=torch.float32, device=device).unsqueeze(0))
    return receptor, batch


def peptide_blocks(tokens, xyz):
    tokens = list(tokens)
    if len(tokens) != len(xyz):
        raise ValueError('Residue token count differs from coordinate count; expanded tokens must remain a list.')
    result, sequence = [], []
    for i, (token, coords) in enumerate(zip(tokens, xyz)):
        base = VOCAB.resolve_symbol(token)
        if base not in sidechain_atoms:
            raise ValueError(f'Unsupported predicted residue token: {token}')
        names = VOCAB.backbone_atoms + sidechain_atoms[base]
        if not np.isfinite(coords).all() or np.abs(coords).max() >= 1000:
            raise ValueError('Non-finite or out-of-range PDB coordinates; refusing silent coordinate clamping.')
        result.append(Block(VOCAB.symbol_to_abrv(base), [Atom(a, list(x), a[0]) for a, x in zip(names, coords)], id=(i + 1, ' ')))
        sequence.append(base)
    return result, ''.join(sequence)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory to avoid overwriting a prior run.')
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    opt = sampling_options(config or {}, args.config.resolve().parent if args.config else Path.cwd())
    model, metadata = load_bundle(args.bundle, args.device)
    if opt.get('energy_func') == 'default' and (model.consec_dist_mean is None or model.consec_dist_std is None):
        raise ValueError('Selected bundle has no latent-distance calibration; set energy_func: null.')
    records = read_manifest(args.input)
    if args.limit:
        records = records[:args.limit]
    started = time.monotonic()
    outputs = []
    for record in records:
        length = args.length if args.length is not None else record.get('peptide_length')
        if not isinstance(length, int) or length < 4:
            raise ValueError('Provide peptide_length >= 4 in the manifest or --length.')
        if length + 1 >= model.abs_position_encoding.num_embeddings:
            raise ValueError(f'Peptide length {length} exceeds the selected checkpoint position vocabulary.')
        receptor, batch = prepare_input(record, length, args.device)
        for k in range(args.num_samples):
            digest = hashlib.sha256(f'{args.seed}:{record["sample_id"]}:{k}'.encode()).digest()
            seed = int.from_bytes(digest[:4], 'little')
            setup_seed(seed)
            X, tokens, _, pair = model.sample(**batch, sample_opt=dict(opt), return_pair=True,
                                              optimize_sidechain=not args.no_sidechain_opt)
            ligand, sequence = peptide_blocks(tokens[0], X[0])
            local_pair = [int(x) for x in pair['pair_hat'][0].tolist()]
            if len(local_pair) != 2 or not all(0 <= i < length for i in local_pair):
                raise ValueError('Model returned out-of-range peptide-local pair indices.')
            filename = f'{record["sample_id"]}_gen_{k+1:02d}.pdb'
            list_blocks_to_pdb([receptor, ligand], [record.get('receptor_chain', 'R'), record.get('peptide_chain', 'L')], str(args.out_dir / filename))
            row = dict(sample_id=record['sample_id'], receptor_id=record['sample_id'], sample_idx=k + 1,
                       seed=seed, sequence=sequence, residue_tokens=list(tokens[0]), pep_len=length,
                       type_hat=IDX2TYPE[int(pair['type_hat'][0])], pair_hat=local_pair,
                       pair_index_base=0, pair_hat_resnames=[ligand[i].abrv for i in local_pair],
                       pair_pdb_residue_numbers=[i+1 for i in local_pair], pdb=filename,
                       receptor_chain=record.get('receptor_chain', 'R'), peptide_chain=record.get('peptide_chain', 'L'),
                       pocket_json=str(record['pocket_json'].resolve()),
                       structure_status='raw_prediction_not_relaxed', classifier_guidance=bool(opt.get('use_cg_guidance')))
            if opt.get('property_guidance_mode') == 'decoded_structure':
                row['property_guidance_mode'] = 'decoded_structure'
                row['property_guidance'] = pair.get('property_guidance', [])
            outputs.append(row)
            print(json.dumps(dict(sample=filename, type=row['type_hat'], length=length, guidance=row['classifier_guidance'])), flush=True)
    (args.out_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in outputs))
    property_files = opt.get('property_ckpts') or opt.get('property_ckpt') or {}
    property_paths = list(property_files.values()) if isinstance(property_files, dict) else (property_files if isinstance(property_files, list) else [property_files])
    if opt.get('property_scorer'):
        property_paths.append(opt['property_scorer'])
    code_root = Path(__file__).resolve().parents[1]
    code_files = sorted(p for folder in ['apcyc', 'models', 'data', 'utils'] for p in (code_root / folder).rglob('*.py'))
    implementation_hash = hashlib.sha256(''.join(str(p.relative_to(code_root)) + ':' + sha256(p) + '\n' for p in code_files).encode()).hexdigest()
    run_metadata = dict(implementation_sha256=implementation_hash, bundle_sha256=sha256(args.bundle), bundle_provenance=metadata['provenance'],
                        input_manifest_sha256=sha256(args.input), input_files={r['sample_id']: {k: sha256(r[k]) for k in ['complex_pdb', 'pocket_json']} for r in records},
                        property_sha256={str(Path(p)): sha256(p) for p in property_paths if p}, seed=args.seed,
                        sample_options=opt, torch_version=str(torch.__version__), device=args.device,
                        gpu_name=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else None,
                        seconds=time.monotonic()-started, samples=len(outputs), sidechain_fitting=not args.no_sidechain_opt, retrained=False)
    (args.out_dir / 'run.json').write_text(json.dumps(run_metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--length', type=int)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-sidechain-opt', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1 or (args.limit is not None and args.limit < 1):
        parser.error('num-samples and limit must be positive')
    run(args)


if __name__ == '__main__':
    main()
