#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import json
from itertools import islice
from pathlib import Path
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import print_log
from utils.random_seed import setup_seed
from apcyc.dataset import CPCoreDataset, CPCoreMMapDataset
from apcyc.property_predictor import PropertyPredictor
from data.dataset_wrapper import DynamicBatchWrapper

import yaml
from apcyc.property_schema import DEFAULT_PROPERTY_KEYS


def parse(argv=None):
    parser = argparse.ArgumentParser(description='Train latent property predictor', allow_abbrev=False)
    parser.add_argument('--config', type=str, default=None, help='APCyc LDM config (recommended)')
    parser.add_argument('--generator_bundle', help='Frozen generator bundle used by inference; required for training.')
    parser.add_argument('--t_hard', type=int, default=10)
    parser.add_argument('--train_index', type=str, required=True)
    parser.add_argument('--valid_index', type=str, required=True)
    parser.add_argument('--pdb_dir', type=str, default=None)
    parser.add_argument('--mmap_dir', type=str, default=None)
    parser.add_argument('--jsonl_path', type=str, default='cpcore_train.jsonl')
    parser.add_argument('--rec_chain', type=str, default='R')
    parser.add_argument('--lig_chain', type=str, default='L')
    parser.add_argument('--pocket_json_dir', type=str, default=None)
    parser.add_argument('--pocket_json_suffix', type=str, default='_complex_pocket.json')
    parser.add_argument('--require_pocket', action='store_true')
    parser.add_argument('--use_covariance_matrix', type=str, default='false')
    parser.add_argument('--property_csv', type=str, default='merged_properties_scored.csv')
    parser.add_argument('--id_col', type=str, default='id')
    parser.add_argument('--property_keys', type=str, default=None)
    parser.add_argument('--k_properties', type=int, default=len(DEFAULT_PROPERTY_KEYS))
    parser.add_argument('--lambda_k', type=str, default=None)
    parser.add_argument('--pos_embed_size', type=int, default=4096)
    parser.add_argument('--gpus', type=int, default=0, help='GPU id to use, -1 for cpu')
    parser.add_argument('--predictor_hidden_size', type=int, default=0)
    parser.add_argument('--predictor_layers', type=int, default=0)
    parser.add_argument('--predictor_mlp_hidden', type=int, default=0)
    parser.add_argument('--predictor_use_attn_pool', type=str, default=None)
    parser.add_argument('--predictor_use_pair_stats', type=str, default=None)
    parser.add_argument('--predictor_ckpt', type=str, default=None)
    parser.add_argument('--strict_predictor_ckpt', type=str, default='true')
    parser.add_argument('--validate_only', action='store_true')
    parser.add_argument('--separate_property_models', type=str, default='false')
    parser.add_argument('--per_property_hidden_size', type=int, default=0)
    parser.add_argument('--per_property_layers', type=int, default=0)
    parser.add_argument('--per_property_mlp_hidden', type=int, default=0)
    parser.add_argument('--use_dynamic_batch', type=str, default='true')
    parser.add_argument('--batch_complexity', type=str, default='n**2')
    parser.add_argument('--ubound_per_batch', type=int, default=60000)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val_full_batch', type=str, default='true')
    parser.add_argument('--property_use_pair_injection', type=str, default='true')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--plateau_factor', type=float, default=0.6)
    parser.add_argument('--plateau_patience', type=int, default=3)
    parser.add_argument('--plateau_min_lr', type=float, default=5.0e-6)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--save_topk', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ckpt_out', type=str, default='./ckpts/property_predictor.pt')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--preflight-batches', type=int, default=0,
                        help='Run bounded forward/backward and checkpoint checks, with zero optimizer updates.')
    parser.add_argument('--preflight-out', type=str)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.preflight_batches < 0 or (args.preflight_batches and not args.preflight_out):
        parser.error('--preflight-batches must be positive and requires --preflight-out')
    if args.preflight_batches and args.validate_only:
        parser.error('--preflight-batches and --validate_only are mutually exclusive')
    args._explicit_cli = {parser._option_string_actions[token.split('=', 1)[0]].dest
                          for token in argv if token.split('=', 1)[0] in parser._option_string_actions}
    args._config_types = {a.dest: a.type for a in parser._actions if a.dest != 'help'}
    return args


def _parse_bool(value: str) -> bool:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {value}')


def _apply_train_config(args, train_cfg: Dict) -> None:
    explicit = getattr(args, '_explicit_cli', set())
    for key, value in train_cfg.items():
        if key.startswith('_') or key in {'preflight_batches', 'preflight_out', 'validate_only'} or not hasattr(args, key):
            raise ValueError(f'Unknown property_train option: {key}')
        if key not in explicit:
            convert = getattr(args, '_config_types', {}).get(key)
            if key == 'property_keys' and isinstance(value, list):
                value = ','.join(value)
            elif convert is not None and value is not None:
                value = convert(str(value))
            setattr(args, key, value)


def _dataset_sample_ids(dataset) -> List[str]:
    if hasattr(dataset, 'sample_ids'):
        return list(dataset.sample_ids)
    if hasattr(dataset, '_indexes'):
        return [dataset.get_sample_id(i) for i in range(len(dataset))]
    return []


def _compute_property_stats(prop_map: Dict[str, List[float]], sample_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    values = [prop_map[sid] for sid in sample_ids if sid in prop_map]
    if not values:
        raise ValueError('No property values found for training IDs.')
    tensor = torch.tensor(values, dtype=torch.float)
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Nonfinite property normalization statistics.")
    return mean, std


def checked_property_backward(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss).all():
        raise RuntimeError('Nonfinite property loss; no optimizer update.')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError('Nonfinite property gradient; no optimizer update.')


def property_error_sums(prediction, target):
    error = prediction - target
    return error.abs().sum(dim=0), error.square().sum(dim=0)


def property_error_metrics(absolute_sum, squared_sum, count):
    if count <= 0 or not torch.isfinite(absolute_sum).all() or not torch.isfinite(squared_sum).all():
        raise ValueError('Empty or nonfinite property validation metrics.')
    return (absolute_sum / count).tolist(), (squared_sum / count).sqrt().tolist()


def _sanitize_key(key: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in key)


def _resolve_property_ckpt_path(base_path: str, key: str) -> str:
    safe_key = _sanitize_key(key)
    if os.path.isdir(base_path):
        return os.path.join(base_path, f'{safe_key}.pt')
    ckpt_dir = os.path.dirname(base_path) or '.'
    ckpt_base = os.path.splitext(os.path.basename(base_path))[0]
    return os.path.join(ckpt_dir, f'{ckpt_base}.{safe_key}.pt')


def _load_predictor_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Predictor checkpoint not found: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('state_dict'), dict):
        raise ValueError('Predictor checkpoint requires a state_dict and target metadata.')
    keys, cfg = payload.get('property_keys'), payload.get('model_cfg')
    if (not isinstance(keys, list) or not keys
            or not all(isinstance(key, str) and key.strip() for key in keys)
            or len(keys) != len(set(keys)) or not isinstance(cfg, dict)
            or cfg.get('n_properties') != len(keys)):
        raise ValueError('Predictor checkpoint target names must match its output dimension.')
    for name in ['property_mean', 'property_std']:
        try:
            value = torch.as_tensor(payload[name], dtype=torch.float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Predictor checkpoint requires valid {name}.') from exc
        if (value.shape != (len(keys),) or not torch.isfinite(value).all()
                or (name == 'property_std' and not (value > 0).all())):
            raise ValueError(f'Predictor checkpoint {name} must match targets and be finite; standard deviations must be positive.')
    return payload


def _load_separate_checkpoints(base_path, keys):
    names = [_sanitize_key(key) for key in keys]
    if len(names) != len(set(names)):
        raise ValueError('Separate property targets must produce distinct checkpoint filenames.')
    if not base_path:
        return {}
    paths = {key: _resolve_property_ckpt_path(base_path, key) for key in keys}
    missing = [str(path) for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing predictor checkpoints: {missing}')
    payloads = {key: _load_predictor_checkpoint(path) for key, path in paths.items()}
    for key, payload in payloads.items():
        if payload['property_keys'] != [key]:
            raise ValueError(f'Predictor checkpoint does not target {key}.')
    return payloads


def load_property_map(path: str, id_col: str, keys: List[str]) -> Dict[str, List[float]]:
    prop_map: Dict[str, List[float]] = {}
    seen = set()
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get(id_col, '').strip()
            if not sample_id:
                continue
            if sample_id in seen:
                raise ValueError(f'Duplicate property ID: {sample_id}')
            seen.add(sample_id)
            values = []
            valid = True
            for k in keys:
                v = row.get(k, None)
                if v is None or v == '':
                    valid = False
                    break
                try:
                    val = float(v)
                except ValueError:
                    valid = False
                    break
                if not math.isfinite(val):
                    valid = False
                    break
                values.append(val)
            if not valid:
                continue
            prop_map[sample_id] = values
    return prop_map


def infer_property_keys(path: str, id_col: str, requested=None) -> List[str]:
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    keys = list(DEFAULT_PROPERTY_KEYS if requested is None else requested)
    if not keys or len(set(keys)) != len(keys) or any(k in (id_col, 'sequence') for k in keys):
        raise ValueError('Property targets must be unique, nonempty numeric target columns.')
    if len(set(header)) != len(header):
        raise ValueError('Duplicate column names in property CSV.')
    missing = [k for k in [id_col, *keys] if k not in header]
    if missing:
        raise ValueError(f'Missing property columns: {missing}; use --property_keys for an explicit custom target set.')
    return keys


def filter_cpcore_dataset(dataset: CPCoreDataset, keep_ids: set):
    indices = [i for i, sid in enumerate(dataset.sample_ids) if sid in keep_ids]
    dataset.sample_ids = [dataset.sample_ids[i] for i in indices]
    dataset._lengths = [dataset._lengths[i] for i in indices]


def filter_mmap_dataset(dataset: CPCoreMMapDataset, keep_ids: set):
    new_indexes, new_props, new_lengths = [], [], []
    for idx, (entry, start, end) in enumerate(dataset._indexes):
        sample_id = dataset.get_sample_id(idx)
        if sample_id in keep_ids:
            new_indexes.append((entry, start, end))
            new_props.append(dataset._properties[idx])
            new_lengths.append(dataset._lengths[idx])
    dataset._indexes = new_indexes
    dataset._properties = new_props
    dataset._lengths = new_lengths


def build_property_predictor(
    diffusion,
    n_properties: int,
    pos_embed_size: int,
    predictor_hidden_size: int,
    predictor_layers: int,
    predictor_mlp_hidden: int,
    use_attn_pool: bool,
    use_pair_stats: bool,
) -> Tuple[PropertyPredictor, Dict[str, int]]:
    injector = diffusion.injector
    edge_attr_dim = injector.struct_proj.out_features
    if injector.include_pair_score:
        edge_attr_dim += 1
    if injector.include_type:
        edge_attr_dim += 3
    if injector.include_edge_type:
        edge_attr_dim += injector.edge_type_embed.embedding_dim
    if injector.include_cycle_edge:
        edge_attr_dim += 1
    dist_dim = diffusion.dist_rbf_dim if hasattr(diffusion, 'dist_rbf_dim') else 0
    edge_size = edge_attr_dim + dist_dim
    edge_type_vocab = injector.edge_type_embed.num_embeddings if injector.include_edge_type else 1

    hidden_size = predictor_hidden_size if predictor_hidden_size > 0 else diffusion.hidden_size
    n_layers = predictor_layers if predictor_layers > 0 else 3
    predictor = PropertyPredictor(
        latent_size=diffusion.latent_size,
        hidden_size=hidden_size,
        n_channel=diffusion.n_channel,
        n_layers=n_layers,
        edge_size=edge_size,
        n_rbf=diffusion.n_rbf,
        cutoff=diffusion.cutoff,
        dropout=diffusion.dropout,
        n_properties=n_properties,
        edge_type_vocab_size=edge_type_vocab,
        pos_embed_size=pos_embed_size,
        use_attn_pool=use_attn_pool,
        use_pair_stats=use_pair_stats,
        mlp_hidden_size=predictor_mlp_hidden,
    )
    cfg = {
        'latent_size': diffusion.latent_size,
        'hidden_size': hidden_size,
        'n_channel': diffusion.n_channel,
        'n_layers': n_layers,
        'edge_size': edge_size,
        'edge_type_vocab_size': edge_type_vocab,
        'n_rbf': diffusion.n_rbf,
        'cutoff': diffusion.cutoff,
        'dropout': diffusion.dropout,
        'pos_embed_size': pos_embed_size,
        'n_properties': n_properties,
        'use_attn_pool': use_attn_pool,
        'use_pair_stats': use_pair_stats,
        'mlp_hidden_size': predictor_mlp_hidden,
    }
    return predictor, cfg


def property_checkpoint_payload(predictor, cfg, keys, mean, std):
    return {'state_dict': {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
            'model_cfg': cfg, 'property_keys': list(keys),
            'feature_signature': getattr(predictor, 'feature_signature', None),
            'property_mean': torch.as_tensor(mean).tolist(), 'property_std': torch.as_tensor(std).tolist()}


def finish_property_preflight(args, predictors, cfgs, keys, mean, std, inputs, before, batches):
    from utils.training_preflight import verify_state_unchanged
    root = Path(args.preflight_out)
    inputs = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in inputs.items()}
    heads = []
    for idx, (predictor, cfg, initial) in enumerate(zip(predictors, cfgs, before)):
        verify_state_unchanged(initial, predictor)
        selected = [keys[idx]] if args.separate_property_models else keys
        m = mean[idx:idx + 1] if args.separate_property_models else mean
        s = std[idx:idx + 1] if args.separate_property_models else std
        payload = property_checkpoint_payload(predictor, cfg, selected, m, s)
        payload['preflight_only'] = True
        path = root / ((_sanitize_key(keys[idx]) if args.separate_property_models else 'multitask') + '.pt')
        torch.save(payload, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        restored = PropertyPredictor(**loaded['model_cfg'])
        restored.load_state_dict(loaded['state_dict'], strict=True)
        verify_state_unchanged(initial, restored)
        predictor.to('cpu').eval()
        restored.eval()
        with torch.no_grad():
            expected, actual = predictor(**inputs), restored(**inputs)
        if not torch.isfinite(actual).all() or not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
            raise RuntimeError('Reloaded property predictions differ.')
        heads.append({'keys': selected, 'checkpoint': str(path), 'exact_state_reload': True,
                      'prediction_max_abs_diff': float((expected - actual).abs().max()),
                      'mean': loaded['property_mean'], 'std': loaded['property_std']})
    report = {'optimizer_steps': 0, 'weights_unchanged': True, 'property_keys': keys,
              'batches': batches, 'heads': heads,
              'reload_output_comparison_device': 'cpu',
              'guidance_compatibility_validated': False,
              'feature_signature': args._feature_signature,
              'effective_args': {k: v for k, v in vars(args).items() if not k.startswith('_')}}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Property preflight passed: {len(batches)} batches, {len(heads)} head(s), zero optimizer updates.', flush=True)


def main():
    args = parse()
    if args.config is None:
        raise ValueError('Please provide --config with APCyc LDM config')
    config = yaml.safe_load(open(args.config, 'r'))
    _apply_train_config(args, config.get('property_train', {}))
    args.separate_property_models = _parse_bool(args.separate_property_models)
    if args.validate_only and not args.predictor_ckpt:
        raise ValueError('Use --predictor_ckpt for --validate_only.')
    setup_seed(args.seed)
    if args.gpus is not None and args.gpus >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpus)
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device('cpu')

    ckpt_payload = None
    if args.predictor_ckpt and not args.separate_property_models:
        ckpt_payload = _load_predictor_checkpoint(args.predictor_ckpt)

    requested_keys = None
    if ckpt_payload and not args.property_keys:
        ckpt_keys = ckpt_payload.get('property_keys')
        if ckpt_keys:
            requested_keys = list(ckpt_keys)
    if args.property_keys:
        requested_keys = [k.strip() for k in args.property_keys.split(',') if k.strip()]
    keys = infer_property_keys(args.property_csv, args.id_col, requested_keys)
    if ckpt_payload and ckpt_payload.get('property_keys') != keys:
        raise ValueError('Predictor checkpoint target order differs from the requested property targets.')
    separate_checkpoints = _load_separate_checkpoints(args.predictor_ckpt, keys) if args.separate_property_models else {}
    if args.preflight_batches:
        Path(args.preflight_out).mkdir(parents=True, exist_ok=False)
        args.ckpt_out = str(Path(args.preflight_out) / 'preflight.pt')
    if len(keys) != args.k_properties:
        print_log(f'Override k_properties from {args.k_properties} to {len(keys)}')
        args.k_properties = len(keys)
    prop_map = load_property_map(args.property_csv, args.id_col, keys)
    print_log(f'Loaded properties: {len(prop_map)} entries, keys={keys}')

    if args.predictor_use_attn_pool is not None:
        args.predictor_use_attn_pool = _parse_bool(args.predictor_use_attn_pool)
    if args.predictor_use_pair_stats is not None:
        args.predictor_use_pair_stats = _parse_bool(args.predictor_use_pair_stats)
    if args.val_full_batch is not None:
        args.val_full_batch = _parse_bool(args.val_full_batch)
    args.use_dynamic_batch = _parse_bool(args.use_dynamic_batch)
    args.property_use_pair_injection = _parse_bool(args.property_use_pair_injection)
    args.strict_predictor_ckpt = _parse_bool(args.strict_predictor_ckpt)
    args.use_covariance_matrix = _parse_bool(args.use_covariance_matrix)

    from apcyc.checkpoint import load_bundle
    from apcyc.property_features import feature_signature
    if not args.generator_bundle:
        raise ValueError('--generator_bundle is required: a randomly initialized frozen extractor is not reproducible.')
    if not args.property_use_pair_injection:
        raise ValueError('Property training must use the generator feature path; disabled pair injection is legacy-only.')
    model, _ = load_bundle(args.generator_bundle, device)
    if not 0 <= args.t_hard <= model.diffusion.num_steps:
        raise ValueError('t_hard must be within the diffusion schedule.')
    model.diffusion.resolve_topology_t_hard(args.t_hard)
    args._feature_signature = feature_signature(model._bundle_sha256, args.t_hard)
    if ckpt_payload and ckpt_payload.get('feature_signature') != args._feature_signature:
        raise ValueError('Predictor checkpoint has a different or unknown frozen feature extractor.')
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.mmap_dir:
        train_set = CPCoreMMapDataset(
            mmap_dir=args.mmap_dir,
            specify_index=args.train_index,
            jsonl_path=args.jsonl_path,
            backbone_only=False,
            use_covariance_matrix=args.use_covariance_matrix,
            return_supervision=False,
            return_sample_id=True,
        )
        valid_set = CPCoreMMapDataset(
            mmap_dir=args.mmap_dir,
            specify_index=args.valid_index,
            jsonl_path=args.jsonl_path,
            backbone_only=False,
            use_covariance_matrix=args.use_covariance_matrix,
            return_supervision=False,
            return_sample_id=True,
        )
    else:
        if args.pdb_dir is None:
            raise ValueError('Provide --pdb_dir when not using mmap')
        train_set = CPCoreDataset(
            pdb_root=args.pdb_dir,
            index_path=args.train_index,
            jsonl_path=args.jsonl_path,
            rec_chain=args.rec_chain,
            lig_chain=args.lig_chain,
            return_supervision=False,
            return_sample_id=True,
            use_covariance_matrix=args.use_covariance_matrix,
            pocket_json_dir=args.pocket_json_dir,
            pocket_json_suffix=args.pocket_json_suffix,
            require_pocket=args.require_pocket,
        )
        valid_set = CPCoreDataset(
            pdb_root=args.pdb_dir,
            index_path=args.valid_index,
            jsonl_path=args.jsonl_path,
            rec_chain=args.rec_chain,
            lig_chain=args.lig_chain,
            return_supervision=False,
            return_sample_id=True,
            use_covariance_matrix=args.use_covariance_matrix,
            pocket_json_dir=args.pocket_json_dir,
            pocket_json_suffix=args.pocket_json_suffix,
            require_pocket=args.require_pocket,
        )

    keep_ids = set(prop_map.keys())
    if isinstance(train_set, CPCoreMMapDataset):
        filter_mmap_dataset(train_set, keep_ids)
        filter_mmap_dataset(valid_set, keep_ids)
    else:
        filter_cpcore_dataset(train_set, keep_ids)
        filter_cpcore_dataset(valid_set, keep_ids)
    print_log(f'Filtered train: {len(train_set)}; valid: {len(valid_set)}')
    for name, dataset in [('training', train_set), ('validation', valid_set)]:
        if len(dataset) == 0:
            raise ValueError(f'No complete property rows remain in the {name} dataset.')

    train_ids = _dataset_sample_ids(train_set)
    prop_mean, prop_std = _compute_property_stats(prop_map, train_ids)
    if ckpt_payload and not args.separate_property_models:
        ckpt_mean = ckpt_payload.get('property_mean')
        ckpt_std = ckpt_payload.get('property_std')
        if ckpt_mean is not None and ckpt_std is not None:
            prop_mean = torch.as_tensor(ckpt_mean, dtype=prop_mean.dtype)
            prop_std = torch.as_tensor(ckpt_std, dtype=prop_std.dtype)
            print_log('Loaded property mean/std from predictor checkpoint.')
    print_log(f'Property mean: {prop_mean.tolist()}')
    print_log(f'Property std: {prop_std.tolist()}')

    if args.use_dynamic_batch:
        train_set = DynamicBatchWrapper(
            train_set,
            complexity=args.batch_complexity,
            ubound_per_batch=args.ubound_per_batch,
        )
    if args.validate_only and valid_set is not None and args.val_full_batch:
        args.val_batch_size = max(len(valid_set), 1)
        print_log(f'validate_only: override val_batch_size={args.val_batch_size}')
    train_loader = DataLoader(
        train_set,
        batch_size=1 if args.use_dynamic_batch else args.batch_size,
        shuffle=False if args.use_dynamic_batch else True,
        num_workers=args.num_workers,
        collate_fn=train_set.collate_fn if hasattr(train_set, 'collate_fn') else None,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=valid_set.collate_fn if hasattr(valid_set, 'collate_fn') else None,
    )

    use_attn_pool = True if args.predictor_use_attn_pool is None else args.predictor_use_attn_pool
    use_pair_stats = True if args.predictor_use_pair_stats is None else args.predictor_use_pair_stats
    if args.separate_property_models:
        per_hidden = args.per_property_hidden_size
        if per_hidden <= 0:
            per_hidden = args.predictor_hidden_size if args.predictor_hidden_size > 0 else max(64, model.diffusion.hidden_size // 2)
        per_layers = args.per_property_layers
        if per_layers <= 0:
            per_layers = args.predictor_layers if args.predictor_layers > 0 else 1
        per_mlp_hidden = args.per_property_mlp_hidden
        if per_mlp_hidden <= 0:
            per_mlp_hidden = args.predictor_mlp_hidden
        predictors = []
        predictor_cfgs = []
        optimizers = []
        schedulers = []
        for idx, key in enumerate(keys):
            ckpt = separate_checkpoints.get(key)
            if ckpt is not None and isinstance(ckpt, dict) and ckpt.get('model_cfg'):
                cfg = ckpt['model_cfg']
                pred = PropertyPredictor(**cfg)
            else:
                pred, cfg = build_property_predictor(
                    model.diffusion,
                    1,
                    args.pos_embed_size,
                    per_hidden,
                    per_layers,
                    per_mlp_hidden,
                    use_attn_pool,
                    use_pair_stats,
                )
            pred.feature_signature = args._feature_signature
            if ckpt is not None and ckpt.get('feature_signature') != args._feature_signature:
                raise ValueError('Per-property checkpoint has a different frozen feature extractor.')
            pred.to(device)
            if ckpt is not None:
                state_dict = ckpt.get('state_dict', ckpt)
                if args.strict_predictor_ckpt:
                    pred.load_state_dict(state_dict, strict=True)
                else:
                    incompatible = pred.load_state_dict(state_dict, strict=False)
                    if incompatible.missing_keys:
                        print_log(f'Predictor[{key}] missing {len(incompatible.missing_keys)} params.')
                    if incompatible.unexpected_keys:
                        print_log(f'Predictor[{key}] unexpected {len(incompatible.unexpected_keys)} params.')
                if isinstance(ckpt, dict) and 'property_mean' in ckpt and 'property_std' in ckpt:
                    prop_mean[idx] = float(ckpt['property_mean'][0])
                    prop_std[idx] = float(ckpt['property_std'][0])
                    if not torch.isfinite(prop_mean[idx]) or not torch.isfinite(prop_std[idx]) or prop_std[idx] <= 0:
                        raise ValueError(f'Invalid normalization statistics for property {key}.')
            pred.train()
            predictors.append(pred)
            predictor_cfgs.append(cfg)
            optimizers.append(torch.optim.AdamW(pred.parameters(), lr=args.lr))
            if args.scheduler.lower() == 'plateau':
                schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[-1],
                    factor=args.plateau_factor,
                    patience=args.plateau_patience,
                    mode='min',
                    min_lr=args.plateau_min_lr,
                ))
            else:
                schedulers.append(None)
    else:
        predictor_cfg = None
        if ckpt_payload and isinstance(ckpt_payload, dict) and ckpt_payload.get('model_cfg'):
            predictor_cfg = ckpt_payload['model_cfg']
            predictor = PropertyPredictor(**predictor_cfg)
        else:
            predictor, predictor_cfg = build_property_predictor(
                model.diffusion,
                args.k_properties,
                args.pos_embed_size,
                args.predictor_hidden_size,
                args.predictor_layers,
                args.predictor_mlp_hidden,
                use_attn_pool,
                use_pair_stats,
            )
        predictor.feature_signature = args._feature_signature
        predictor.to(device)
        if ckpt_payload:
            state_dict = ckpt_payload.get('state_dict', ckpt_payload)
            if args.strict_predictor_ckpt:
                predictor.load_state_dict(state_dict, strict=True)
            else:
                incompatible = predictor.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys:
                    print_log(f'Predictor checkpoint missing {len(incompatible.missing_keys)} params.')
                if incompatible.unexpected_keys:
                    print_log(f'Predictor checkpoint unexpected {len(incompatible.unexpected_keys)} params.')
        predictor.train()
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr)
        scheduler = None
        if args.scheduler.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                mode='min',
                min_lr=args.plateau_min_lr,
            )

    lambda_k = None
    if args.lambda_k:
        lambda_k = [float(x) for x in args.lambda_k.split(',')]
        if len(lambda_k) != args.k_properties:
            raise ValueError('lambda_k length mismatch')
        lambda_k = torch.tensor(lambda_k, device=device)

    best_metric = float('inf')
    bad_epochs = 0
    topk = {key: [] for key in keys} if args.separate_property_models else []
    ckpt_dir = os.path.dirname(args.ckpt_out) or '.'
    ckpt_base = os.path.splitext(os.path.basename(args.ckpt_out))[0]
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, f'{ckpt_base}_train.log')
    log_fh = open(log_path, 'a', encoding='utf-8')

    def log_line(message: str) -> None:
        print_log(message)
        log_fh.write(f'{message}\n')
        log_fh.flush()

    def run_validation() -> Tuple[float, List[float]]:
        if valid_loader is None:
            raise ValueError('Property validation requires a validation loader.')
        if args.separate_property_models:
            for pred in predictors:
                pred.eval()
        else:
            predictor.eval()
        with torch.no_grad():
            mae_sum = torch.zeros(args.k_properties, device=device)
            rmse_sum = torch.zeros(args.k_properties, device=device)
            mae_raw_sum = torch.zeros(args.k_properties, device=device)
            rmse_raw_sum = torch.zeros(args.k_properties, device=device)
            n_count = 0
            for batch in valid_loader:
                sample_ids = batch['sample_id']
                y_raw = torch.tensor([prop_map[sid] for sid in sample_ids], device=device)
                y = (y_raw - prop_mean.to(device)) / prop_std.to(device)
                X = batch['X'].to(device)
                S = batch['S'].to(device)
                mask = batch['mask'].to(device)
                position_ids = batch['position_ids'].to(device)
                lengths = batch['lengths'].to(device)
                atom_mask = batch['atom_mask'].to(device)
                L = batch.get('L', None)
                if L is not None:
                    L = L.to(device)

                with torch.no_grad():
                    H, Z, _, _ = model.autoencoder.encode(
                        X, S, mask, position_ids, lengths, atom_mask,
                        no_randomness=model.autoencoder_no_randomness
                    )
                    S_true = S
                    if model.train_sequence:
                        S = S.clone()
                        S[mask] = model.latent_idx

                    H_0, (atom_embeddings, _) = model.autoencoder.aa_feature(S, position_ids)
                    if model.train_sequence:
                        H_0 = model.hidden2latent(H_0)
                        H_0 = H_0.clone()
                        H_0[mask] = H

                    if model.train_structure:
                        X_0 = X.clone()
                        X_0[mask] = model.autoencoder._fill_latent_channels(Z, S_true[mask])
                        atom_mask = atom_mask.clone()
                        atom_mask_gen = atom_mask[mask]
                        atom_mask_gen[:, : model.autoencoder.latent_n_channel] = 1
                        atom_mask_gen[:, model.autoencoder.latent_n_channel :] = 0
                        atom_mask[mask] = atom_mask_gen
                    else:
                        X_0 = X

                batch_ids = model.diffusion._get_batch_ids(mask, lengths)
                t = torch.randint(1, model.diffusion.num_steps + 1, (1,), device=device).expand(int(batch_ids.max().item()) + 1)

                X_0_norm, centers = model.diffusion._normalize_position(X_0, batch_ids, mask, atom_mask, L)
                X_t, _ = model.diffusion.trans_x.add_noise(X_0_norm, mask, batch_ids, t)
                H_t, _ = model.diffusion.trans_h.add_noise(H_0, mask, batch_ids, t)

                from apcyc.property_features import property_features
                with torch.no_grad():
                    feature_S = S
                    if model.diffusion.chemistry_profile == 'paper':
                        feature_S = model.decode_residue_identities(H_t, model.diffusion._unnormalize_position(X_t, centers, batch_ids, L),
                                                                    S, mask, position_ids, lengths, atom_mask)
                    features = property_features(model.diffusion, H_t, X_t, feature_S, mask, lengths,
                                                 atom_mask, position_ids, batch_ids, centers, L,
                                                 t, args.t_hard, atom_embeddings)
                ctx_edges, inter_edges = features['ctx_edges'], features['inter_edges']
                edge_attr, edge_gate, edge_bias = features['edge_attr'], features['edge_gate'], features['edge_bias']
                edge_type_ids, p_c = features['edge_type'], features['type_probs']
                pair_out = dict(pair_struct=features['pair_struct'], pair_cyc=features['pair_cyc'], s_ij_full=features['s_ij'])

                if args.separate_property_models:
                    for idx, pred in enumerate(predictors):
                        y_hat = pred(
                            H_t,
                            X_t,
                            mask,
                            atom_mask,
                            position_ids,
                            ctx_edges,
                            inter_edges,
                            edge_attr=edge_attr,
                            edge_gate=edge_gate,
                            edge_bias=edge_bias,
                            edge_type=edge_type_ids,
                            pair_struct=pair_out['pair_struct'] if pair_out is not None else None,
                            pair_cyc=pair_out['pair_cyc'] if pair_out is not None else None,
                            type_probs=p_c,
                            s_ij=pair_out['s_ij_full'] if pair_out is not None else None,
                            t=t,
                            batch_ids=batch_ids,
                            atom_embeddings=atom_embeddings,
                        )
                        y_hat = y_hat.view(-1, 1)
                        y_k = y[:, idx:idx + 1]
                        y_raw_k = y_raw[:, idx:idx + 1]
                        absolute, squared = property_error_sums(y_hat, y_k)
                        mae_sum[idx] += absolute.item()
                        rmse_sum[idx] += squared.item()
                        y_hat_raw = y_hat * prop_std[idx].to(device) + prop_mean[idx].to(device)
                        absolute, squared = property_error_sums(y_hat_raw, y_raw_k)
                        mae_raw_sum[idx] += absolute.item()
                        rmse_raw_sum[idx] += squared.item()
                else:
                    y_hat = predictor(
                        H_t,
                        X_t,
                        mask,
                        atom_mask,
                        position_ids,
                        ctx_edges,
                        inter_edges,
                        edge_attr=edge_attr,
                        edge_gate=edge_gate,
                        edge_bias=edge_bias,
                        edge_type=edge_type_ids,
                        pair_struct=pair_out['pair_struct'] if pair_out is not None else None,
                        pair_cyc=pair_out['pair_cyc'] if pair_out is not None else None,
                        type_probs=p_c,
                        s_ij=pair_out['s_ij_full'] if pair_out is not None else None,
                        t=t,
                        batch_ids=batch_ids,
                        atom_embeddings=atom_embeddings,
                    )
                    absolute, squared = property_error_sums(y_hat, y)
                    mae_sum += absolute
                    rmse_sum += squared
                    y_hat_raw = y_hat * prop_std.to(device) + prop_mean.to(device)
                    absolute, squared = property_error_sums(y_hat_raw, y_raw)
                    mae_raw_sum += absolute
                    rmse_raw_sum += squared
                n_count += len(sample_ids)
            if n_count > 0:
                mae, rmse = property_error_metrics(mae_sum, rmse_sum, n_count)
                mae_raw, rmse_raw = property_error_metrics(mae_raw_sum, rmse_raw_sum, n_count)
                log_line(f'Validation MAE: {mae}')
                log_line(f'Validation RMSE: {rmse}')
                log_line(f'Validation MAE (raw): {mae_raw}')
                log_line(f'Validation RMSE (raw): {rmse_raw}')
                return float(sum(mae) / max(len(mae), 1)), mae
        raise ValueError('Property validation produced no samples.')

    if args.validate_only:
        log_line('Running validation only (no training).')
        metric, _ = run_validation()
        log_line(f'Validation-only mean MAE: {metric:.6f}')
        log_fh.close()
        return

    if args.preflight_batches:
        from utils.training_preflight import gradient_summary, verify_state_unchanged
        checked_predictors = predictors if args.separate_property_models else [predictor]
        checked_cfgs = predictor_cfgs if args.separate_property_models else [predictor_cfg]
        preflight_before = [{k: v.detach().cpu().clone() for k, v in pred.state_dict().items()} for pred in checked_predictors]
        extractor_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        preflight_checks = []
    if args.early_stop_patience >= 0 and not args.preflight_batches:
        log_line('Running pre-train validation (no checkpoint).')
        _ = run_validation()
    for epoch in range(1 if args.preflight_batches else args.epochs):
        if hasattr(train_set, 'update_epoch'):
            train_set.update_epoch()
        if args.separate_property_models:
            for pred in predictors:
                pred.train()
        else:
            predictor.train()
        total_loss = 0.0
        per_loss_sum = torch.zeros(args.k_properties, device=device)
        per_weighted_sum = torch.zeros(args.k_properties, device=device)
        n_batches = 0
        iterator = islice(train_loader, args.preflight_batches) if args.preflight_batches else train_loader
        for batch in tqdm(iterator, desc='preflight' if args.preflight_batches else f'train epoch {epoch}', leave=False):
            sample_ids = batch['sample_id']
            y_raw = torch.tensor([prop_map[sid] for sid in sample_ids], device=device)
            y = (y_raw - prop_mean.to(device)) / prop_std.to(device)

            X = batch['X'].to(device)
            S = batch['S'].to(device)
            mask = batch['mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            lengths = batch['lengths'].to(device)
            atom_mask = batch['atom_mask'].to(device)
            L = batch.get('L', None)
            if L is not None:
                L = L.to(device)

            with torch.no_grad():
                model.autoencoder.eval()
                H, Z, _, _ = model.autoencoder.encode(
                    X, S, mask, position_ids, lengths, atom_mask,
                    no_randomness=model.autoencoder_no_randomness
                )

                S_true = S
                if model.train_sequence:
                    S = S.clone()
                    S[mask] = model.latent_idx

                H_0, (atom_embeddings, _) = model.autoencoder.aa_feature(S, position_ids)
                if model.train_sequence:
                    H_0 = model.hidden2latent(H_0)
                    H_0 = H_0.clone()
                    H_0[mask] = H

                if model.train_structure:
                    X_0 = X.clone()
                    X_0[mask] = model.autoencoder._fill_latent_channels(Z, S_true[mask])
                    atom_mask = atom_mask.clone()
                    atom_mask_gen = atom_mask[mask]
                    atom_mask_gen[:, : model.autoencoder.latent_n_channel] = 1
                    atom_mask_gen[:, model.autoencoder.latent_n_channel :] = 0
                    atom_mask[mask] = atom_mask_gen
                else:
                    X_0 = X

            batch_ids = model.diffusion._get_batch_ids(mask, lengths)
            t = torch.randint(1, model.diffusion.num_steps + 1, (1,), device=device).expand(int(batch_ids.max().item()) + 1)

            X_0_norm, centers = model.diffusion._normalize_position(X_0, batch_ids, mask, atom_mask, L)
            X_t, _ = model.diffusion.trans_x.add_noise(X_0_norm, mask, batch_ids, t)
            H_t, _ = model.diffusion.trans_h.add_noise(H_0, mask, batch_ids, t)

            from apcyc.property_features import property_features
            with torch.no_grad():
                feature_S = S
                if model.diffusion.chemistry_profile == 'paper':
                    feature_S = model.decode_residue_identities(H_t, model.diffusion._unnormalize_position(X_t, centers, batch_ids, L),
                                                                S, mask, position_ids, lengths, atom_mask)
                features = property_features(model.diffusion, H_t, X_t, feature_S, mask, lengths,
                                             atom_mask, position_ids, batch_ids, centers, L,
                                             t, args.t_hard, atom_embeddings)
            ctx_edges, inter_edges = features['ctx_edges'], features['inter_edges']
            edge_attr, edge_gate, edge_bias = features['edge_attr'], features['edge_gate'], features['edge_bias']
            edge_type_ids, p_c = features['edge_type'], features['type_probs']
            pair_out = dict(pair_struct=features['pair_struct'], pair_cyc=features['pair_cyc'], s_ij_full=features['s_ij'])

            if args.separate_property_models:
                for idx, pred in enumerate(predictors):
                    y_hat = pred(
                        H_t,
                        X_t,
                        mask,
                        atom_mask,
                        position_ids,
                        ctx_edges,
                        inter_edges,
                        edge_attr=edge_attr,
                        edge_gate=edge_gate,
                        edge_bias=edge_bias,
                        edge_type=edge_type_ids,
                        pair_struct=pair_out['pair_struct'] if pair_out is not None else None,
                        pair_cyc=pair_out['pair_cyc'] if pair_out is not None else None,
                        type_probs=p_c,
                        s_ij=pair_out['s_ij_full'] if pair_out is not None else None,
                        t=t,
                        batch_ids=batch_ids,
                        atom_embeddings=atom_embeddings,
                    )
                    y_hat = y_hat.view(-1, 1)
                    y_k = y[:, idx:idx + 1]
                    loss_k = F.smooth_l1_loss(y_hat, y_k, reduction='mean')
                    per_loss_sum[idx] += loss_k.detach()
                    if lambda_k is not None:
                        loss_weighted = loss_k * lambda_k[idx]
                        per_weighted_sum[idx] += loss_weighted.detach()
                    else:
                        loss_weighted = loss_k
                    checked_property_backward(loss_weighted, pred, optimizers[idx])
                    if not args.preflight_batches:
                        optimizers[idx].step()
                    total_loss += float(loss_weighted.item())
            else:
                y_hat = predictor(
                    H_t,
                    X_t,
                    mask,
                    atom_mask,
                    position_ids,
                    ctx_edges,
                    inter_edges,
                    edge_attr=edge_attr,
                    edge_gate=edge_gate,
                    edge_bias=edge_bias,
                    edge_type=edge_type_ids,
                    pair_struct=pair_out['pair_struct'] if pair_out is not None else None,
                    pair_cyc=pair_out['pair_cyc'] if pair_out is not None else None,
                    type_probs=p_c,
                    s_ij=pair_out['s_ij_full'] if pair_out is not None else None,
                    t=t,
                    batch_ids=batch_ids,
                    atom_embeddings=atom_embeddings,
                )

                loss_per = F.smooth_l1_loss(y_hat, y, reduction='none')
                per_loss_sum += loss_per.detach().mean(dim=0)
                if lambda_k is not None:
                    per_weighted_sum += (loss_per.detach() * lambda_k[None, :]).mean(dim=0)
                loss = loss_per
                if lambda_k is not None:
                    loss = loss * lambda_k[None, :]
                loss = loss.mean()

                checked_property_backward(loss, predictor, optimizer)
                if not args.preflight_batches:
                    optimizer.step()

                total_loss += float(loss.item())
            n_batches += 1
            if args.preflight_batches:
                preflight_checks.append({'batch': n_batches, 'sample_ids': sample_ids,
                                         'cumulative_loss': total_loss,
                                         'gradients': [gradient_summary(pred) for pred in checked_predictors]})

        if args.preflight_batches:
            if n_batches != args.preflight_batches:
                raise ValueError(f'Requested {args.preflight_batches} preflight batches, got {n_batches}.')
            verify_state_unchanged(extractor_before, model)
            inputs = dict(H_noisy=H_t, X_noisy=X_t, mask_generate=mask, atom_mask=atom_mask,
                          position_ids=position_ids, ctx_edges=ctx_edges, inter_edges=inter_edges,
                          edge_attr=edge_attr, edge_gate=edge_gate, edge_bias=edge_bias, edge_type=edge_type_ids,
                          pair_struct=pair_out['pair_struct'] if pair_out is not None else None,
                          pair_cyc=pair_out['pair_cyc'] if pair_out is not None else None, type_probs=p_c,
                          s_ij=pair_out['s_ij_full'] if pair_out is not None else None, t=t,
                          batch_ids=batch_ids, atom_embeddings=atom_embeddings)
            finish_property_preflight(args, checked_predictors, checked_cfgs, keys, prop_mean, prop_std,
                                      inputs, preflight_before, preflight_checks)
            log_fh.close()
            return

        denom = max(n_batches, 1)
        log_line(f'Epoch {epoch+1}/{args.epochs} train loss {total_loss / denom:.4f}')
        per_loss = (per_loss_sum / denom).tolist()
        if keys:
            log_line(f'Property/Loss/Train: {dict(zip(keys, per_loss))}')
        else:
            log_line(f'Property/Loss/Train: {per_loss}')
        if lambda_k is not None:
            per_w = (per_weighted_sum / denom).tolist()
            if keys:
                log_line(f'Property/WeightedLoss/Train: {dict(zip(keys, per_w))}')
            else:
                log_line(f'Property/WeightedLoss/Train: {per_w}')

        val_metric, val_mae = run_validation()

        if args.separate_property_models:
            for idx, sch in enumerate(schedulers):
                if sch is not None and idx < len(val_mae):
                    sch.step(val_mae[idx])
            current_lr = optimizers[0].param_groups[0]['lr']
            log_line(f'LR: {current_lr:.2e}')
        else:
            if scheduler is not None:
                scheduler.step(val_metric)
                current_lr = optimizer.param_groups[0]['lr']
                log_line(f'LR: {current_lr:.2e}')

        if val_metric < best_metric:
            best_metric = val_metric
            bad_epochs = 0
        else:
            bad_epochs += 1
            log_line(f'Early stop counter: {bad_epochs}/{args.early_stop_patience}')
            if bad_epochs >= args.early_stop_patience:
                log_line('Early stopping triggered.')
                break

        os.makedirs(ckpt_dir, exist_ok=True)
        if args.separate_property_models:
            for idx, key in enumerate(keys):
                if idx >= len(val_mae):
                    continue
                pred = predictors[idx]
                cfg = predictor_cfgs[idx]
                ckpt = property_checkpoint_payload(pred, cfg, [key], prop_mean[idx:idx + 1], prop_std[idx:idx + 1])
                safe_val = f'{val_mae[idx]:.6f}'.replace('.', 'p')
                safe_key = _sanitize_key(key)
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f'{ckpt_base}.{safe_key}_epoch{epoch+1}_val{safe_val}.pt',
                )
                torch.save(ckpt, ckpt_path)
                log_line(f'Saved property predictor[{key}] to {ckpt_path}')
                topk[key].append((val_mae[idx], ckpt_path))
                topk[key] = sorted(topk[key], key=lambda x: x[0])
                if args.save_topk > 0 and len(topk[key]) > args.save_topk:
                    for _, drop_path in topk[key][args.save_topk:]:
                        if os.path.exists(drop_path):
                            os.remove(drop_path)
                    topk[key] = topk[key][: args.save_topk]
                topk_map = os.path.join(ckpt_dir, f'topk_map_{safe_key}.txt')
                with open(topk_map, 'w') as f:
                    for score, path in topk[key]:
                        f.write(f'{score:.6f}: {path}\n')
        else:
            ckpt = property_checkpoint_payload(predictor, predictor_cfg, keys, prop_mean, prop_std)
            safe_val = f'{val_metric:.6f}'.replace('.', 'p')
            ckpt_path = os.path.join(ckpt_dir, f'{ckpt_base}_epoch{epoch+1}_val{safe_val}.pt')
            torch.save(ckpt, ckpt_path)
            log_line(f'Saved property predictor to {ckpt_path}')
            topk.append((val_metric, ckpt_path))
            topk = sorted(topk, key=lambda x: x[0])
            if args.save_topk > 0 and len(topk) > args.save_topk:
                for _, drop_path in topk[args.save_topk:]:
                    if os.path.exists(drop_path):
                        os.remove(drop_path)
                topk = topk[: args.save_topk]
            topk_map = os.path.join(ckpt_dir, 'topk_map.txt')
            with open(topk_map, 'w') as f:
                for score, path in topk:
                    f.write(f'{score:.6f}: {path}\n')
    log_fh.close()


if __name__ == '__main__':
    main()
