#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import argparse
import csv
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
from apcyc.cyclic_injection import cyclic_lambda
from apcyc.property_predictor import PropertyPredictor
from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
from data.dataset_wrapper import DynamicBatchWrapper

import yaml
import models
from utils import register as R


def parse():
    parser = argparse.ArgumentParser(description='Train latent property predictor')
    parser.add_argument('--config', type=str, default=None, help='APCyc LDM config (recommended)')
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
    parser.add_argument('--k_properties', type=int, default=7)
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
    parser.add_argument('--property_use_pair_injection', type=str, default='false')
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
    args = parser.parse_args()
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            defaults[action.dest] = action.default
    args._defaults = defaults
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
    defaults = getattr(args, '_defaults', {})
    for key, value in train_cfg.items():
        if not hasattr(args, key):
            continue
        if key in defaults and getattr(args, key) == defaults[key]:
            setattr(args, key, value)


def _dataset_sample_ids(dataset) -> List[str]:
    if hasattr(dataset, 'sample_ids'):
        return list(dataset.sample_ids)
    if hasattr(dataset, '_indexes'):
        return [entry.split('.')[0] for entry, _, _ in dataset._indexes]
    return []


def _compute_property_stats(prop_map: Dict[str, List[float]], sample_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    values = [prop_map[sid] for sid in sample_ids if sid in prop_map]
    if not values:
        raise ValueError('No property values found for training IDs.')
    tensor = torch.tensor(values, dtype=torch.float)
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0).clamp_min(1e-6)
    return mean, std


def _sanitize_key(key: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in key)


def _resolve_property_ckpt_path(base_path: str, key: str) -> str:
    safe_key = _sanitize_key(key)
    if os.path.isdir(base_path):
        return os.path.join(base_path, f'{safe_key}.pt')
    ckpt_dir = os.path.dirname(base_path) or '.'
    ckpt_base = os.path.splitext(os.path.basename(base_path))[0]
    return os.path.join(ckpt_dir, f'{ckpt_base}.{safe_key}.pt')


def load_property_map(path: str, id_col: str, keys: List[str]) -> Dict[str, List[float]]:
    prop_map: Dict[str, List[float]] = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get(id_col, '').strip()
            if not sample_id:
                continue
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
                if val != val:  # NaN check
                    valid = False
                    break
                values.append(val)
            if not valid:
                continue
            prop_map[sample_id] = values
    return prop_map


def infer_property_keys(path: str, id_col: str) -> List[str]:
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    keys = [k for k in header if k not in (id_col, 'sequence')]
    return keys


def filter_cpcore_dataset(dataset: CPCoreDataset, keep_ids: set):
    dataset.sample_ids = [sid for sid in dataset.sample_ids if sid in keep_ids]
    dataset._lengths = []
    for sample_id in dataset.sample_ids:
        record = dataset.meta.get(sample_id, {})
        pep_len = int(record.get('peptide_length', 0)) if record is not None else 0
        pocket_len = 0
        if dataset.pocket_json_dir is not None:
            pocket_set = dataset._load_pocket_set(sample_id)
            pocket_len = len(pocket_set) if pocket_set is not None else 0
        dataset._lengths.append(pep_len + pocket_len)


def filter_mmap_dataset(dataset: CPCoreMMapDataset, keep_ids: set):
    new_indexes, new_props, new_lengths = [], [], []
    for idx, (entry, start, end) in enumerate(dataset._indexes):
        sample_id = entry.split('.')[0]
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


def main():
    args = parse()
    setup_seed(args.seed)
    if args.gpus is not None and args.gpus >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpus)
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device('cpu')

    ckpt_payload = None
    if args.predictor_ckpt and os.path.isfile(args.predictor_ckpt):
        ckpt_payload = torch.load(args.predictor_ckpt, map_location='cpu')

    keys = infer_property_keys(args.property_csv, args.id_col)
    if ckpt_payload and not args.property_keys:
        ckpt_keys = ckpt_payload.get('property_keys')
        if ckpt_keys:
            keys = list(ckpt_keys)
    if args.property_keys:
        keys = [k.strip() for k in args.property_keys.split(',') if k.strip()]
    if len(keys) != args.k_properties:
        print_log(f'Override k_properties from {args.k_properties} to {len(keys)}')
        args.k_properties = len(keys)
    prop_map = load_property_map(args.property_csv, args.id_col, keys)
    print_log(f'Loaded properties: {len(prop_map)} entries, keys={keys}')

    if args.config is not None:
        config = yaml.safe_load(open(args.config, 'r'))
        train_cfg = config.get('property_train', {})
        if train_cfg:
            _apply_train_config(args, train_cfg)
        cyc_cfg = config.get('cyc_enhance', {})
        if cyc_cfg:
            set_config(cyc_cfg)
            apply_enlarged_vocab(cyc_cfg)
    else:
        raise ValueError('Please provide --config with APCyc LDM config')

    if args.predictor_use_attn_pool is not None:
        args.predictor_use_attn_pool = _parse_bool(args.predictor_use_attn_pool)
    if args.predictor_use_pair_stats is not None:
        args.predictor_use_pair_stats = _parse_bool(args.predictor_use_pair_stats)
    if args.val_full_batch is not None:
        args.val_full_batch = _parse_bool(args.val_full_batch)
    args.use_dynamic_batch = _parse_bool(args.use_dynamic_batch)
    args.property_use_pair_injection = _parse_bool(args.property_use_pair_injection)
    args.separate_property_models = _parse_bool(args.separate_property_models)
    args.strict_predictor_ckpt = _parse_bool(args.strict_predictor_ckpt)
    args.use_covariance_matrix = _parse_bool(args.use_covariance_matrix)

    model = R.construct(config['model'])
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
        ckpt_found = False
        for key in keys:
            ckpt = None
            ckpt_path = None
            if args.predictor_ckpt:
                ckpt_path = _resolve_property_ckpt_path(args.predictor_ckpt, key)
                if os.path.exists(ckpt_path):
                    ckpt_found = True
                    ckpt = torch.load(ckpt_path, map_location='cpu')
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
                    try:
                        prop_mean[idx] = float(ckpt['property_mean'][0])
                        prop_std[idx] = float(ckpt['property_std'][0])
                    except Exception:
                        pass
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
        if args.validate_only and args.predictor_ckpt and not ckpt_found:
            raise FileNotFoundError('No predictor checkpoints found for separate_property_models.')
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
            return float('inf'), []
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
                t = torch.randint(0, model.diffusion.num_steps + 1, (int(batch_ids.max().item()) + 1,), device=device)

                X_0_norm, centers = model.diffusion._normalize_position(X_0, batch_ids, mask, atom_mask, L)
                X_t, _ = model.diffusion.trans_x.add_noise(X_0_norm, mask, batch_ids, t)
                H_t, _ = model.diffusion.trans_h.add_noise(H_0, mask, batch_ids, t)

                ctx_edges, inter_edges = model.diffusion._get_edges(mask, batch_ids, lengths)
                edges = torch.cat([ctx_edges, inter_edges], dim=-1)

                X_t_unorm = model.diffusion._unnormalize_position(X_t, centers, batch_ids, L)
                if args.property_use_pair_injection:
                    pair_out = model.diffusion._pair_forward(
                        H_t, X_t_unorm, S_true, mask, batch_ids, position_ids, type_labels=None
                    )
                    p_c = torch.softmax(pair_out['type_logits'], dim=-1)
                    lambda_cyc = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_cyc_max, model.diffusion.lambda_cyc_schedule)
                    lambda_b = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_b_max, model.diffusion.lambda_cyc_schedule)
                    lambda_g = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_g_max, model.diffusion.lambda_cyc_schedule)

                    s_ij_inject = pair_out['s_ij_full'] + pair_out['s_ij_full'].transpose(0, 1)
                    edge_type = model.diffusion._build_edge_type(mask, hard_pair=None)
                    edge_type_ids = edge_type[edges[0], edges[1]]

                    pair_edge_attr = model.diffusion.injector(
                        edges,
                        batch_ids,
                        pair_out['pair_struct'],
                        s_ij_inject,
                        p_c,
                        lambda_cyc,
                        edge_type=edge_type_ids,
                        hard_pair=None,
                    )
                    ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
                    inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]

                    edge_bias, edge_gate = model.diffusion.bias_gate_injector(
                        edges,
                        batch_ids,
                        pair_out['pair_struct'],
                        s_ij_inject,
                        lambda_b,
                        lambda_g,
                    )

                    if hasattr(model.diffusion, 'dist_rbf'):
                        ctx_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, ctx_edges, atom_mask)
                        inter_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, inter_edges, atom_mask)
                        ctx_edge_attr = model.diffusion.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                        inter_edge_attr = model.diffusion.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                        ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                        inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
                    else:
                        ctx_edge_attr, inter_edge_attr = ctx_pair_attr, inter_pair_attr
                    edge_attr = torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)
                else:
                    pair_out = None
                    p_c = None
                    edge_type_ids = None
                    edge_bias, edge_gate = None, None
                    if hasattr(model.diffusion, 'dist_rbf'):
                        ctx_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, ctx_edges, atom_mask)
                        inter_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, inter_edges, atom_mask)
                        ctx_edge_attr = model.diffusion.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                        inter_edge_attr = model.diffusion.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                    else:
                        ctx_edge_attr = torch.zeros(ctx_edges.shape[1], 0, device=X_t.device)
                        inter_edge_attr = torch.zeros(inter_edges.shape[1], 0, device=X_t.device)
                    ctx_pair_attr = torch.zeros(ctx_edges.shape[1], model.diffusion.pair_edge_dim, device=X_t.device)
                    inter_pair_attr = torch.zeros(inter_edges.shape[1], model.diffusion.pair_edge_dim, device=X_t.device)
                    ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                    inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
                    edge_attr = torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)

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
                        mae_sum[idx] += torch.abs(y_hat - y_k).mean()
                        rmse_sum[idx] += torch.sqrt(torch.mean((y_hat - y_k) ** 2))
                        y_hat_raw = y_hat * prop_std[idx].to(device) + prop_mean[idx].to(device)
                        mae_raw_sum[idx] += torch.abs(y_hat_raw - y_raw_k).mean()
                        rmse_raw_sum[idx] += torch.sqrt(torch.mean((y_hat_raw - y_raw_k) ** 2))
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
                    mae_sum += torch.abs(y_hat - y).mean(dim=0)
                    rmse_sum += torch.sqrt(torch.mean((y_hat - y) ** 2, dim=0))
                    y_hat_raw = y_hat * prop_std.to(device) + prop_mean.to(device)
                    mae_raw_sum += torch.abs(y_hat_raw - y_raw).mean(dim=0)
                    rmse_raw_sum += torch.sqrt(torch.mean((y_hat_raw - y_raw) ** 2, dim=0))
                n_count += 1
            if n_count > 0:
                mae = (mae_sum / n_count).tolist()
                rmse = (rmse_sum / n_count).tolist()
                mae_raw = (mae_raw_sum / n_count).tolist()
                rmse_raw = (rmse_raw_sum / n_count).tolist()
                log_line(f'Validation MAE: {mae}')
                log_line(f'Validation RMSE: {rmse}')
                log_line(f'Validation MAE (raw): {mae_raw}')
                log_line(f'Validation RMSE (raw): {rmse_raw}')
                return float(sum(mae) / max(len(mae), 1)), mae
        return float('inf'), []

    if args.validate_only:
        if not args.predictor_ckpt:
            raise ValueError('Use --predictor_ckpt for --validate_only.')
        log_line('Running validation only (no training).')
        metric, _ = run_validation()
        log_line(f'Validation-only mean MAE: {metric:.6f}')
        log_fh.close()
        return

    if args.early_stop_patience >= 0:
        log_line('Running pre-train validation (no checkpoint).')
        _ = run_validation()
    for epoch in range(args.epochs):
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
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
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
            t = torch.randint(0, model.diffusion.num_steps + 1, (int(batch_ids.max().item()) + 1,), device=device)

            X_0_norm, centers = model.diffusion._normalize_position(X_0, batch_ids, mask, atom_mask, L)
            X_t, _ = model.diffusion.trans_x.add_noise(X_0_norm, mask, batch_ids, t)
            H_t, _ = model.diffusion.trans_h.add_noise(H_0, mask, batch_ids, t)

            ctx_edges, inter_edges = model.diffusion._get_edges(mask, batch_ids, lengths)
            edges = torch.cat([ctx_edges, inter_edges], dim=-1)

            with torch.no_grad():
                X_t_unorm = model.diffusion._unnormalize_position(X_t, centers, batch_ids, L)
                if args.property_use_pair_injection:
                    pair_out = model.diffusion._pair_forward(
                        H_t, X_t_unorm, S_true, mask, batch_ids, position_ids, type_labels=None
                    )
                    p_c = torch.softmax(pair_out['type_logits'], dim=-1)
                    lambda_cyc = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_cyc_max, model.diffusion.lambda_cyc_schedule)
                    lambda_b = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_b_max, model.diffusion.lambda_cyc_schedule)
                    lambda_g = cyclic_lambda(t, model.diffusion.num_steps, model.diffusion.lambda_g_max, model.diffusion.lambda_cyc_schedule)

                    s_ij_inject = pair_out['s_ij_full'] + pair_out['s_ij_full'].transpose(0, 1)
                    edge_type = model.diffusion._build_edge_type(mask, hard_pair=None)
                    edge_type_ids = edge_type[edges[0], edges[1]]

                    pair_edge_attr = model.diffusion.injector(
                        edges,
                        batch_ids,
                        pair_out['pair_struct'],
                        s_ij_inject,
                        p_c,
                        lambda_cyc,
                        edge_type=edge_type_ids,
                        hard_pair=None,
                    )
                    ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
                    inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]

                    edge_bias, edge_gate = model.diffusion.bias_gate_injector(
                        edges,
                        batch_ids,
                        pair_out['pair_struct'],
                        s_ij_inject,
                        lambda_b,
                        lambda_g,
                    )

                    if hasattr(model.diffusion, 'dist_rbf'):
                        ctx_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, ctx_edges, atom_mask)
                        inter_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, inter_edges, atom_mask)
                        ctx_edge_attr = model.diffusion.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                        inter_edge_attr = model.diffusion.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                        ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                        inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
                    else:
                        ctx_edge_attr, inter_edge_attr = ctx_pair_attr, inter_pair_attr
                    edge_attr = torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)
                else:
                    pair_out = None
                    p_c = None
                    edge_type_ids = None
                    edge_bias, edge_gate = None, None
                    if hasattr(model.diffusion, 'dist_rbf'):
                        ctx_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, ctx_edges, atom_mask)
                        inter_edge_attr = model.diffusion._get_edge_dist(X_t_unorm, inter_edges, atom_mask)
                        ctx_edge_attr = model.diffusion.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                        inter_edge_attr = model.diffusion.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                    else:
                        ctx_edge_attr = torch.zeros(ctx_edges.shape[1], 0, device=X_t.device)
                        inter_edge_attr = torch.zeros(inter_edges.shape[1], 0, device=X_t.device)
                    ctx_pair_attr = torch.zeros(ctx_edges.shape[1], model.diffusion.pair_edge_dim, device=X_t.device)
                    inter_pair_attr = torch.zeros(inter_edges.shape[1], model.diffusion.pair_edge_dim, device=X_t.device)
                    ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                    inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
                    edge_attr = torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)

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
                    optimizers[idx].zero_grad(set_to_none=True)
                    loss_weighted.backward()
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
                per_loss_sum += loss_per.mean(dim=0)
                if lambda_k is not None:
                    per_weighted_sum += (loss_per * lambda_k[None, :]).mean(dim=0)
                loss = loss_per
                if lambda_k is not None:
                    loss = loss * lambda_k[None, :]
                loss = loss.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
            n_batches += 1

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
                ckpt = {
                    'state_dict': pred.state_dict(),
                    'model_cfg': cfg,
                    'property_keys': [key],
                    'property_mean': [float(prop_mean[idx].item())],
                    'property_std': [float(prop_std[idx].item())],
                }
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
            ckpt = {
                'state_dict': predictor.state_dict(),
                'model_cfg': predictor_cfg,
                'property_keys': keys,
                'property_mean': prop_mean.tolist(),
                'property_std': prop_std.tolist(),
            }
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
