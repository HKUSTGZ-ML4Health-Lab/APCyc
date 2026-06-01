#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Compare latent distributions from two AutoEncoder checkpoints on a dataset split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import print_log
from data import create_dataset, create_dataloader
from models.autoencoder.model import AutoEncoder
from apcyc.cyc_enhance import apply_enlarged_vocab, set_config


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_autoencoder(ckpt_path: str, strict: bool = True) -> AutoEncoder:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, AutoEncoder):
        return ckpt
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        if any(k.startswith('autoencoder.') for k in state_dict):
            state_dict = _strip_prefix(state_dict, 'autoencoder.')
        cfg = ckpt.get('model_cfg')
        if cfg is None:
            raise ValueError('Autoencoder checkpoint missing model_cfg for strict load.')
        autoencoder = AutoEncoder(**cfg)
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder
    raise TypeError(f'Unsupported autoencoder checkpoint type: {type(ckpt)}')


class RunningStats:
    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self.n = 0
        self.mean = None
        self.M2 = None

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        if self.dim is None:
            self.dim = x.shape[-1]
        x = x.view(-1, self.dim)
        if x.numel() == 0:
            return
        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
            return
        delta = batch_mean - self.mean
        total_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / total_n)
        self.M2 = self.M2 + batch_var * batch_n + delta.pow(2) * self.n * batch_n / total_n
        self.n = total_n

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n == 0:
            raise ValueError('No samples to finalize stats.')
        var = self.M2 / max(self.n, 1)
        std = torch.sqrt(var + 1e-8)
        return self.mean, std


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val.to(device)
        else:
            out[key] = val
    lengths = out.get('lengths')
    if lengths is not None and not torch.is_tensor(lengths):
        lengths = torch.tensor([lengths], dtype=torch.long, device=device)
        out['lengths'] = lengths
    return out


def _sym_kl_diag(mu_a: torch.Tensor, std_a: torch.Tensor, mu_b: torch.Tensor, std_b: torch.Tensor) -> float:
    var_a = std_a.pow(2).clamp_min(1e-8)
    var_b = std_b.pow(2).clamp_min(1e-8)
    kl_ab = 0.5 * (torch.log(var_b / var_a) + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1.0).sum()
    kl_ba = 0.5 * (torch.log(var_a / var_b) + (var_b + (mu_b - mu_a).pow(2)) / var_a - 1.0).sum()
    return float((kl_ab + kl_ba) / 2.0)


def _compute_latent_stats(
    autoencoder: AutoEncoder,
    loader,
    device: torch.device,
    max_samples: int,
    no_randomness: bool,
) -> Dict:
    h_stats = RunningStats()
    z_stats = RunningStats()
    h_norm = RunningStats(dim=1)
    z_norm = RunningStats(dim=1)

    autoencoder.eval()
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            H, Z, _, _ = autoencoder.encode(
                batch['X'],
                batch['S'],
                batch['mask'],
                batch['position_ids'],
                batch['lengths'],
                batch['atom_mask'],
                no_randomness=no_randomness,
            )
            if H is None:
                raise ValueError('Autoencoder returned None for latent H.')
            h_stats.update(H)
            h_norm.update(torch.norm(H, dim=-1, keepdim=True))

            if Z is not None:
                z_flat = Z.view(Z.shape[0], -1)
                z_stats.update(z_flat)
                z_norm.update(torch.norm(z_flat, dim=-1, keepdim=True))

            lengths = batch.get('lengths')
            if lengths is None:
                total_samples += 1
            else:
                total_samples += int(lengths.numel())
            if max_samples > 0 and total_samples >= max_samples:
                break

    h_mean, h_std = h_stats.finalize()
    z_mean, z_std = (None, None)
    if z_stats.n > 0:
        z_mean, z_std = z_stats.finalize()
    h_norm_mean, h_norm_std = h_norm.finalize()
    z_norm_mean, z_norm_std = (None, None)
    if z_norm.n > 0:
        z_norm_mean, z_norm_std = z_norm.finalize()

    return {
        'samples': total_samples,
        'h_mean': h_mean.cpu(),
        'h_std': h_std.cpu(),
        'h_norm_mean': h_norm_mean.item(),
        'h_norm_std': h_norm_std.item(),
        'z_mean': z_mean.cpu() if z_mean is not None else None,
        'z_std': z_std.cpu() if z_std is not None else None,
        'z_norm_mean': z_norm_mean.item() if z_norm_mean is not None else None,
        'z_norm_std': z_norm_std.item() if z_norm_std is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare latent distributions for two AE checkpoints.')
    parser.add_argument('--data_config', required=True, help='YAML with dataset/dataloader sections.')
    parser.add_argument('--ckpt_a', required=True, help='First autoencoder checkpoint path.')
    parser.add_argument('--ckpt_b', required=True, help='Second autoencoder checkpoint path.')
    parser.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--max_samples', type=int, default=0, help='Limit number of sequences (0 = all).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU.')
    parser.add_argument('--no_randomness', type=str, default='true')
    parser.add_argument('--out_json', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cpu' if args.gpu is None or args.gpu < 0 or not torch.cuda.is_available() else f'cuda:{args.gpu}')
    no_randomness = args.no_randomness.strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    config = yaml.safe_load(open(args.data_config, 'r'))
    cyc_cfg = config.get('cyc_enhance', {})
    if cyc_cfg:
        set_config(cyc_cfg)
        apply_enlarged_vocab(cyc_cfg)
    train_set, valid_set, test_set = create_dataset(config['dataset'])
    split_map = {'train': train_set, 'valid': valid_set, 'test': test_set}
    dataset = split_map.get(args.split)
    if dataset is None:
        raise ValueError(f'Split {args.split} not found in {args.data_config}')
    loader = create_dataloader(dataset, config.get('dataloader', {}), validation=(args.split != 'train'))

    ae_a = load_autoencoder(args.ckpt_a, strict=True).to(device)
    ae_b = load_autoencoder(args.ckpt_b, strict=True).to(device)

    print_log(f'Comparing {args.ckpt_a} vs {args.ckpt_b} on {args.split} split.')
    stats_a = _compute_latent_stats(ae_a, loader, device, args.max_samples, no_randomness)
    stats_b = _compute_latent_stats(ae_b, loader, device, args.max_samples, no_randomness)

    h_mean_a, h_std_a = stats_a['h_mean'], stats_a['h_std']
    h_mean_b, h_std_b = stats_b['h_mean'], stats_b['h_std']
    h_dim_match = h_mean_a.shape == h_mean_b.shape

    z_mean_a, z_std_a = stats_a['z_mean'], stats_a['z_std']
    z_mean_b, z_std_b = stats_b['z_mean'], stats_b['z_std']
    z_dim_match = (z_mean_a is not None and z_mean_b is not None and z_mean_a.shape == z_mean_b.shape)

    diff = {
        'h_dim_match': h_dim_match,
        'z_dim_match': z_dim_match,
        'h_norm_mean_diff': float(abs(stats_a['h_norm_mean'] - stats_b['h_norm_mean'])),
        'h_norm_std_diff': float(abs(stats_a['h_norm_std'] - stats_b['h_norm_std'])),
        'z_norm_mean_diff': None,
        'z_norm_std_diff': None,
        'h_mean_l2': None,
        'h_std_l2': None,
        'h_sym_kl': None,
        'z_mean_l2': None,
        'z_std_l2': None,
        'z_sym_kl': None,
    }

    if stats_a['z_norm_mean'] is not None and stats_b['z_norm_mean'] is not None:
        diff['z_norm_mean_diff'] = float(abs(stats_a['z_norm_mean'] - stats_b['z_norm_mean']))
        diff['z_norm_std_diff'] = float(abs(stats_a['z_norm_std'] - stats_b['z_norm_std']))

    if h_dim_match:
        diff['h_mean_l2'] = float(torch.norm(h_mean_a - h_mean_b).item())
        diff['h_std_l2'] = float(torch.norm(h_std_a - h_std_b).item())
        diff['h_sym_kl'] = _sym_kl_diag(h_mean_a, h_std_a, h_mean_b, h_std_b)

    if z_dim_match:
        diff['z_mean_l2'] = float(torch.norm(z_mean_a - z_mean_b).item())
        diff['z_std_l2'] = float(torch.norm(z_std_a - z_std_b).item())
        diff['z_sym_kl'] = _sym_kl_diag(z_mean_a, z_std_a, z_mean_b, z_std_b)

    print_log(f"H dim match: {diff['h_dim_match']}, Z dim match: {diff['z_dim_match']}")
    print_log(f"H norm mean/std diff: {diff['h_norm_mean_diff']:.6f}, {diff['h_norm_std_diff']:.6f}")
    if diff['z_norm_mean_diff'] is not None:
        print_log(f"Z norm mean/std diff: {diff['z_norm_mean_diff']:.6f}, {diff['z_norm_std_diff']:.6f}")
    if diff['h_mean_l2'] is not None:
        print_log(f"H mean/std L2: {diff['h_mean_l2']:.6f}, {diff['h_std_l2']:.6f}")
        print_log(f"H sym KL (diag): {diff['h_sym_kl']:.6f}")
    if diff['z_mean_l2'] is not None:
        print_log(f"Z mean/std L2: {diff['z_mean_l2']:.6f}, {diff['z_std_l2']:.6f}")
        print_log(f"Z sym KL (diag): {diff['z_sym_kl']:.6f}")

    if args.out_json:
        payload = {
            'ckpt_a': args.ckpt_a,
            'ckpt_b': args.ckpt_b,
            'split': args.split,
            'samples_a': stats_a['samples'],
            'samples_b': stats_b['samples'],
            'diff': diff,
            'stats_a': {
                'h_mean': stats_a['h_mean'].tolist(),
                'h_std': stats_a['h_std'].tolist(),
                'h_norm_mean': stats_a['h_norm_mean'],
                'h_norm_std': stats_a['h_norm_std'],
                'z_mean': stats_a['z_mean'].tolist() if stats_a['z_mean'] is not None else None,
                'z_std': stats_a['z_std'].tolist() if stats_a['z_std'] is not None else None,
                'z_norm_mean': stats_a['z_norm_mean'],
                'z_norm_std': stats_a['z_norm_std'],
            },
            'stats_b': {
                'h_mean': stats_b['h_mean'].tolist(),
                'h_std': stats_b['h_std'].tolist(),
                'h_norm_mean': stats_b['h_norm_mean'],
                'h_norm_std': stats_b['h_norm_std'],
                'z_mean': stats_b['z_mean'].tolist() if stats_b['z_mean'] is not None else None,
                'z_std': stats_b['z_std'].tolist() if stats_b['z_std'] is not None else None,
                'z_norm_mean': stats_b['z_norm_mean'],
                'z_norm_std': stats_b['z_norm_std'],
            },
        }
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print_log(f'Wrote report: {args.out_json}')


if __name__ == '__main__':
    main()
