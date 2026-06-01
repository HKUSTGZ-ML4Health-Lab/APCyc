#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from copy import deepcopy

import yaml
import torch
import inspect

import models
from utils import register as R
from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
from data import create_dataset, create_dataloader
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.format import VOCAB, Atom, Block
from utils.const import sidechain_atoms
from apcyc.constants import IDX2TYPE

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # noqa: BLE001
    tqdm = None

def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def _compare_state_dicts(ref_state, ckpt_state):
    missing = []
    unexpected = []
    mismatched = []
    for key, val in ckpt_state.items():
        if key not in ref_state:
            unexpected.append(key)
            continue
        if tuple(ref_state[key].shape) != tuple(val.shape):
            mismatched.append((key, tuple(val.shape), tuple(ref_state[key].shape)))
    for key in ref_state.keys():
        if key not in ckpt_state:
            missing.append(key)
    return missing, unexpected, mismatched


def overwrite_blocks(blocks, seq=None, X=None):
    if seq is not None:
        assert len(blocks) == len(seq), f'{len(blocks)} {len(seq)}'
    new_blocks = []
    for i, block in enumerate(blocks):
        block = deepcopy(block)
        if seq is None:
            abrv = block.abrv
        else:
            abrv = VOCAB.symbol_to_abrv(seq[i])
            if block.abrv != abrv and X is None:
                block.units = [atom for atom in block.units if atom.name in VOCAB.backbone_atoms]
        if X is not None:
            coords = X[i]
            atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
            block.units = [
                Atom(atom_name, clamp_coord(coord), atom_name[0]) for atom_name, coord in zip(atoms, coords)
            ]
        block.abrv = abrv
        new_blocks.append(block)
    return new_blocks


def build_blocks_from_pred(seq, X):
    blocks = []
    for i, sym in enumerate(seq):
        abrv = VOCAB.symbol_to_abrv(sym) or sym
        atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
        coords = X[i]
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().tolist()
        units = [
            Atom(atom_name, clamp_coord(coord), atom_name[0])
            for atom_name, coord in zip(atoms, coords)
        ]
        blocks.append(Block(abrv, units))
    return blocks


def get_receptor_id(sample_id, mode="prefix"):
    if mode == "sample":
        return sample_id
    if mode == "pdb4":
        return sample_id[:4]
    # default: prefix before first underscore
    return sample_id.split("_", 1)[0]


def _select_batch_item(batch, idx, batch_size):
    item = {}
    for key, val in batch.items():
        if key == 'sample_id':
            item[key] = [val[idx]] if isinstance(val, list) else [val]
            continue
        if torch.is_tensor(val) and val.dim() > 0 and val.size(0) == batch_size:
            item[key] = val[idx: idx + 1] if val.dim() > 1 else val[idx: idx + 1]
            continue
        if isinstance(val, list) and len(val) == batch_size:
            item[key] = [val[idx]]
            continue
        item[key] = val
    return item


def _rebuild_batch_with_pep_len(batch, target_len):
    mask = batch['mask']
    if mask.dim() > 1:
        mask = mask[0]
    X = batch['X']
    if X.dim() > 3:
        X = X[0]
    S = batch['S']
    if S.dim() > 1:
        S = S[0]
    atom_mask = batch['atom_mask']
    if atom_mask.dim() > 2:
        atom_mask = atom_mask[0]
    position_ids = batch['position_ids']
    if position_ids.dim() > 1:
        position_ids = position_ids[0]

    pocket_idx = torch.nonzero(~mask, as_tuple=False).view(-1)
    pocket_len = int(pocket_idx.numel())
    target_len = int(target_len)
    n_channel = X.size(-2)

    new_n = pocket_len + target_len
    new_X = torch.zeros((new_n, n_channel, 3), device=X.device, dtype=X.dtype)
    new_atom_mask = torch.zeros((new_n, n_channel), device=atom_mask.device, dtype=atom_mask.dtype)
    new_S = torch.full(
        (new_n,),
        fill_value=VOCAB.symbol_to_idx(VOCAB.LAT),
        device=S.device,
        dtype=S.dtype,
    )
    new_mask = torch.zeros((new_n,), device=mask.device, dtype=mask.dtype)
    new_position_ids = torch.zeros((new_n,), device=position_ids.device, dtype=position_ids.dtype)

    new_X[:pocket_len] = X[pocket_idx]
    new_atom_mask[:pocket_len] = atom_mask[pocket_idx]
    new_S[:pocket_len] = S[pocket_idx]
    new_mask[:pocket_len] = False
    new_mask[pocket_len:] = True
    new_position_ids[:pocket_len] = position_ids[pocket_idx]
    new_position_ids[pocket_len:] = torch.arange(
        1, target_len + 1, device=position_ids.device, dtype=position_ids.dtype
    )

    new_batch = dict(batch)
    new_batch['X'] = new_X
    new_batch['S'] = new_S
    new_batch['mask'] = new_mask
    new_batch['position_ids'] = new_position_ids
    new_batch['atom_mask'] = new_atom_mask
    new_batch['lengths'] = torch.tensor([new_n], device=mask.device, dtype=torch.long)
    return new_batch


def parse():
    parser = argparse.ArgumentParser(description='APCyc sampling')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2021)
    return parser.parse_known_args()


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    cyc_cfg = config.get('cyc_enhance', {})
    if cyc_cfg:
        from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
        set_config(cyc_cfg)
        apply_enlarged_vocab(cyc_cfg)

    sample_opt = dict(config.get('sample_opt', {}))
    strict_ckpt_check = bool(sample_opt.pop('strict_ckpt_check', False))
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model = torch.load(args.ckpt, map_location='cpu')
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Checkpoint does not contain a model object. Please use a model .ckpt.')
    if hasattr(model, 'autoencoder'):
        ae = model.autoencoder
        if not hasattr(ae, 'enable_latent_anchor_channels'):
            ae.enable_latent_anchor_channels = False
        if not hasattr(ae, 'latent_anchor_atoms'):
            ae.latent_anchor_atoms = None
        if not hasattr(ae, 'latent_anchor_pad_atom'):
            ae.latent_anchor_pad_atom = 'CA'
        if not hasattr(ae, 'latent_anchor_strict'):
            ae.latent_anchor_strict = False
        if not hasattr(ae, 'anchor_channel_map'):
            ae.anchor_channel_map = None
    if strict_ckpt_check:
        ref_model = R.construct(config['model'])
        missing, unexpected, mismatched = _compare_state_dicts(
            ref_model.state_dict(), model.state_dict()
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                'Checkpoint incompatible with config. '
                f'missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}'
            )
        del ref_model
    model.to(device)
    model.eval()

    dataset, _, _ = create_dataset(config['dataset'])
    loader = create_dataloader(dataset, config['dataloader'], n_gpu=1, validation=True)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'apcyc_predictions.jsonl')
    use_pbar = sample_opt.get('pbar', True)
    # Disable per-sample diffusion progress bar; keep only receptor-level bar.
    sample_opt['pbar'] = False
    group_by_receptor = sample_opt.get('group_by_receptor', True)
    receptor_id_mode = sample_opt.get('receptor_id_mode', 'prefix')
    num_samples = int(sample_opt.pop('num_samples', 1))
    base_seed = sample_opt.pop('sample_seed', None)
    enable_length_override = sample_opt.pop('enable_length_override', False)
    target_pep_len = sample_opt.pop('pep_len', None)
    pep_len_min = sample_opt.pop('pep_len_min', None)
    pep_len_max = sample_opt.pop('pep_len_max', None)
    auto_min_pep_len = sample_opt.pop('auto_min_pep_len', None)
    auto_min_threshold = sample_opt.pop('auto_min_pep_len_threshold', None)
    if auto_min_pep_len is not None:
        auto_min_pep_len = int(auto_min_pep_len)
        if auto_min_threshold is None:
            auto_min_threshold = auto_min_pep_len
        else:
            auto_min_threshold = int(auto_min_threshold)
    print_log(f'Sampling {len(dataset)} complexes. Output: {args.out_dir}')

    def _filter_sample_opt(model_obj, sample_opt_dict):
        if not isinstance(sample_opt_dict, dict):
            return dict(sample_opt_dict)
        if hasattr(model_obj, 'diffusion'):
            try:
                sig = inspect.signature(model_obj.diffusion.sample)
            except (TypeError, ValueError):
                return dict(sample_opt_dict)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return dict(sample_opt_dict)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in sample_opt_dict.items() if k in allowed}
            dropped = sorted(set(sample_opt_dict.keys()) - set(filtered.keys()))
            if dropped:
                print_log(f'Sampling: dropped unsupported sample_opt keys: {dropped}')
            return filtered
        return dict(sample_opt_dict)

    filtered_sample_opt = _filter_sample_opt(model, sample_opt)

    with open(results_path, 'w') as f:
        pbar = None
        if tqdm is not None and use_pbar:
            pbar = tqdm(total=len(dataset), desc='Sampling', unit='receptor')
        for batch in loader:
            batch = to_device(batch, device)
            lengths = batch.get('lengths', None)
            if lengths is not None:
                if not torch.is_tensor(lengths):
                    lengths = torch.tensor([lengths], dtype=torch.long, device=device)
                else:
                    lengths = lengths.to(device)
                    if lengths.dim() == 0:
                        lengths = lengths.view(1)
                batch['lengths'] = lengths
            sample_ids = batch.get('sample_id', None)
            if isinstance(sample_ids, str):
                batch['sample_id'] = [sample_ids]
            sample_ids = batch['sample_id']
            batch_size = len(sample_ids)
            def _sample_with_pair(model_obj, *sample_args, **sample_kwargs):
                params = inspect.signature(model_obj.sample).parameters
                supports_return_pair = 'return_pair' in params
                if supports_return_pair:
                    return model_obj.sample(*sample_args, return_pair=True, **sample_kwargs)
                out = model_obj.sample(*sample_args, **sample_kwargs)
                if isinstance(out, tuple) and len(out) == 3:
                    return out[0], out[1], out[2], None
                return out

            if enable_length_override or auto_min_pep_len is not None:
                if target_pep_len is None and pep_len_min is None:
                    if enable_length_override:
                        raise ValueError("enable_length_override requires pep_len or pep_len_min/pep_len_max")
                if pep_len_min is not None and pep_len_max is None:
                    pep_len_max = pep_len_min
                for i in range(batch_size):
                    single = _select_batch_item(batch, i, batch_size)
                    mask_i = single['mask']
                    if torch.is_tensor(mask_i) and mask_i.dim() > 1:
                        mask_i = mask_i[0]
                    if torch.is_tensor(mask_i):
                        mask_i = mask_i.detach().cpu()
                    pep_indices_current = torch.nonzero(mask_i, as_tuple=False).view(-1).tolist() if mask_i is not None else []
                    pep_len_current = len(pep_indices_current)
                    for k in range(num_samples):
                        if base_seed is not None:
                            torch.manual_seed(int(base_seed) + k)
                        if enable_length_override:
                            if target_pep_len is not None:
                                new_len = int(target_pep_len)
                            else:
                                new_len = int(torch.randint(int(pep_len_min), int(pep_len_max) + 1, (1,)).item())
                        else:
                            new_len = pep_len_current
                            if auto_min_pep_len is not None and pep_len_current < auto_min_threshold:
                                new_len = auto_min_pep_len
                        if new_len != pep_len_current:
                            single = _rebuild_batch_with_pep_len(single, new_len)
                        X, S, ppls, pair_out = _sample_with_pair(
                            model,
                            single['X'], single['S'], single['mask'], single['position_ids'], single['lengths'], single['atom_mask'],
                            L=single.get('L', None), sample_opt=dict(filtered_sample_opt),
                        )
                        for j, sample_id in enumerate(single['sample_id']):
                            mask_i = single['mask']
                            if torch.is_tensor(mask_i) and mask_i.dim() > 1:
                                mask_i = mask_i[j]
                            if torch.is_tensor(mask_i):
                                mask_i = mask_i.detach().cpu()
                            pep_indices = torch.nonzero(mask_i, as_tuple=False).view(-1).tolist() if mask_i is not None else []
                            pep_len = len(pep_indices)
                            rec_blocks, lig_blocks = pdb_to_list_blocks(
                                os.path.join(config['pdb_root'], sample_id + '.pdb'),
                                selected_chains=[config.get('rec_chain', 'R'), config.get('lig_chain', 'L')]
                            )
                            if len(lig_blocks) != len(S[j]):
                                print_log(
                                    f'Length override: rebuild ligand blocks for {sample_id} '
                                    f'({len(lig_blocks)} -> {len(S[j])}).'
                                )
                                lig_blocks = build_blocks_from_pred(S[j], X[j])
                            else:
                                lig_blocks = overwrite_blocks(lig_blocks, S[j], X[j])
                            if group_by_receptor:
                                receptor_id = get_receptor_id(sample_id, receptor_id_mode)
                                out_dir = os.path.join(args.out_dir, receptor_id)
                                os.makedirs(out_dir, exist_ok=True)
                            else:
                                receptor_id = sample_id
                                out_dir = args.out_dir
                            if num_samples == 1:
                                out_name = sample_id + '_gen.pdb'
                                sample_idx = 0
                            else:
                                out_name = f"{sample_id}_gen_{k + 1:02d}.pdb"
                                sample_idx = k + 1
                            out_pdb = os.path.join(out_dir, out_name)
                            list_blocks_to_pdb(
                                [rec_blocks, lig_blocks],
                                [config.get('rec_chain', 'R'), config.get('lig_chain', 'L')],
                                out_pdb,
                            )

                            if pair_out is not None:
                                type_hat = int(pair_out['type_hat'][j].item())
                                pair_hat = pair_out['pair_hat'][j].tolist()
                            else:
                                type_hat = None
                                pair_hat = None
                            pair_resnames = None
                            if pair_hat is not None and pep_len > 0 and len(pair_hat) == 2:
                                pair_i, pair_j = pair_hat
                                if 0 <= pair_i < pep_len and 0 <= pair_j < pep_len:
                                    seq_full = S[j]
                                    if isinstance(seq_full, str):
                                        gi = pep_indices[pair_i]
                                        gj = pep_indices[pair_j]
                                        if gi < len(seq_full) and gj < len(seq_full):
                                            sym_i = seq_full[gi]
                                            sym_j = seq_full[gj]
                                            res_i = VOCAB.symbol_to_abrv(sym_i) or sym_i
                                            res_j = VOCAB.symbol_to_abrv(sym_j) or sym_j
                                            pair_resnames = [res_i, res_j]
                            f.write(json.dumps({
                                'sample_id': sample_id,
                                'receptor_id': receptor_id,
                                'sample_idx': sample_idx,
                                'type_hat': IDX2TYPE[type_hat] if type_hat is not None else None,
                                'pair_hat': pair_hat,
                                'pep_len': pep_len,
                                'pair_hat_resnames': pair_resnames,
                                'pdb': out_pdb,
                            }) + '\n')
                if pbar is not None:
                    pbar.update(batch_size)
                continue

            for k in range(num_samples):
                if base_seed is not None:
                    torch.manual_seed(int(base_seed) + k)
                X, S, ppls, pair_out = _sample_with_pair(
                    model,
                    batch['X'], batch['S'], batch['mask'], batch['position_ids'], batch['lengths'], batch['atom_mask'],
                    L=batch.get('L', None), sample_opt=dict(filtered_sample_opt),
                )
                for i, sample_id in enumerate(batch['sample_id']):
                    mask_i = batch['mask']
                    if torch.is_tensor(mask_i) and mask_i.dim() > 1:
                        mask_i = mask_i[i]
                    if torch.is_tensor(mask_i):
                        mask_i = mask_i.detach().cpu()
                    pep_indices = torch.nonzero(mask_i, as_tuple=False).view(-1).tolist() if mask_i is not None else []
                    pep_len = len(pep_indices)
                    rec_blocks, lig_blocks = pdb_to_list_blocks(
                        os.path.join(config['pdb_root'], sample_id + '.pdb'),
                        selected_chains=[config.get('rec_chain', 'R'), config.get('lig_chain', 'L')]
                    )
                    if len(lig_blocks) != len(S[i]):
                        print_log(
                            f'Rebuild ligand blocks for {sample_id} '
                            f'({len(lig_blocks)} -> {len(S[i])}).'
                        )
                        lig_blocks = build_blocks_from_pred(S[i], X[i])
                    else:
                        lig_blocks = overwrite_blocks(lig_blocks, S[i], X[i])
                    if group_by_receptor:
                        receptor_id = get_receptor_id(sample_id, receptor_id_mode)
                        out_dir = os.path.join(args.out_dir, receptor_id)
                        os.makedirs(out_dir, exist_ok=True)
                    else:
                        receptor_id = sample_id
                        out_dir = args.out_dir
                    if num_samples == 1:
                        out_name = sample_id + '_gen.pdb'
                        sample_idx = 0
                    else:
                        out_name = f"{sample_id}_gen_{k + 1:02d}.pdb"
                        sample_idx = k + 1
                    out_pdb = os.path.join(out_dir, out_name)
                    list_blocks_to_pdb(
                        [rec_blocks, lig_blocks],
                        [config.get('rec_chain', 'R'), config.get('lig_chain', 'L')],
                        out_pdb,
                    )

                    if pair_out is not None:
                        type_hat = int(pair_out['type_hat'][i].item())
                        pair_hat = pair_out['pair_hat'][i].tolist()
                    else:
                        type_hat = None
                        pair_hat = None
                    pair_resnames = None
                    if pair_hat is not None and pep_len > 0 and len(pair_hat) == 2:
                        pair_i, pair_j = pair_hat
                        if 0 <= pair_i < pep_len and 0 <= pair_j < pep_len:
                            seq_full = S[i]
                            if isinstance(seq_full, str):
                                gi = pep_indices[pair_i]
                                gj = pep_indices[pair_j]
                                if gi < len(seq_full) and gj < len(seq_full):
                                    sym_i = seq_full[gi]
                                    sym_j = seq_full[gj]
                                    res_i = VOCAB.symbol_to_abrv(sym_i) or sym_i
                                    res_j = VOCAB.symbol_to_abrv(sym_j) or sym_j
                                    pair_resnames = [res_i, res_j]
                    f.write(json.dumps({
                        'sample_id': sample_id,
                        'receptor_id': receptor_id,
                        'sample_idx': sample_idx,
                        'type_hat': IDX2TYPE[type_hat] if type_hat is not None else None,
                        'pair_hat': pair_hat,
                        'pep_len': pep_len,
                        'pair_hat_resnames': pair_resnames,
                        'pdb': out_pdb,
                    }) + '\n')
            if pbar is not None:
                pbar.update(batch_size)
        if pbar is not None:
            pbar.close()

    print_log(f'Saved predictions to {results_path}')


if __name__ == '__main__':
    args, opt_args = parse()
    setup_seed(args.seed)
    main(args, opt_args)
