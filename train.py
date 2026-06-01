#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse

import yaml
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values
from utils import register as R

########### Import your packages below ##########
import models
from trainer import create_trainer
from data import create_dataset, create_dataloader
from utils.nn_utils import count_parameters


def parse():
    parser = argparse.ArgumentParser(description='training')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # config
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    return parser.parse_known_args()


def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _extract_state_dict(model, trained):
    if isinstance(trained, dict):
        if 'state_dict' in trained:
            state_dict = trained['state_dict']
        elif 'model' in trained:
            state_dict = trained['model']
        else:
            state_dict = trained
    else:
        if hasattr(trained, 'module'):
            trained = trained.module
        if isinstance(model, models.AutoEncoder) and hasattr(trained, 'autoencoder'):
            state_dict = trained.autoencoder.state_dict()
        else:
            state_dict = trained.state_dict()

    if isinstance(model, models.AutoEncoder):
        if any(k.startswith('autoencoder.') for k in state_dict):
            state_dict = _strip_prefix(state_dict, 'autoencoder.')
    return state_dict


def _filter_state_dict_by_shape(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    unexpected = []
    mismatched = []
    for key, val in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != val.shape:
            mismatched.append((key, tuple(val.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = val
    missing = [k for k in model_state.keys() if k not in filtered]
    return filtered, missing, unexpected, mismatched


def load_ckpt(model, ckpt):
    strict = True
    skip_mismatch = False
    ckpt_path = ckpt
    if isinstance(ckpt, dict):
        ckpt_path = ckpt.get('path')
        strict = ckpt.get('strict', True)
        skip_mismatch = ckpt.get('skip_mismatch', not strict)
    trained_model = torch.load(ckpt_path, map_location='cpu')
    state_dict = _extract_state_dict(model, trained_model)
    if skip_mismatch:
        filtered, missing, unexpected, mismatched = _filter_state_dict_by_shape(model, state_dict)
        if mismatched:
            print_log(f'Loading checkpoint with {len(mismatched)} mismatched params (skipped).')
        if unexpected:
            print_log(f'Loading checkpoint with {len(unexpected)} unexpected params (skipped).')
        if missing:
            print_log(f'Loading checkpoint missing {len(missing)} params (kept init).')
        state_dict = filtered
        strict = False
    model.load_state_dict(state_dict, strict=strict)
    return model


def main(args, opt_args):

    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    print_log(f'Config: {args.config}')
    cyc_cfg = config.get('cyc_enhance', {})
    if cyc_cfg:
        from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
        set_config(cyc_cfg)
        apply_enlarged_vocab(cyc_cfg)

    ########## define your model #########
    model = R.construct(config['model'])
    if 'load_ckpt' in config:
        ckpt_path = config['load_ckpt']['path'] if isinstance(config['load_ckpt'], dict) else config['load_ckpt']
        print_log(f'Loading checkpoint: {ckpt_path}')
        model = load_ckpt(model, config['load_ckpt'])

    ########### load your train / valid set ###########
    train_set, valid_set, _ = create_dataset(config['dataset'])
    if args.local_rank <= 0:
        train_len = len(train_set) if train_set is not None else 0
        valid_len = len(valid_set) if valid_set is not None else 0
        print_log(f'Train samples: {train_len}, Valid samples: {valid_len}')

    ########## define your trainer/trainconfig #########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
    else:
        args.local_rank = -1

    if args.local_rank <= 0:
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = create_dataloader(train_set, config['dataloader'], len(args.gpus))
    valid_loader = create_dataloader(valid_set, config['dataloader'], validation=True)
    
    trainer = create_trainer(config, model, train_loader, valid_loader)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed)
    main(args, opt_args)
