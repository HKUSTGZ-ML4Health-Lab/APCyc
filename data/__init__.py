#!/usr/bin/python
# -*- coding:utf-8 -*-
from .dataset_wrapper import MixDatasetWrapper, EpochDistributedSampler
from copy import deepcopy


import torch
from torch.utils.data import DataLoader

import utils.register as R
from utils.logger import print_log

def create_dataset(config: dict):
    # Ensure APCyc datasets are registered without importing during package init.
    import apcyc.dataset  # noqa: F401
    splits = []
    for split_name in ['train', 'valid', 'test']:
        split_config = config.get(split_name, None)
        if split_config is None:
            splits.append(None)
            continue
        if isinstance(split_config, list):
            dataset = MixDatasetWrapper(
                *[R.construct(cfg) for cfg in split_config]
            )
        else:
            dataset = R.construct(split_config)
        splits.append(dataset)
    return splits  # train/valid/test


def create_dataloader(dataset, config: dict, n_gpu: int=1, validation: bool=False):
    if 'wrapper' in config:
        wrapper_cfg = deepcopy(config['wrapper'])
        if validation and wrapper_cfg.get('class') == 'DynamicBatchWrapper':
            # Do not resample validation data.
            wrapper_cfg['type_balanced'] = False
            wrapper_cfg['type_total_samples'] = 0
        dataset = R.construct(wrapper_cfg, dataset=dataset)
    scope = config.get('batch_size_scope', 'global')
    if scope not in ('global', 'per_device'):
        raise ValueError('batch_size_scope must be global or per_device')
    batch_size = config.get('batch_size', n_gpu if scope == 'global' else 1)
    if validation:
        batch_size = config.get('val_batch_size', batch_size)
    shuffle = config.get('shuffle', False)
    num_workers = config.get('num_workers', 4)
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f'batch_size must be a positive integer, got {batch_size!r}')
    if n_gpu > 1:
        if scope == 'global':
            if batch_size < n_gpu or batch_size % n_gpu:
                raise ValueError(f'Global batch_size={batch_size} must be divisible by world size {n_gpu}; '
                                 'use batch_size_scope: per_device for a per-GPU batch.')
            batch_size //= n_gpu
        sampler = EpochDistributedSampler(dataset, shuffle=shuffle)
        print_log(f'Batch size on a single GPU: {batch_size}')
    else:
        sampler = None
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(shuffle and sampler is None),
        collate_fn=collate_fn,
        sampler=sampler
    )
