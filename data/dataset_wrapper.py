from typing import Callable
from tqdm import tqdm
from math import log

import numpy as np
import torch
import sympy

from utils import register as R


class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets, collate_fn: Callable=None) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        for dataset in datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn
        if hasattr(datasets[0], '_lengths'):
            self._lengths = []
            for dataset in datasets:
                self._lengths.extend(dataset._lengths)
    
    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()

    def get_len(self, idx):
        return self._lengths[idx]

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].__getitem__(idx - last_cum_len)
            last_cum_len = cum_len
        return None # this is not possible

    def get_type(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                ds = self.datasets[i]
                if hasattr(ds, 'get_type'):
                    return ds.get_type(idx - last_cum_len)
                return None
            last_cum_len = cum_len
        return None


@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        complexity,
        ubound_per_batch,
        type_balanced: bool = False,
        type_target: dict = None,
        type_seed: int = 0,
        type_total_samples: int = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.base_indexes = [i for i in range(len(dataset))]
        self.indexes = list(self.base_indexes)
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.type_balanced = type_balanced
        self.type_target = type_target or {}
        self.type_seed = type_seed
        self.type_total_samples = type_total_samples
        self._epoch = 0
        self._type_index_map = None
        if self.type_balanced:
            self._build_type_index_map()
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._epoch += 1
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):
        self.indexes = self._get_epoch_indexes()
        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_complexity = 0
        batch = []

        for i in tqdm(self.indexes):
            item_len = self.eval_func(self.dataset.get_len(i))
            if item_len > self.ubound_per_batch:
                continue
            cur_complexity += item_len
            if cur_complexity > self.ubound_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_complexity = item_len
            batch.append(i)
        self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def _build_type_index_map(self):
        if not hasattr(self.dataset, 'get_type'):
            self._type_index_map = None
            return
        type_index_map = {}
        for idx in self.base_indexes:
            label = self.dataset.get_type(idx)
            if label is None:
                continue
            type_index_map.setdefault(label, []).append(idx)
        self._type_index_map = type_index_map if type_index_map else None

    def _get_epoch_indexes(self):
        if not self.type_balanced or not self._type_index_map:
            return list(self.base_indexes)

        labels = list(self._type_index_map.keys())
        if self.type_total_samples is None or int(self.type_total_samples) <= 0:
            total = len(self.base_indexes)
        else:
            total = int(self.type_total_samples)
        target = {k: float(self.type_target.get(k, 0.0)) for k in labels}
        target_sum = sum(target.values())
        if target_sum <= 0:
            target = {k: 1.0 for k in labels}
            target_sum = float(len(labels))
        target = {k: v / target_sum for k, v in target.items()}

        desired = {}
        for k in labels:
            desired[k] = int(round(total * target[k]))
        diff = total - sum(desired.values())
        if diff != 0:
            adjust_key = max(desired, key=desired.get)
            desired[adjust_key] += diff

        rng = np.random.RandomState(self.type_seed + self._epoch)
        sampled = []
        for k, n in desired.items():
            idxs = self._type_index_map.get(k, [])
            if not idxs:
                continue
            replace = n > len(idxs)
            sampled.extend(rng.choice(idxs, size=n, replace=replace).tolist())
        if len(sampled) < total:
            extra = rng.choice(self.base_indexes, size=total - len(sampled), replace=True).tolist()
            sampled.extend(extra)
        if len(sampled) > total:
            sampled = sampled[:total]
        rng.shuffle(sampled)
        return sampled

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)
