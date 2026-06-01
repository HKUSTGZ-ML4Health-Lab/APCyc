#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from utils import register as R
from utils.const import sidechain_atoms
from data.format import VOCAB, Block
from data.mmap_dataset import MMAPDataset
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks

from apcyc.constants import TYPE2IDX
from apcyc.cyc_enhance import get_config as get_cyc_config, rewrite_tokens_by_cyclization


def _normalize_icode(icode) -> str:
    if icode is None:
        return ' '
    if not isinstance(icode, str):
        return str(icode)
    return icode if icode != '' else ' '


def calculate_covariance_matrix(point_cloud):
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    return covariance_matrix


@R.register('CPCoreDataset')
class CPCoreDataset(torch.utils.data.Dataset):
    MAX_N_ATOM = 14

    def __init__(
        self,
        pdb_root: str,
        index_path: str,
        jsonl_path: str,
        rec_chain: str = 'R',
        lig_chain: str = 'L',
        backbone_only: bool = False,
        padding_collate: bool = False,
        use_covariance_matrix: bool = False,
        return_supervision: bool = True,
        return_sample_id: bool = False,
        pocket_json_dir: Optional[str] = None,
        pocket_json_suffix: str = '_complex_pocket.json',
        require_pocket: bool = False,
        strict_context: bool = False,
        enable_enlarged_vocab: bool = False,
        vocab_strict: bool = False,
    ) -> None:
        super().__init__()
        self.pdb_root = pdb_root
        self.rec_chain = rec_chain
        self.lig_chain = lig_chain
        self.backbone_only = backbone_only
        self.padding_collate = padding_collate
        self.use_covariance_matrix = use_covariance_matrix
        self.return_supervision = return_supervision
        self.return_sample_id = return_sample_id
        self.pocket_json_dir = pocket_json_dir
        self.pocket_json_suffix = pocket_json_suffix
        self.require_pocket = require_pocket
        self._pocket_cache = {}
        self.strict_context = strict_context
        self.enable_enlarged_vocab = enable_enlarged_vocab
        self.vocab_strict = vocab_strict

        with open(index_path, 'r') as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        self.meta = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.meta[record['sample_id']] = record
        self._lengths = []
        for sample_id in self.sample_ids:
            record = self.meta.get(sample_id, {})
            pep_len = int(record.get('peptide_length', 0)) if record is not None else 0
            pocket_len = 0
            if self.pocket_json_dir is not None:
                pocket_set = self._load_pocket_set(sample_id)
                pocket_len = len(pocket_set) if pocket_set is not None else 0
            self._lengths.append(pep_len + pocket_len)

    def _load_pocket_set(self, sample_id: str):
        if sample_id in self._pocket_cache:
            return self._pocket_cache[sample_id]
        if self.pocket_json_dir is None:
            return None
        pocket_path = os.path.join(self.pocket_json_dir, sample_id + self.pocket_json_suffix)
        if not os.path.exists(pocket_path):
            if self.require_pocket:
                raise FileNotFoundError(f'Missing pocket JSON for {sample_id}: {pocket_path}')
            self._pocket_cache[sample_id] = None
            return None
        with open(pocket_path, 'r') as f:
            pocket_json = json.load(f)
        pocket_set = set()
        for entry in pocket_json:
            if not entry or len(entry) < 2:
                continue
            chain_id = entry[0]
            if chain_id != self.rec_chain:
                continue
            res = entry[1]
            if isinstance(res, (list, tuple)):
                if len(res) == 0:
                    continue
                resseq = res[0]
                icode = res[1] if len(res) > 1 else ' '
            else:
                resseq = res
                icode = ' '
            try:
                resseq = int(resseq)
            except Exception:
                pass
            pocket_set.add((resseq, _normalize_icode(icode)))
        if not pocket_set and self.require_pocket:
            raise ValueError(f'Empty pocket definition for {sample_id}: {pocket_path}')
        self._pocket_cache[sample_id] = pocket_set
        return pocket_set

    def __len__(self):
        return len(self.sample_ids)

    def get_len(self, idx: int):
        return self._lengths[idx]

    def get_type(self, idx: int):
        sample_id = self.sample_ids[idx]
        record = self.meta.get(sample_id, {})
        return record.get('c_star', None)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        if sample_id not in self.meta:
            raise KeyError(f'Metadata missing for {sample_id}')
        record = self.meta[sample_id]

        pdb_path = os.path.join(self.pdb_root, sample_id + '.pdb')
        list_blocks = pdb_to_list_blocks(pdb_path, selected_chains=[self.rec_chain, self.lig_chain])
        if len(list_blocks) != 2:
            raise ValueError(f'Expected 2 chains for {sample_id}, got {len(list_blocks)}')
        rec_blocks = [Block.from_tuple(tup.to_tuple()) if isinstance(tup, Block) else Block.from_tuple(tup) for tup in list_blocks[0]]
        lig_blocks = [Block.from_tuple(tup.to_tuple()) if isinstance(tup, Block) else Block.from_tuple(tup) for tup in list_blocks[1]]

        rec_position_ids = [i + 1 for i, _ in enumerate(rec_blocks)]
        pocket_set = self._load_pocket_set(sample_id)
        if pocket_set is not None:
            keep_idx = []
            for i, block in enumerate(rec_blocks):
                resseq, icode = block.id[0], _normalize_icode(block.id[1])
                if (resseq, icode) in pocket_set:
                    keep_idx.append(i)
            if keep_idx:
                rec_blocks = [rec_blocks[i] for i in keep_idx]
                rec_position_ids = [rec_position_ids[i] for i in keep_idx]
            elif self.require_pocket:
                raise ValueError(f'No pocket residues found in {sample_id}')

        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = rec_position_ids + [i + 1 for i, _ in enumerate(lig_blocks)]

        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = self.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)
            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)

        if self.enable_enlarged_vocab or get_cyc_config().get('enable_enlarged_vocab', False):
            S = rewrite_tokens_by_cyclization(S, mask, record, strict=self.vocab_strict)

        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        if self.backbone_only:
            X, atom_mask = X[:, :4], atom_mask[:, :4]
        if self.strict_context:
            ctx_mask = (~mask[:, None]) & atom_mask
            if not bool(ctx_mask.any().item()):
                raise ValueError(f'No context atoms found in {sample_id}')

        if self.use_covariance_matrix:
            cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy())
            eps = 1e-4
            cov = cov + eps * np.identity(cov.shape[0])
            L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)
        else:
            L = None

        item = {
            'X': X,
            'S': torch.tensor(S, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'mask': mask,
            'atom_mask': atom_mask,
            'lengths': len(S),
        }
        if self.return_sample_id:
            item['sample_id'] = sample_id
        if self.return_supervision:
            item['c_star'] = torch.tensor(TYPE2IDX[record['c_star']], dtype=torch.long)
            item['gt_pair'] = torch.tensor(record['gt_pair'], dtype=torch.long)
            item['peptide_length'] = torch.tensor(record.get('peptide_length', len(lig_blocks)), dtype=torch.long)
        if L is not None:
            item['L'] = L
        return item

    def collate_fn(self, batch):
        if self.padding_collate:
            results = {}
            pad_idx = VOCAB.symbol_to_idx(VOCAB.PAD)
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key in ('lengths', 'c_star', 'peptide_length'):
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'sample_id':
                    results[key] = values
                elif key == 'gt_pair':
                    results[key] = torch.stack(values, dim=0)
                elif key == 'S':
                    results[key] = pad_sequence(values, batch_first=True, padding_value=pad_idx)
                else:
                    results[key] = pad_sequence(values, batch_first=True, padding_value=0)
            return results
        if len(batch) == 1:
            return batch[0]
        return default_collate(batch)


@R.register('CPCoreMMapDataset')
class CPCoreMMapDataset(MMAPDataset):
    MAX_N_ATOM = 14

    def __init__(
        self,
        mmap_dir: str,
        jsonl_path: str,
        backbone_only: bool = False,
        padding_collate: bool = False,
        use_covariance_matrix: bool = False,
        return_supervision: bool = True,
        return_sample_id: bool = False,
        specify_data: Optional[str] = None,
        specify_index: Optional[str] = None,
        enable_enlarged_vocab: bool = False,
        vocab_strict: bool = False,
    ) -> None:
        super().__init__(mmap_dir, specify_data=specify_data, specify_index=specify_index)
        self.backbone_only = backbone_only
        self.padding_collate = padding_collate
        self.use_covariance_matrix = use_covariance_matrix
        self.return_supervision = return_supervision
        self.return_sample_id = return_sample_id
        self.enable_enlarged_vocab = enable_enlarged_vocab
        self.vocab_strict = vocab_strict

        self.meta = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.meta[record['sample_id']] = record

        self._lengths = [len(prop[-1].split(',')) + int(prop[1]) for prop in self._properties]

    def get_len(self, idx):
        return self._lengths[idx]

    def get_type(self, idx: int):
        sample_id = self._indexes[idx][0].split('.')[0]
        record = self.meta.get(sample_id, {})
        return record.get('c_star', None)

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        sample_id = self._indexes[idx][0].split('.')[0]
        if sample_id not in self.meta:
            raise KeyError(f'Metadata missing for {sample_id}')
        record = self.meta[sample_id]

        rec_blocks, lig_blocks = data
        pocket_idx = [int(i) for i in self._properties[idx][-1].split(',')]
        rec_position_ids = [i + 1 for i, _ in enumerate(rec_blocks)]

        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        rec_position_ids = [rec_position_ids[i] for i in pocket_idx]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks]

        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = rec_position_ids + [i + 1 for i, _ in enumerate(lig_blocks)]

        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = self.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)
            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)

        if self.enable_enlarged_vocab or get_cyc_config().get('enable_enlarged_vocab', False):
            S = rewrite_tokens_by_cyclization(S, mask, record, strict=self.vocab_strict)

        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        if self.backbone_only:
            X, atom_mask = X[:, :4], atom_mask[:, :4]

        if self.use_covariance_matrix:
            cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy())
            eps = 1e-4
            cov = cov + eps * np.identity(cov.shape[0])
            L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)
        else:
            L = None

        item = {
            'X': X,
            'S': torch.tensor(S, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'mask': mask,
            'atom_mask': atom_mask,
            'lengths': len(S),
        }
        if self.return_sample_id:
            item['sample_id'] = sample_id
        if self.return_supervision:
            item['c_star'] = torch.tensor(TYPE2IDX[record['c_star']], dtype=torch.long)
            item['gt_pair'] = torch.tensor(record['gt_pair'], dtype=torch.long)
            item['peptide_length'] = torch.tensor(record.get('peptide_length', len(lig_blocks)), dtype=torch.long)
        if L is not None:
            item['L'] = L
        return item

    def collate_fn(self, batch):
        if self.padding_collate:
            results = {}
            pad_idx = VOCAB.symbol_to_idx(VOCAB.PAD)
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key in ('lengths', 'c_star', 'peptide_length'):
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'sample_id':
                    results[key] = values
                elif key == 'gt_pair':
                    results[key] = torch.stack(values, dim=0)
                elif key == 'S':
                    results[key] = pad_sequence(values, batch_first=True, padding_value=pad_idx)
                else:
                    results[key] = pad_sequence(values, batch_first=True, padding_value=0)
            return results
        else:
            results = {}
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key == 'lengths':
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key in ('c_star', 'peptide_length'):
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'sample_id':
                    results[key] = values
                elif key == 'gt_pair':
                    results[key] = torch.stack(values, dim=0)
                else:
                    results[key] = torch.cat(values, dim=0)
            return results
