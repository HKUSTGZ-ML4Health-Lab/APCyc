#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import pi, cos

import torch
from torch_scatter import scatter_mean

from .abs_trainer import Trainer
from utils import register as R
from utils.logger import print_log


@R.register('LDMTrainer')
class LDMTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict, criterion: str='AAR'):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)
        self.criterion = criterion
        assert criterion in ['AAR', 'RMSD', 'Loss'], f'Criterion {criterion} not implemented'
        self.rng_state = None

    ########## Override start ##########

    def train_step(self, batch, batch_idx):
        results = self.model(**batch)
        if self.is_oom_return(results):
            return results
        loss, loss_dict = results

        self.log('Overall/Loss/Train', loss, batch_idx, val=False)

        if 'H' in loss_dict:
            self.log('Seq/Loss_H/Train', loss_dict['H'], batch_idx, val=False)

        if 'X' in loss_dict:
            self.log('Struct/Loss_X/Train', loss_dict['X'], batch_idx, val=False)

        for key, value in loss_dict.items():
            if key in ('H', 'X'):
                continue
            if torch.is_tensor(value):
                self.log(f'{key}/Train', value, batch_idx, val=False)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log('lr', lr, batch_idx, val=False)

        return loss

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    def valid_step(self, batch, batch_idx):
        results = self.model(**batch)
        if self.is_oom_return(results):
            print_log(f'Out of memory in validation forward, skip batch {batch_idx}', level='WARN')
            return results.fake_loss.detach()
        loss, loss_dict = results
        self.log('Overall/Loss/Validation', loss, batch_idx, val=True)
        if 'H' in loss_dict: self.log('Seq/Loss_H/Validation', loss_dict['H'], batch_idx, val=True)
        if 'X' in loss_dict: self.log('Struct/Loss_X/Validation', loss_dict['X'], batch_idx, val=True)
        for key, value in loss_dict.items():
            if key in ('H', 'X'):
                continue
            if torch.is_tensor(value):
                self.log(f'{key}/Validation', value, batch_idx, val=True)

        if self.criterion == 'AAR':
            return loss.detach()
        elif self.criterion == 'RMSD':
            return loss.detach()
        elif self.criterion == 'Loss':
            return loss.detach()
        else:
            raise NotImplementedError(f'Criterion {self.criterion} not implemented')

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########
