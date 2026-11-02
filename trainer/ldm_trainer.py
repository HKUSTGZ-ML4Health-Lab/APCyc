#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch

from .abs_trainer import Trainer
from utils import register as R


@R.register('LDMTrainer')
class LDMTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict, criterion: str='Loss'):
        if criterion != 'Loss':
            raise ValueError('LDMTrainer supports criterion=Loss only.')
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion
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
        self.cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        if self.cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng_state)
        return super()._valid_epoch_end(device)

    def valid_step(self, batch, batch_idx):
        results = self.model(**batch)
        if self.is_oom_return(results):
            return results
        loss, loss_dict = results
        self.log('Overall/Loss/Validation', loss, batch_idx, val=True)
        if 'H' in loss_dict: self.log('Seq/Loss_H/Validation', loss_dict['H'], batch_idx, val=True)
        if 'X' in loss_dict: self.log('Struct/Loss_X/Validation', loss_dict['X'], batch_idx, val=True)
        for key, value in loss_dict.items():
            if key in ('H', 'X'):
                continue
            if torch.is_tensor(value):
                self.log(f'{key}/Validation', value, batch_idx, val=True)

        return loss.detach()

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########
