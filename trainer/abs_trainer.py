#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
import yaml
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.oom_decorator import OOMReturn, safe_backward
from utils.logger import print_log

########### Import your packages below ##########


class TrainConfig:
    def __init__(self, save_dir, max_epoch, warmup=0,
                 metric_min_better=True, patience=3,
                 grad_clip=None, save_topk=-1,  # -1 for save all
                 grad_interval=1,  # parameter update interval
                 val_freq=1,       # frequence for validation
                 **kwargs):
        self.save_dir = save_dir#模型日志与保存目录
        self.max_epoch = max_epoch#最大训练轮数
        self.warmup = warmup#学习率预热周期
        self.metric_min_better = metric_min_better#是否越小越好
        self.patience = patience if patience > 0 else max_epoch#早停策略
        self.grad_clip = grad_clip #梯度裁剪阈值
        self.save_topk = save_topk #只保留性能最好的k个checkpoint
        self.grad_interval = grad_interval #多少步更新一次梯度
        self.val_freq = val_freq #验证频率
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        #模型配置与加载
        self.model = model
        self.config = TrainConfig(**config)
        self.save_config = save_config
        #初始化优化器
        self.optimizer = self.get_optimizer()
        #初始化调度器
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        #训练与验证数据加载器
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        #日志与版本控制
        # log
        self.version = self._get_version()
        self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []  # smaller index means better ckpt
        self.patience = self.config.patience

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                #捕获所有的文件夹名
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def is_oom_return(self, value):
        return isinstance(value, OOMReturn)

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        if hasattr(t_iter, 'set_description'):
            t_iter.set_description(f'train epoch {self.epoch}')
        save_every = getattr(self.config, 'save_every_n_steps', 0) or 0
        val_every = getattr(self.config, 'val_every_n_steps', 0) or 0
        grad_norm_every = getattr(self.config, 'grad_norm_every_n_steps', 0) or 0
        throughput_every = getattr(self.config, 'throughput_every_n_steps', 0) or 0
        for batch in t_iter:
            step_start = time.time()
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            if self.is_oom_return(loss):
                print_log(f'Out of memory, local rank {self.local_rank}', level='WARN')
                loss = loss.fake_loss
            elif torch.isnan(loss):
                print_log(f'Loss is nan, local_rank {self.local_rank}', level='WARN')
                loss = sum([p.norm() for p in self.model.parameters() if p.dtype == torch.float]) * 0.0
                self.log('Train/NaN', 1.0, self.global_step, val=False)
            else:
                self.log('Train/NaN', 0.0, self.global_step, val=False)
            self.optimizer.zero_grad()
            backward_ok = safe_backward(loss, self.model)
            if not backward_ok:
                print_log(f'Backward out of memory, skip', level='WARN')
                loss = loss.detach() # manually delete the computing graph
            if self.config.grad_clip is not None:
                ori_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                # recording gradients
                self.log('Grad Norm', ori_grad_norm.cpu(), self.global_step)
            elif grad_norm_every > 0 and (self.global_step + 1) % grad_norm_every == 0:
                grad_norm = self._grad_norm()
                if grad_norm is not None:
                    self.log('Grad Norm', grad_norm, self.global_step)
            self.optimizer.step()
            step_time = time.time() - step_start
            if throughput_every > 0 and (self.global_step + 1) % throughput_every == 0:
                batch_size, token_count = self._batch_stats(batch)
                if batch_size is not None and step_time > 0:
                    self.log('Throughput/Samples_per_sec', batch_size / step_time, self.global_step)
                if token_count is not None and step_time > 0:
                    self.log('Throughput/Tokens_per_sec', token_count / step_time, self.global_step)
            if hasattr(t_iter, 'set_postfix'):
                lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0
                batch_size, token_count = self._batch_stats(batch)
                postfix = {
                    'loss': f'{loss.item():.4g}',
                    'lr': f'{lr:.2e}',
                    'version': self.version,
                }
                if batch_size is not None:
                    postfix['bs'] = batch_size
                if token_count is not None:
                    postfix['tokens'] = token_count
                t_iter.set_postfix(postfix)
            self.global_step += 1
            if save_every > 0 and self._is_main_proc() and self.global_step % save_every == 0:
                self._save_step_checkpoint()
            if val_every > 0 and self.global_step % val_every == 0:
                if self._is_main_proc():
                    print_log(f'validating step {self.global_step} ...')
                self._valid_epoch(device)
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
        self._train_epoch_end(device)
    
    def _train_epoch_end(self, device):
        return

    def _aggregate_val_metric(self, metric_arr):
        return np.mean(metric_arr)

    def _valid_epoch_begin(self, device):
        return

    def _valid_epoch(self, device, save_ckpt: bool = True, update_state: bool = True):
        metric_arr = []
        self.model.eval()
        self._valid_epoch_begin(device)
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            if hasattr(t_iter, 'set_description'):
                t_iter.set_description(f'valid epoch {self.epoch}')
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        
        # judge
        valid_metric = self._aggregate_val_metric(metric_arr)
        if save_ckpt and self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            torch.save(module_to_save, save_path)
            self._maintain_topk_checkpoint(valid_metric, save_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if update_state:
            if self._metric_better(valid_metric):
                self.patience = self.config.patience
            else:
                self.patience -= 1
            if self.sched_freq == 'val_epoch':
                self.scheduler.step(valid_metric)
            self.last_valid_metric = valid_metric
        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            if self._is_main_proc():
                print_log(f'{name}: {value}')
            self.log(name, value, self.epoch)
        self.writer_buffer = {}
        self._valid_epoch_end(device)
        self.model.train()
    
    def _valid_epoch_end(self, device):
        return

    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        # save map
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def _modify_writer(self):
        return

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            self._modify_writer()
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.yaml'), 'w') as fout:
                yaml.safe_dump(self.save_config, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            print_log(f'training on {device_ids}')
        if getattr(self.config, 'validate_before_train', True):
            if self._is_main_proc():
                print_log('running pre-train validation (no checkpoint)')
            self._valid_epoch(device, save_ckpt=False, update_state=False)

        for _ in range(self.config.max_epoch):
            if self._is_main_proc():
                print_log(f'epoch {self.epoch + 1}/{self.config.max_epoch} starts')
            self._train_epoch(device)
            if (self.epoch + 1) % self.config.val_freq == 0:
                if self._is_main_proc():
                    print_log(f'validating epoch {self.epoch + 1} ...')
                self._valid_epoch(device)
            self.epoch += 1
            if self.patience <= 0:
                break

    def _batch_stats(self, batch):
        if isinstance(batch, dict) and 'lengths' in batch:
            lengths = batch['lengths']
            if torch.is_tensor(lengths):
                batch_size = int(lengths.numel())
                token_count = int(lengths.sum().item())
                return batch_size, token_count
            if isinstance(lengths, (list, tuple)):
                return len(lengths), int(sum(lengths))
        return None, None

    def _grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
        if total_norm_sq == 0.0:
            return None
        return total_norm_sq ** 0.5

    def _save_step_checkpoint(self):
        save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
        module_to_save = self.model.module if self.local_rank == 0 else self.model
        torch.save(module_to_save, save_path)
        print_log(f'Step checkpoint saved: {save_path}')

    def log(self, name, value, step, val=False, batch_size=1):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].extend([value] * batch_size)
            else:
                self.writer.add_scalar(name, value, step)

    # define optimizer
    def get_optimizer(self):
        opt_cfg = deepcopy(self.config.optimizer)
        #此时 cls 变量指向类对象 torch.optim.optimizer_type
        cls = getattr(torch.optim, opt_cfg.pop('class'))
        # optimizer = cls(self.model.parameters(), **opt_cfg)
        #filter(func,interable):会把iterable里的元素一个个传给func，这里从模型的所有参数重需哪一个需要更新的参数
        #cls是优化器类，接受参数例子：torch.optim.Adam(params, lr=1e-3, weight_decay=1e-2)
        optimizer = cls(filter(lambda p: p.requires_grad, self.model.parameters()), **opt_cfg)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        if not hasattr(self.config, 'scheduler'):
            return None
        sched_cfg = deepcopy(self.config.scheduler)
        cls = getattr(torch.optim.lr_scheduler, sched_cfg.pop('class'))
        freq = sched_cfg.pop('frequency')
        return {
            'scheduler': cls(optimizer, **sched_cfg),
            'frequency': freq # batch/epoch/val_epoch
        }

    ########## Overload these functions below ##########
    # train step, note that batch should be dict/list/tuple/instance. Objects with .to(device) attribute will be automatically moved to the same device as the model
    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
