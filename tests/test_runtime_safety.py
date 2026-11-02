"""Optimizer controls use scalar toy models only; no APCyc training is performed."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from trainer.abs_trainer import Trainer, TrainConfig
from apcyc.train_property import _compute_property_stats, load_property_map, checked_property_backward, property_error_sums, property_error_metrics
from pipelines.process_utils import run_isolated
from utils.oom_decorator import OOMReturn


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.weight * x.mean()


class ScalarTrainer(Trainer):
    def log(self, *args, **kwargs):
        pass

    def valid_step(self, batch, batch_idx):
        return batch['value'].mean()


def make_trainer(root, interval=1, data=(1., 3., 5.)):
    model = ScalarModel()
    loader = DataLoader(torch.tensor(data), batch_size=1)
    return ScalarTrainer(model, loader, [],
                         {'save_dir': str(root), 'max_epoch': 1, 'grad_interval': interval,
                          'optimizer': {'class': 'SGD', 'lr': .1}}, {'model': {}})


def distributed_worker(rank, init_path, root):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=2)
    try:
        trainer = make_trainer(root, data=(1.,))
        trainer.local_rank = rank
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model)
        trainer.config.grad_interval = 2
        trainer.train_loader = DataLoader(torch.tensor([1., 3., 5.]) + rank * 2, batch_size=1)
        trainer._train_epoch(torch.device('cpu'))
        healthy_weight, healthy_updates = trainer.model.module.weight.item(), trainer.optimizer_steps
        with torch.no_grad():
            trainer.model.module.weight.fill_(1.)
        trainer.optimizer_steps = 0
        trainer.valid_loader = [{'lengths': torch.ones(1 if rank == 0 else 3),
                                 'value': torch.tensor(2. if rank == 0 else 4.)}]
        metric = trainer._valid_epoch(torch.device('cpu'), save_ckpt=False)
        scheduler = Mock()
        trainer.scheduler, trainer.sched_freq = scheduler, 'val_epoch'
        trainer._valid_epoch(torch.device('cpu'), save_ckpt=False)
        # Only one rank has an invalid forward loss; both must stop before backward.
        trainer.train_step = lambda batch, idx: trainer.model(batch) * (float('nan') if rank else 1.)
        try:
            trainer._train_epoch(torch.device('cpu'))
        except RuntimeError as exc:
            error = str(exc)
        else:
            raise AssertionError('invalid rank must stop all ranks')
        Path(root, f'rank{rank}.json').write_text(json.dumps({
            'healthy_weight': healthy_weight, 'healthy_updates': healthy_updates,
            'metric': metric, 'last_metric': trainer.last_valid_metric,
            'scheduler_metric': scheduler.step.call_args.args[0],
            'patience': trainer.patience, 'updates': trainer.optimizer_steps,
            'weight': trainer.model.module.weight.item(), 'error': error}))
    finally:
        torch.distributed.destroy_process_group()


class RuntimeSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accumulation_final_partial_window_and_scheduler(self):
        trainer = make_trainer(self.root, interval=2)
        trainer.scheduler, trainer.sched_freq = Mock(), 'batch'
        trainer._train_epoch(torch.device('cpu'))
        self.assertAlmostEqual(trainer.model.weight.item(), .3, places=6)
        self.assertEqual((trainer.global_step, trainer.optimizer_steps), (3, 2))
        self.assertEqual(trainer.scheduler.step.call_count, 2)
        self.assertIsNone(trainer.model.weight.grad)
        for invalid in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                TrainConfig('.', 1, grad_interval=invalid)

    def test_nan_inf_and_forward_oom_never_update(self):
        for bad in [torch.tensor(float('nan')), torch.tensor(float('inf')), OOMReturn(torch.tensor(0.))]:
            trainer = make_trainer(self.root, interval=2, data=(1., 2.))
            trainer.train_step = lambda batch, idx: trainer.model(batch) if idx == 0 else bad
            trainer.optimizer.step = Mock()
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite loss or forward OOM'):
                trainer._train_epoch(torch.device('cpu'))
            trainer.optimizer.step.assert_not_called()
            self.assertEqual(trainer.model.weight.item(), 1.)
            self.assertIsNone(trainer.model.weight.grad)

    def test_backward_oom_and_nonfinite_gradient_never_update(self):
        trainer = make_trainer(self.root, data=(1.,))
        trainer.optimizer.step = Mock()
        with patch('trainer.abs_trainer.safe_backward', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Backward OOM'):
                trainer._train_epoch(torch.device('cpu'))
        trainer.optimizer.step.assert_not_called()
        trainer.model.weight.register_hook(lambda grad: grad * float('inf'))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite gradient'):
            trainer._train_epoch(torch.device('cpu'))
        trainer.optimizer.step.assert_not_called()

    def test_validation_weighting_failure_and_mode_restoration(self):
        trainer = make_trainer(self.root)
        trainer.valid_loader = [{'lengths': torch.ones(3), 'value': torch.tensor(2.)},
                                {'lengths': torch.ones(1), 'value': torch.tensor(10.)}]
        self.assertEqual(trainer._valid_epoch(torch.device('cpu'), save_ckpt=False), 4.)
        self.assertTrue(trainer.model.training)
        trainer.valid_step = lambda *args: OOMReturn(torch.tensor(0.))
        trainer.save_checkpoint = Mock()
        with self.assertRaisesRegex(RuntimeError, 'validation metric or OOM'):
            trainer._valid_epoch(torch.device('cpu'))
        trainer.save_checkpoint.assert_not_called()
        self.assertTrue(trainer.model.training)

    def test_distributed_selection_and_bad_rank(self):
        torch.multiprocessing.spawn(distributed_worker,
                                    args=(str(self.root/'init'), str(self.root)), nprocs=2, join=True)
        a, b = [json.loads((self.root/f'rank{i}.json').read_text()) for i in range(2)]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['healthy_weight'], .1, places=6)
        self.assertEqual(a['healthy_updates'], 2)
        self.assertEqual((a['metric'], a['scheduler_metric'], a['last_metric']), (3.5, 3.5, 3.5))
        self.assertEqual((a['updates'], a['weight'], a['patience']), (0, 1., 2))

    def test_property_single_row_and_duplicate_incomplete_row(self):
        mean, std = _compute_property_stats({'a': [2., 4.]}, ['a'])
        self.assertTrue(torch.equal(mean, torch.tensor([2., 4.])))
        self.assertTrue(torch.isfinite(std).all() and (std > 0).all())
        mean, std = _compute_property_stats({'a': [2.], 'b': [4.]}, ['a', 'b'])
        self.assertEqual((mean.item(), std.item()), (3., 1.))
        path = self.root/'properties.csv'
        path.write_text('id,p\na,\na,2\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate property ID'):
            load_property_map(path, 'id', ['p'])

    def test_property_metrics_ignore_batch_partition(self):
        prediction = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        for chunks in [[5], [4, 1], [2, 2, 1]]:
            absolute, squared, count = torch.zeros(1), torch.zeros(1), 0
            for chunk in prediction.split(chunks):
                a, b = property_error_sums(chunk, torch.zeros_like(chunk))
                absolute += a
                squared += b
                count += len(chunk)
            mae, rmse = property_error_metrics(absolute, squared, count)
            self.assertEqual(mae, [3.])
            self.assertAlmostEqual(rmse[0], math.sqrt(11), places=6)
        with self.assertRaises(ValueError):
            property_error_metrics(torch.tensor([float('nan')]), torch.ones(1), 1)

    def test_property_backward_guards(self):
        model = ScalarModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite property loss'):
            checked_property_backward(model(torch.tensor(float('nan'))), model, optimizer)
        model.weight.register_hook(lambda grad: grad * float('inf'))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite property gradient'):
            checked_property_backward(model(torch.tensor(1.)), model, optimizer)
        self.assertEqual(model.weight.item(), 1.)
        self.assertIsNone(model.weight.grad)

    @unittest.skipUnless(os.name == 'posix' and Path('/proc').exists(), 'Linux process-state check')
    def test_external_children_cleaned_on_timeout_exit_and_success(self):
        for mode in ['timeout', 'failure', 'success']:
            ready = self.root/(mode+'.pid')
            child = ('import os,signal,time;from pathlib import Path;'
                     'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                     f'Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)')
            parent = ('import subprocess,sys,time;from pathlib import Path;'
                      f'subprocess.Popen([sys.executable,"-c",{child!r}]);'
                      f'p=Path({str(ready)!r});\nwhile not p.exists(): time.sleep(.01)\n' +
                      ('time.sleep(60)' if mode == 'timeout' else f'sys.exit({2 if mode == "failure" else 0})'))
            try:
                if mode == 'success':
                    run_isolated([sys.executable, '-c', parent], timeout=3, cleanup_grace=.1)
                else:
                    expected = subprocess.TimeoutExpired if mode == 'timeout' else subprocess.CalledProcessError
                    with self.assertRaises(expected):
                        run_isolated([sys.executable, '-c', parent], timeout=.8 if mode == 'timeout' else 3,
                                     cleanup_grace=.1)
                pid = int(ready.read_text())
                for _ in range(50):
                    status = Path(f'/proc/{pid}/stat')
                    if not status.exists() or status.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.02)
                else:
                    self.fail(f'{mode}: descendant remains alive')
            finally:
                if ready.exists():
                    try:
                        os.kill(int(ready.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
