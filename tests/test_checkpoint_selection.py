"""Checkpoint selection and warm-start contracts using small synthetic models."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from train import load_ckpt
from trainer.abs_trainer import Trainer
from trainer.ldm_trainer import LDMTrainer


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, value):
        return self.weight * value.mean()


class SelectionTrainer(Trainer):
    def log(self, *args, **kwargs):
        pass

    def save_checkpoint(self, path, preflight=False):
        Path(path).write_text(str(self.model.weight.item()))


class CheckpointSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_trainer(self, **overrides):
        config = dict(save_dir=str(self.root), max_epoch=1, save_topk=1,
                      patience=3, validate_before_train=False,
                      optimizer={'class': 'SGD', 'lr': .1})
        config.update(overrides)
        return SelectionTrainer(ScalarModel(), DataLoader(torch.ones(2), batch_size=1),
                                [torch.ones(1)], config, {})

    def test_revalidation_keeps_one_existing_checkpoint(self):
        for minimize in [True, False]:
            trainer = self.make_trainer(metric_min_better=minimize)
            Path(trainer.model_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(trainer.model_dir)/'epoch0_step2.ckpt')
            for metric in [2., 2., 1., 3.]:
                Path(path).write_text(str(metric))
                trainer._maintain_topk_checkpoint(metric, path)
                self.assertEqual(trainer.topk_ckpt_map, [(metric, path)])
                self.assertTrue(Path(path).is_file())

    def test_updated_metric_reorders_topk_and_removes_only_evicted_file(self):
        trainer = self.make_trainer(save_topk=2)
        Path(trainer.model_dir).mkdir(parents=True)
        paths = [str(Path(trainer.model_dir)/f'{i}.ckpt') for i in range(3)]
        for path, metric in zip(paths[:2], [1., 2.]):
            Path(path).touch()
            trainer._maintain_topk_checkpoint(metric, path)
        trainer._maintain_topk_checkpoint(4., paths[0])
        self.assertEqual(trainer.topk_ckpt_map, [(2., paths[1]), (4., paths[0])])
        Path(paths[2]).touch()
        trainer._maintain_topk_checkpoint(3., paths[2])
        self.assertEqual(trainer.topk_ckpt_map, [(2., paths[1]), (3., paths[2])])
        self.assertFalse(Path(paths[0]).exists())
        self.assertTrue(all(Path(p).is_file() for _, p in trainer.topk_ckpt_map))

    def test_final_step_validation_counts_once_at_epoch_boundary(self):
        for interval, expected in [(1, 2), (2, 1), (3, 1)]:
            trainer = self.make_trainer(val_every_n_steps=interval)
            trainer.scheduler, trainer.sched_freq = Mock(), 'val_epoch'
            with patch('trainer.abs_trainer.SummaryWriter'):
                trainer.train([-1], -1)
            self.assertEqual(trainer.valid_global_step, expected)
            self.assertEqual(trainer.scheduler.step.call_count, expected)
            self.assertEqual(len(trainer.topk_ckpt_map), 1)
            self.assertTrue(Path(trainer.topk_ckpt_map[0][1]).is_file())
            self.assertEqual(trainer.patience, 3)

    def test_conflicting_warm_start_flags_fail_before_loading(self):
        model = torch.nn.Linear(2, 2)
        with patch('train.torch.load') as reader:
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                load_ckpt(model, {'path': 'unused.pt', 'strict': True, 'skip_mismatch': True})
            reader.assert_not_called()
        path = self.root/'partial.pt'
        bias = model.bias.detach().clone()
        torch.save({'weight': torch.ones(2, 2), 'bias': torch.ones(3)}, path)
        load_ckpt(model, {'path': str(path), 'strict': False, 'skip_mismatch': True})
        self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))
        self.assertTrue(torch.equal(model.bias, bias))
        torch.save(model.state_dict(), path)
        restored = torch.nn.Linear(2, 2)
        load_ckpt(restored, {'path': str(path), 'strict': True, 'skip_mismatch': False})
        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_early_stopping_tracks_the_best_metric_for_both_directions(self):
        for minimize, metrics in [(True, [1., 3., 2.]), (False, [3., 1., 2.])]:
            with self.subTest(minimize=minimize):
                trainer = self.make_trainer(patience=2, metric_min_better=minimize)
                trainer.log = Mock()
                values = iter(metrics)
                trainer.valid_step = lambda batch, step: torch.tensor(next(values))
                patience = []
                for _ in metrics:
                    trainer._valid_epoch(torch.device('cpu'), save_ckpt=False)
                    patience.append(trainer.patience)
                self.assertEqual(patience, [2, 1, 0])
                self.assertEqual(trainer.best_valid_metric, metrics[0])
                self.assertEqual(trainer.last_valid_metric, metrics[-1])
                self.assertEqual(trainer.optimizer_steps, 0)

    def test_early_stop_during_step_validation_stops_the_current_epoch(self):
        trainer = self.make_trainer(patience=1, val_every_n_steps=1)
        trainer.train_loader = DataLoader(torch.ones(3), batch_size=1)
        trainer.log = Mock()
        trainer.optimizer.step = Mock()
        before = trainer.model.weight.detach().clone()
        def exhaust_patience(*args, **kwargs):
            trainer.patience = 0
        trainer._valid_epoch = Mock(side_effect=exhaust_patience)
        trainer._train_epoch(torch.device('cpu'))
        self.assertEqual(trainer.optimizer.step.call_count, 1)
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(trainer._valid_epoch.call_count, 1)
        self.assertTrue(torch.equal(before, trainer.model.weight))

    def test_ldm_selection_defaults_to_loss_and_rejects_other_metrics(self):
        for criterion in ['AAR', 'RMSD', 'unknown']:
            with self.assertRaisesRegex(ValueError, 'criterion=Loss'):
                LDMTrainer(None, None, None, {}, {}, criterion=criterion)
        trainer = LDMTrainer(ScalarModel(), [1], [],
                             {'save_dir': str(self.root), 'max_epoch': 1,
                              'optimizer': {'class': 'SGD', 'lr': .1}}, {})
        trainer.log = Mock()
        trainer.model = Mock(return_value=(torch.tensor(2., requires_grad=True), {}))
        metric = trainer.valid_step({}, 0)
        self.assertEqual(trainer.criterion, 'Loss')
        self.assertEqual(metric.item(), 2.)
        self.assertFalse(metric.requires_grad)


if __name__ == '__main__':
    unittest.main()
