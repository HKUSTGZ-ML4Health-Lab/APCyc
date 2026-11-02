"""Training boundary regressions; no optimizer updates or private assets required."""
import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import models
from apcyc.model import APCycLDMPepDesign, _load_autoencoder
from apcyc.checkpoint import load_bundle
from apcyc.property_schema import DEFAULT_PROPERTY_KEYS
from apcyc.train_property import parse, _apply_train_config, infer_property_keys, load_property_map
from data import create_dataloader
from data.dataset_wrapper import DynamicBatchWrapper, EpochDistributedSampler
from utils.checkpoint_utils import save_model_checkpoint, load_model_checkpoint
from utils.config_utils import overwrite_values


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_per_device_and_global_batch_contracts(self):
        ds = torch.utils.data.TensorDataset(torch.arange(8))
        original = EpochDistributedSampler
        def sampler(data, shuffle):
            return original(data, num_replicas=2, rank=0, shuffle=shuffle)
        with patch('data.EpochDistributedSampler', side_effect=sampler):
            per_device = create_dataloader(ds, {'batch_size': 1, 'batch_size_scope': 'per_device', 'num_workers': 0}, 2)
            self.assertEqual(per_device.batch_size, 1)
            self.assertEqual(len(list(per_device)), 4)
            global_loader = create_dataloader(ds, {'batch_size': 4, 'batch_size_scope': 'global', 'num_workers': 0}, 2)
            self.assertEqual(global_loader.batch_size, 2)
            for invalid in [1, 3, 0, -1, True, 1.5]:
                with self.assertRaises(ValueError):
                    create_dataloader(ds, {'batch_size': invalid, 'num_workers': 0}, 2)

    def test_ae_to_ldm_to_inference_checkpoint_chain(self):
        ae_cfg = dict(class_='AutoEncoder', embed_size=16, hidden_size=16, latent_size=8,
                      n_channel=14, latent_n_channel=1, n_layers=1)
        ae_cfg['class'] = ae_cfg.pop('class_')
        ae = models.AutoEncoder(**{k: v for k, v in ae_cfg.items() if k != 'class'})
        ae_path = self.root / 'ae.ckpt'
        save_model_checkpoint(ae, {'model': ae_cfg}, ae_path)
        restored = _load_autoencoder(ae_path, True)
        for key, value in ae.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        cfg = dict(autoencoder_ckpt=str(ae_path), autoencoder_no_randomness=True,
                   hidden_size=16, num_steps=10, n_layers=1, dist_rbf=2, n_rbf=2)
        ldm = APCycLDMPepDesign(**cfg)
        ldm.set_consec_dist(3.2, .4)
        ldm_path = self.root / 'ldm.ckpt'
        save_model_checkpoint(ldm, {'model': {'class': 'APCycLDMPepDesign', **cfg}}, ldm_path)
        # The saved LDM is self-contained; its original AE file is no longer needed.
        ae_path.unlink()
        loaded, _ = load_bundle(ldm_path)
        self.assertEqual((loaded.consec_dist_mean, loaded.consec_dist_std), (3.2, .4))
        for key, value in ldm.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        extracted = _load_autoencoder(ldm_path, True)
        for key, value in ae.state_dict().items():
            self.assertTrue(torch.equal(value, extracted.state_dict()[key]), key)

    def test_training_checkpoint_requires_exact_state(self):
        cfg = dict(embed_size=16, hidden_size=16, latent_size=8, n_channel=14, n_layers=1)
        model = models.AutoEncoder(**cfg)
        path = self.root / 'model.ckpt'
        save_model_checkpoint(model, {'model': {'class': 'AutoEncoder', **cfg}}, path)
        payload = torch.load(path, weights_only=True)
        payload['state_dict'].pop(next(iter(payload['state_dict'])))
        torch.save(payload, path)
        with self.assertRaises(RuntimeError):
            load_model_checkpoint(path)

    def test_property_targets_exclude_auxiliary_columns(self):
        path = self.root / 'scored.csv'
        header = ['id', 'sequence', 'rosetta_dG', *reversed(DEFAULT_PROPERTY_KEYS), 'vina_score']
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerow(['sample', 'ACDE', -3, *range(7), -4])
        self.assertEqual(infer_property_keys(path, 'id'), list(DEFAULT_PROPERTY_KEYS))
        self.assertEqual(infer_property_keys(path, 'id', ['vina_score']), ['vina_score'])
        with self.assertRaises(ValueError):
            infer_property_keys(path, 'id', ['missing'])
        with self.assertRaises(ValueError):
            infer_property_keys(path, 'id', ['affinity', 'affinity'])

    def test_explicit_cli_default_values_win_over_yaml(self):
        base = ['--train_index', 'train.txt', '--valid_index', 'valid.txt']
        args = parse(base + ['--epochs=10', '--lr', '0.0001', '--batch_size', '1', '--use_covariance_matrix', 'false'])
        _apply_train_config(args, {'epochs': 9999, 'lr': .2, 'batch_size': 8, 'use_covariance_matrix': True, 'num_workers': 0})
        self.assertEqual((args.epochs, args.lr, args.batch_size, args.use_covariance_matrix, args.num_workers),
                         (10, .0001, 1, 'false', 0))
        implicit = parse(base)
        _apply_train_config(implicit, {'epochs': 9999, 'property_keys': ['affinity']})
        self.assertEqual(implicit.epochs, 9999)
        self.assertEqual(implicit.property_keys, 'affinity')
        with self.assertRaises(ValueError):
            _apply_train_config(implicit, {'epochz': 2})

    def test_typed_overrides_preserve_false_negative_and_null(self):
        config = {'model': {'enabled': True, 'weight': 1., 'count': 2, 'path': None}, 'load_ckpt': None}
        actual = overwrite_values(deepcopy(config), ['--model.enabled=false', '--model.weight', '-0.5',
                                                    '--model.count', '-1', '--model.path', 'weights/ae.pt'])
        self.assertIs(actual['model']['enabled'], False)
        self.assertEqual(actual['model']['weight'], -.5)
        self.assertEqual(actual['model']['count'], -1)
        self.assertEqual(actual['model']['path'], 'weights/ae.pt')
        for args in [['--model.count'], ['--model.count', '1.5'], ['--model.enabled', 'nope'], ['--unknown', '1']]:
            with self.assertRaises((ValueError, KeyError)):
                overwrite_values(deepcopy(config), args)


    def test_dynamic_packing_preserves_epoch_samples_and_rejects_oversized_items(self):
        class Lengths:
            def __len__(self): return 4
            def get_len(self, index): return [6, 6, 4, 4][index]
            def __getitem__(self, index): return index
        counts = set()
        for seed in range(6):
            wrapped = DynamicBatchWrapper(Lengths(), 'n', 10, type_seed=seed)
            for _ in range(4):
                self.assertEqual(sorted(i for batch in wrapped.batch_indexes for i in batch), [0, 1, 2, 3])
                self.assertTrue(all(batch and sum(Lengths().get_len(i) for i in batch) <= 10
                                    for batch in wrapped.batch_indexes))
                counts.add(len(wrapped)); wrapped.update_epoch()
        self.assertEqual(counts, {2, 3})
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            DynamicBatchWrapper(Lengths(), 'n', 5)
        with self.assertRaisesRegex(ValueError, 'complexity'):
            DynamicBatchWrapper(Lengths(), '-n', 10)

    def test_dynamic_packing_is_rank_independent_and_sampler_refreshes_sizes(self):
        import numpy as np
        class Lengths:
            def __len__(self): return 4
            def get_len(self, index): return [6, 6, 4, 4][index]
            def __getitem__(self, index): return index
        np.random.seed(1)
        first = DynamicBatchWrapper(Lengths(), 'n', 10, type_seed=2)
        np.random.seed(991)
        second = DynamicBatchWrapper(Lengths(), 'n', 10, type_seed=2)
        self.assertEqual(first.batch_indexes, second.batch_indexes)
        samplers = [EpochDistributedSampler(first, num_replicas=2, rank=rank, shuffle=False) for rank in range(2)]
        sizes = set()
        for epoch in range(4):
            sizes.add(len(first))
            partitions = [list(sampler) for sampler in samplers]
            self.assertEqual(set(i for part in partitions for i in part), set(range(len(first))))
            self.assertTrue(all(len(part) == len(sampler) for part, sampler in zip(partitions, samplers)))
            self.assertTrue(all(sampler.total_size == 2 * len(sampler) for sampler in samplers))
            first.update_epoch(); np.random.rand(17); second.update_epoch()
            self.assertEqual(first.batch_indexes, second.batch_indexes)
        self.assertEqual(sizes, {2, 3})

if __name__ == '__main__':
    unittest.main()
