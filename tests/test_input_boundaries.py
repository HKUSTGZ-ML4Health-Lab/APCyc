"""Checkpoint identity, complete warm starts and recoverable input failures."""
import copy
import csv
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

import test_property_scorer as scorer_fixtures
import pipelines.collect_properties as collector
import pipelines.postprocess as post

import test_regressions as fixtures
from apcyc.train_property import _load_predictor_checkpoint, _load_separate_checkpoints
from data.format import VOCAB
from pipelines.postprocess import run as postprocess
from pipelines.score_properties import main as score_main
from train import load_ckpt


class InputBoundaries(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReleaseRegressions()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def checkpoint(self, name, keys):
        payload = dict(state_dict={'weight': torch.tensor([1.])},
                       model_cfg={'n_properties': len(keys)}, property_keys=keys,
                       property_mean=[0.] * len(keys), property_std=[1.] * len(keys))
        path = self.root / name
        torch.save(payload, path)
        return path, payload

    def test_predictor_requires_every_requested_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, 'not found'):
            _load_predictor_checkpoint(self.root / 'missing.pt')
        self.checkpoint('affinity.pt', ['affinity'])
        with self.assertRaisesRegex(FileNotFoundError, 'Pproxy'):
            _load_separate_checkpoints(str(self.root), ['affinity', 'Pproxy'])
        self.checkpoint('Pproxy.pt', ['Pproxy'])
        payloads = _load_separate_checkpoints(str(self.root), ['affinity', 'Pproxy'])
        self.assertEqual(list(payloads), ['affinity', 'Pproxy'])
        self.assertEqual(payloads['Pproxy']['property_keys'], ['Pproxy'])

    def test_predictor_statistics_cannot_broadcast_or_be_nonfinite(self):
        path, payload = self.checkpoint('multi.pt', ['affinity', 'Pproxy'])
        self.assertEqual(_load_predictor_checkpoint(path)['property_std'], [1., 1.])
        for key, value in [('property_mean', [0.]), ('property_std', [1.]),
                           ('property_mean', [0., float('nan')]), ('property_std', [1., 0.]),
                           ('property_std', [1., -1.]), ('property_std', None)]:
            invalid = copy.deepcopy(payload)
            invalid[key] = value
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, key):
                _load_predictor_checkpoint(path)

    def test_separate_targets_cannot_share_a_filename(self):
        with self.assertRaisesRegex(ValueError, 'distinct checkpoint filenames'):
            _load_separate_checkpoints(None, ['target/a', 'target:a'])

    def test_warmstart_checks_vocabulary_before_changing_weights(self):
        model = torch.nn.Linear(2, 1)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        new_state = {key: value + 1 for key, value in before.items()}
        vocab = [VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))]
        wrong = list(vocab)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        path = self.root / 'warmstart.pt'
        torch.save({'state_dict': new_state, 'vocabulary': wrong}, path)
        with self.assertRaisesRegex(ValueError, 'vocabulary differs'):
            load_ckpt(model, path)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        torch.save({'state_dict': new_state, 'vocabulary': vocab}, path)
        load_ckpt(model, path)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in new_state.items()))

    def test_missing_pdb_records_failure_and_continues(self):
        source = self.fixture.structure()
        base = dict(sequence='AAAA', peptide_chain='L', pair_hat=[0, 3], pair_index_base=0, type_hat='HT')
        manifest = self.root / 'predictions.jsonl'
        manifest.write_text(''.join(json.dumps({**base, 'pdb': name}) + '\n'
                                    for name in ['missing.pdb', source.name]))
        out = self.root / 'processed'
        args = SimpleNamespace(predictions=manifest, out_dir=out, relax_python=None,
                               cpsea_relax_dir=None, timeout=1, strict=True)
        with self.assertRaises(SystemExit) as error:
            postprocess(args)
        self.assertEqual(error.exception.code, 1)
        report = json.loads((out / 'status.json').read_text())
        self.assertEqual((report['samples'], report['failed']), (2, 1))
        self.assertIn('FileNotFoundError', report['records'][0]['reason'])
        self.assertEqual(report['records'][1]['status'], 'topology_eligible_not_relaxed')
        self.assertEqual((out / 'predictions.jsonl').read_text(), '')

    def test_score_cli_preserves_na_identifiers_and_sequences(self):
        source = self.root / 'source.csv'
        source.write_text('id,sequence,logP\nNA,NA,1\nNULL,AA,2\n')
        ids = self.root / 'ids.txt'; ids.write_text('NA\nNULL\n')
        output = self.root / 'scored.csv'; normalizer = self.root / 'normalizer.json'
        argv = ['score_properties', '--input', str(source), '--output', str(output),
                '--fit-ids', str(ids), '--save-normalizer', str(normalizer)]
        with patch.object(sys, 'argv', argv):
            score_main()
        with output.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row['id'], row['sequence']) for row in rows], [('NA', 'NA'), ('NULL', 'AA')])
        self.assertEqual(json.loads(normalizer.read_text())['fit_ids'], ['NA', 'NULL'])


    def test_postprocess_complete_structure_and_portable_pocket_reference(self):
        lines, serial = [], 1
        def atom(name, chain, index, xyz):
            nonlocal serial
            line = (f'ATOM  {serial:5d} {name:^4s} GLY {chain}{index:4d}    '
                    f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.:6.2f}{20.:6.2f}          {name[0]:>2s}\n')
            serial += 1
            return line
        for index in range(3):
            for name, dx in [('N', 0), ('CA', 1.4), ('C', 2.8), ('O', 4.)]:
                lines.append(atom(name, 'R', index + 1, [index * 6 + dx, 20, 0]))
        lines.append('TER\n')
        coordinates = [[[0,0,0],[1.45,0,0],[2.9,0,0],[3.5,1,0]],
                       [[5,0,0],[6.45,0,0],[7.9,0,0],[8.5,1,0]],
                       [[7.9,4,0],[6.45,4,0],[5,4,0],[4.4,5,0]],
                       [[2.9,4,0],[1.45,4,0],[0,1.35,0],[-1.2,1.35,0]]]
        for index, coords in enumerate(coordinates):
            for name, xyz in zip(['N', 'CA', 'C', 'O'], coords):
                lines.append(atom(name, 'L', index + 1, xyz))
        lines += ['CONECT   27   13\n', 'END\n']
        source = self.root / 'closed.pdb'; source.write_text(''.join(lines))
        pocket = self.root / 'pocket.json'; pocket.write_text('[]')
        row = dict(pdb=source.name, sequence='GGGG', peptide_chain='L', type_hat='HT',
                   pair_hat=[0, 3], pair_index_base=0, pocket_json=pocket.name)
        _, residues = post.peptide(source, 'L'); spec = post.closure_spec(row, residues)
        self.assertTrue(post.validate_output(source, source, row, spec)['chemical_closure_validated'])
        incomplete = self.root / 'incomplete.pdb'
        incomplete.write_text(''.join(line for line in lines if not (
            line.startswith('ATOM') and line[21] == 'L' and int(line[22:26]) == 2
            and line[12:16].strip() == 'O')))
        with self.assertRaisesRegex(ValueError, 'incomplete_peptide_backbone'):
            post.validate_output(source, incomplete, row, spec)
        multiple = self.root / 'multiple.pdb'
        body = ''.join(lines[:-2])
        multiple.write_text('MODEL        1\n' + body + 'ENDMDL\nMODEL        2\n' + body + 'ENDMDL\nEND\n')
        with self.assertRaisesRegex(ValueError, 'exactly_one'):
            post.peptide(multiple, 'L')
        manifest = self.root / 'closed.jsonl'; manifest.write_text(json.dumps(row) + '\n')
        def copy_result(argv, **kwargs):
            request = json.loads(Path(argv[argv.index('--request') + 1]).read_text())
            shutil.copyfile(request['source'], request['output'])
        args = SimpleNamespace(predictions=manifest, out_dir=self.root/'relaxed',
            relax_python=Path(sys.executable), cpsea_relax_dir=self.root, timeout=1, strict=True)
        with patch.object(post, 'run_isolated', side_effect=copy_result):
            post.run(args)
        accepted = json.loads((args.out_dir/'predictions.jsonl').read_text())
        self.assertEqual(Path(accepted['pocket_json']), pocket.resolve())
        self.assertTrue(Path(accepted['pocket_json']).is_file())

    def test_collector_rejects_invalid_ids_sequences_and_incomplete_backbones(self):
        source = self.fixture.structure()
        manifest = self.root/'manifest.jsonl'
        for sid in [0, '', None, 'a/b']:
            manifest.write_text(json.dumps(dict(id=sid, pdb=str(source))) + '\n')
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                collector.load_inputs(manifest)
        imported = self.root/'imported.csv'; imported.write_text('id,score\na,5\n')
        config = self.root/'config.yaml'
        config.write_text(yaml.safe_dump({'stages': {'probe': {'import': str(imported), 'columns': {'value': 'score'}}}}))
        incomplete = self.root/'partial.pdb'
        incomplete.write_text(''.join(line for line in source.read_text().splitlines(keepends=True)
            if not (line.startswith('ATOM') and line[21] == 'L' and int(line[22:26]) == 11
                    and line[12:16].strip() == 'C')))
        for index, (pdb, sequence) in enumerate([(source, 'GGGG'), (incomplete, 'AAAA')]):
            manifest.write_text(json.dumps(dict(id='a', pdb=str(pdb), sequence=sequence)) + '\n')
            out = self.root/f'failed-{index}'
            args = SimpleNamespace(manifest=manifest, config=config, out_dir=out,
                                   only=None, run_external=False, strict=True)
            with self.assertRaises(SystemExit):
                collector.collect(args)
            result = next(csv.DictReader((out/'raw_properties.csv').open()))
            self.assertEqual((result['sequence'], result['value']), ('', ''))
            self.assertEqual((out/'sequences.fasta').read_text(), '')
            self.assertTrue(all(not (out/folder/'a.pdb').exists() for folder in ['inputs', 'peptides']))
            self.assertFalse((out/'structure_zips/a.zip').exists())
            status = json.loads((out/'status.json').read_text())
            self.assertEqual(status['stages']['probe']['missing_ids'], ['a'])
        records = [dict(id='a', pdb=source, chain='L')]
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'separate'):
            collector.prepare(records, self.root)
        self.assertEqual(source.read_bytes(), original)
        rows, failures = collector.prepare(records, self.root/'valid')
        self.assertEqual((rows[0]['sequence'], failures), ('AAAA', []))

    def test_scoring_artifact_roles_cannot_alias(self):
        source = self.root/'raw.csv'; source.write_text('id,logP\na,1\n')
        ids = self.root/'ids.txt'; ids.write_text('a\n')
        both = self.root/'both.json'
        output = self.root/'output.csv'; norm = self.root/'normalizer.json'
        cases = [(both, both), (source, norm), (output, source), (ids, norm)]
        link = self.root/'hardlink.csv'; os.link(source, link); cases.append((link, norm))
        for target, stats in cases:
            before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
            argv = ['score_properties', '--input', str(source), '--output', str(target),
                    '--fit-ids', str(ids), '--save-normalizer', str(stats)]
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                score_main()
            self.assertTrue(all(p.read_bytes() == body for p, body in before.items()))
            self.assertFalse(norm.exists())
        self.assertFalse(both.exists())

    def test_generator_export_preserves_partial_named_source_and_metadata(self):
        from scripts.models.export_checkpoint import export
        from apcyc.checkpoint import load_bundle
        f = scorer_fixtures.PropertyScorerTests(); f.setUp()
        try:
            target = f.root/'generator.pt'; source = f.root/'generator.pt.partial'
            torch.save(f.generator, source); before = source.read_bytes()
            train = f.root/'train.yaml'
            train.write_text(yaml.safe_dump({'model': {'class': 'APCycLDMPepDesign', **f.cfg}}))
            ae = f.root/'ae.yaml'
            ae.write_text(yaml.safe_dump({'model': {'class': 'AutoEncoder', **f.cfg['autoencoder_ckpt']}}))
            args = SimpleNamespace(legacy=source, out=target, train_config=train, ae_config=ae,
                                   selection='synthetic regression fixture', trust_legacy_pickle=True)
            sidecar = target.with_suffix('.json'); sidecar.write_text('existing metadata')
            with self.assertRaises(FileExistsError):
                export(args)
            self.assertEqual(sidecar.read_text(), 'existing metadata'); self.assertFalse(target.exists())
            sidecar.unlink(); args.out = f.root/'collision.json'
            with self.assertRaisesRegex(ValueError, 'different'):
                export(args)
            args.out = target; export(args)
            self.assertEqual(source.read_bytes(), before)
            restored, _ = load_bundle(target)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in f.generator.state_dict().items()))
            self.assertTrue(json.loads(sidecar.read_text())['tensor_equality'])
            self.assertFalse(list(f.root.glob('.apcyc-export-*')))
        finally:
            f.tearDown()

    def test_autoencoder_export_preserves_partial_named_source(self):
        from scripts.models.export_autoencoder import main
        from utils.checkpoint_utils import load_autoencoder_checkpoint
        f = scorer_fixtures.PropertyScorerTests(); f.setUp()
        try:
            target = f.root/'portable-ae.pt'; source = f.root/'portable-ae.pt.partial'
            torch.save(f.generator.autoencoder, source); before = source.read_bytes()
            config = f.root/'ae.yaml'
            config.write_text(yaml.safe_dump({'model': {'class': 'AutoEncoder', **f.cfg['autoencoder_ckpt']}}))
            argv = ['export_autoencoder', '--checkpoint', str(source), '--out', str(target),
                    '--config', str(config), '--trust-legacy-pickle']
            with patch.object(sys, 'argv', argv):
                main()
            self.assertEqual(source.read_bytes(), before)
            restored = load_autoencoder_checkpoint(target)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in f.generator.autoencoder.state_dict().items()))
            with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(list(f.root.glob('.apcyc-ae-export-*')))
        finally:
            f.tearDown()

    def pdb_dataset(self, record=None):
        from apcyc.dataset import CPCoreDataset
        source = self.fixture.structure()
        index = self.root/'pdb-index.txt'; index.write_text('structure\n')
        metadata = self.root/'metadata.jsonl'
        record = record or dict(sample_id='structure', peptide_length=4, c_star='HT', gt_pair=[0, 3])
        metadata.write_text(json.dumps(record) + '\n')
        return CPCoreDataset(str(self.root), str(index), str(metadata), return_sample_id=True)

    def test_pdb_batches_use_flat_tensors_and_actual_complexity(self):
        dataset = self.pdb_dataset()
        item = dataset[0]
        self.assertEqual(dataset.get_len(0), 9)
        single = dataset.collate_fn([item])
        self.assertEqual(single['lengths'].tolist(), [9])
        self.assertEqual(single['X'].shape, (9, 14, 3))
        shorter = {key: value for key, value in item.items()}
        for key in ['X', 'S', 'mask', 'position_ids', 'atom_mask']:
            shorter[key] = shorter[key][:-1]
        shorter.update(lengths=8, peptide_length=torch.tensor(3), gt_pair=torch.tensor([0, 2]), sample_id='short')
        batch = dataset.collate_fn([item, shorter])
        self.assertEqual(batch['lengths'].tolist(), [9, 8])
        self.assertEqual(batch['X'].shape, (17, 14, 3))
        self.assertEqual(batch['gt_pair'].tolist(), [[0, 3], [0, 2]])
        self.assertEqual(batch['sample_id'], ['structure', 'short'])

    def test_dataset_rejects_duplicate_metadata_and_invalid_supervision(self):
        from apcyc.dataset import CPCoreDataset
        base = dict(sample_id='structure', peptide_length=4, c_star='HT', gt_pair=[0, 3])
        for changes in [dict(gt_pair=[-1, 3]), dict(gt_pair=[0, 4]), dict(gt_pair=[0, 0]),
                        dict(gt_pair=[False, 3]), dict(c_star='unknown'), dict(peptide_length=3)]:
            dataset = self.pdb_dataset({**base, **changes})
            with self.assertRaisesRegex(ValueError, 'supervision|peptide_length'):
                dataset[0]
        metadata = self.root/'metadata.jsonl'
        metadata.write_text(json.dumps(base) + '\n' + json.dumps(base) + '\n')
        with self.assertRaisesRegex(ValueError, 'unique'):
            CPCoreDataset(str(self.root), str(self.root/'pdb-index.txt'), str(metadata))
        metadata.write_text(json.dumps(base) + '\n')
        index = self.root/'pdb-index.txt'; index.write_text('structure\nstructure\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            CPCoreDataset(str(self.root), str(index), str(metadata))

    def test_mmap_ids_preserve_dots_through_loading_and_property_filtering(self):
        from apcyc.dataset import CPCoreMMapDataset
        from apcyc.train_property import filter_mmap_dataset, _dataset_sample_ids
        from data.mmap_dataset import create_mmap
        from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
        source = self.fixture.structure()
        chains = pdb_to_list_blocks(str(source), ['R', 'L'])
        mmap = self.root/'mmap'
        create_mmap(iter([('sample.v1', [[block.to_tuple() for block in chain] for chain in chains],
                           [5, 4, 'AAAA', '0,1,2,3,4'], 1)]), str(mmap))
        metadata = self.root/'metadata.jsonl'
        metadata.write_text(json.dumps(dict(sample_id='sample.v1', peptide_length=4, c_star='HT', gt_pair=[0, 3])) + '\n')
        dataset = CPCoreMMapDataset(str(mmap), str(metadata), return_sample_id=True)
        self.assertEqual(dataset[0]['sample_id'], 'sample.v1')
        self.assertEqual(dataset.get_type(0), 'HT')
        self.assertEqual(_dataset_sample_ids(dataset), ['sample.v1'])
        filter_mmap_dataset(dataset, {'sample.v1'})
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.collate_fn([dataset[0]])['lengths'].tolist(), [9])
        del dataset

if __name__ == '__main__':
    unittest.main()
