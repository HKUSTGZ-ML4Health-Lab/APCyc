"""Behavioral regressions for model geometry, topology and evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import models
from apcyc.diffusion.apcyc_dpm import APCycFullDPM
from apcyc.model import APCycLDMPepDesign
from apcyc.property_features import property_features
from data.format import VOCAB, Atom, Block
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from scripts.inspect_predictions import inspect
from pipelines.adapters.rosetta import read_score_rows
from pipelines.evaluate_paper import evaluate, single_linkage_labels, cramers_v
from utils.checkpoint_utils import save_model_checkpoint, load_model_checkpoint


class PaperFixes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, batches=1):
        n = 7 * batches
        mask = torch.tensor(([False] * 3 + [True] * 4) * batches)
        atom_mask = torch.ones(n, 14, dtype=torch.bool)
        atom_mask[mask, 1:] = False
        return dict(H_0=torch.randn(n, 8), X_0=torch.randn(n, 14, 3),
                    position_embedding=torch.zeros(n, 8), mask_generate=mask,
                    lengths=torch.tensor([7] * batches), atom_embeddings=torch.zeros(n, 14, 4),
                    atom_mask=atom_mask, S=torch.full((n,), VOCAB.symbol_to_idx('A'), dtype=torch.long),
                    position_ids=torch.tensor([1, 2, 3, 1, 2, 3, 4] * batches),
                    L=torch.eye(3).repeat(batches, 1, 1))

    def dpm(self):
        return APCycFullDPM(latent_size=8, hidden_size=16, n_channel=14, num_steps=3,
                            n_layers=1, dropout=0, chemistry_profile='paper',
                            relax_residue_constraints=False, enable_covalent_edges=True,
                            topology_hard_start_t=1).eval()

    def test_pair_geometry_and_gradients_ignore_inactive_channels(self):
        d, a = self.dpm(), self.fixture()
        x = a['X_0'].requires_grad_()
        ids = torch.zeros(7, dtype=torch.long)
        pair = d._pair_forward(a['H_0'], x, a['S'], a['mask_generate'], ids,
                               a['position_ids'], None, atom_mask=a['atom_mask'])
        expected = torch.cdist(x[3:, 0], x[3:, 0])
        self.assertTrue(torch.allclose(pair['pair_dist'][3:, 3:], expected))
        gradient, = torch.autograd.grad(pair['pair_dist'][3, 4], x)
        self.assertGreater(gradient[3:, 0].abs().sum().item(), 0)
        self.assertEqual(gradient[3:, 1:].abs().sum().item(), 0)
        changed = x.detach().clone(); changed[3:, 1:] += 100
        repeated = d._pair_forward(a['H_0'], changed, a['S'], a['mask_generate'], ids,
                                   a['position_ids'], None, atom_mask=a['atom_mask'])
        self.assertTrue(torch.equal(pair['type_logits'], repeated['type_logits']))
        self.assertTrue(torch.equal(pair['s_ij_full'], repeated['s_ij_full']))
        self.assertTrue(torch.equal(pair['pair_dist'][:3, :3], torch.cdist(x[:3, 1], x[:3, 1])))

    def test_hard_edges_have_gradients_and_match_sampling(self):
        d, a = self.dpm(), self.fixture()
        records = []
        def capture(module, args, kwargs):
            records.append((kwargs['edge_type'].detach().clone(), kwargs['hard_pair'].detach().clone()
                            if kwargs['hard_pair'] is not None else None))
        handle = d.injector.register_forward_pre_hook(capture, with_kwargs=True)
        loss, _ = d(**a, t=torch.tensor([1]), type_labels=torch.tensor([0]))
        gradient, = torch.autograd.grad(loss['X'] + loss['H'], d.injector.edge_type_embed.weight)
        self.assertGreater(gradient[2].abs().sum().item(), 0)
        self.assertGreater(gradient[3].abs().sum().item(), 0)
        training = records[-1]
        sample = dict(a); sample['H'] = sample.pop('H_0'); sample['X'] = sample.pop('X_0')
        d.sample(**sample, t_hard=1)
        self.assertTrue(torch.equal(training[0], records[-1][0]))
        self.assertTrue(torch.equal(training[1], records[-1][1]))
        handle.remove()
        with self.assertRaisesRegex(ValueError, 'must match'):
            d.sample(**sample, t_hard=2)

    def test_mixed_batch_times_and_property_feature_equivalence(self):
        d, a = self.dpm(), self.fixture(batches=2)
        ids = d._get_batch_ids(a['mask_generate'], a['lengths'])
        x, centers = d._normalize_position(a['X_0'], ids, a['mask_generate'], a['atom_mask'], a['L'])
        pair = d._pair_forward(a['H_0'], a['X_0'], a['S'], a['mask_generate'], ids,
                               a['position_ids'], None, atom_mask=a['atom_mask'])
        features = d.topology_edge_inputs(pair, a['X_0'], a['atom_mask'], a['mask_generate'], ids,
                                          a['lengths'], torch.tensor([1, 3]), training=True)
        edges = torch.cat([features['ctx_edges'], features['inter_edges']], dim=1)
        self.assertTrue(torch.all(ids[edges[0]] == ids[edges[1]]))
        self.assertEqual(set(features['edge_type'][ids[edges[0]] == 1].tolist()), {0, 1})
        self.assertEqual(set(features['edge_type'][ids[edges[0]] == 0].tolist()), {0, 1, 2, 3})
        shared = d.topology_edge_inputs(pair, a['X_0'], a['atom_mask'], a['mask_generate'], ids,
                                        a['lengths'], 1, training=True)
        prop = property_features(d, a['H_0'], x, a['S'], a['mask_generate'], a['lengths'],
                                 a['atom_mask'], a['position_ids'], ids, centers, a['L'], 1, 1,
                                 a['atom_embeddings'])
        self.assertTrue(torch.allclose(prop['edge_attr'], torch.cat([shared['ctx_edge_attr'], shared['inter_edge_attr']])))
        self.assertTrue(torch.equal(prop['edge_type'], shared['edge_type']))

    def test_clean_time_is_rejected_and_default_draw_excludes_zero(self):
        d, a = self.dpm(), self.fixture()
        with self.assertRaisesRegex(ValueError, '1..T'):
            d(**a, t=torch.tensor([0]))
        with patch('torch.randint', wraps=torch.randint) as draw:
            d(**a)
        self.assertEqual(draw.call_args.args[:2], (1, 4))

    def model(self):
        ae = dict(embed_size=16, hidden_size=16, latent_size=8, n_channel=14,
                  n_layers=1, dropout=0, latent_n_channel=1)
        config = dict(autoencoder_ckpt=ae, autoencoder_no_randomness=True, hidden_size=16,
                      num_steps=3, n_layers=1,
                      pair_opt=dict(chemistry_profile='paper', relax_residue_constraints=False,
                                    enable_covalent_edges=True, topology_hard_start_t=1))
        m = APCycLDMPepDesign(**config).eval()
        with torch.no_grad():
            m.autoencoder.seq_linear.weight.zero_(); m.autoencoder.seq_linear.bias.fill_(-10)
            m.autoencoder.seq_linear.bias[0] = 10
            m.autoencoder.seq_linear.bias[int(m.autoencoder.s_map[VOCAB.symbol_to_idx('C')])] = 9
            m.diffusion.cycle_head.type_mlp[-1].weight.zero_()
            m.diffusion.cycle_head.type_mlp[-1].bias.copy_(torch.tensor([0., 10., -10.]))
        return m, config

    def test_final_sequence_mask_and_iterative_refinement_agree(self):
        m, _ = self.model(); a = self.fixture()
        inputs = dict(X=a['X_0'], S=a['S'], mask=a['mask_generate'], position_ids=a['position_ids'],
                      lengths=a['lengths'], atom_mask=a['atom_mask'], L=a['L'])
        kwargs = dict(sample_opt={'t_hard': 1}, return_pair=True, optimize_sidechain=False)
        _, seq, _, pair = m.sample(**inputs, **kwargs)
        self.assertEqual(seq[0], ['C'] * 4)
        self.assertEqual(int(pair['type_hat'][0]), 1)
        # The final decoded sequence determines eligible cyclization chemistry.
        original = m.autoencoder.test
        def refined(*args, **kw):
            x, s, p = original(*args, **kw)
            return x, torch.full_like(s, VOCAB.symbol_to_idx('A')), p
        with patch.object(m.autoencoder, 'test', side_effect=refined):
            _, seq, _, pair = m.sample(**inputs, **{**kwargs, 'sample_opt': {'t_hard': 1, 'autoencoder_n_iter': 2}})
        self.assertEqual(seq[0], ['A'] * 4)
        self.assertEqual(int(pair['type_hat'][0]), 0)

    def test_topology_schedule_survives_checkpoint_reload(self):
        m, cfg = self.model(); path = self.root / 'model.pt'
        save_model_checkpoint(m, {'model': {'class': 'APCycLDMPepDesign', **cfg}}, path)
        restored, _ = load_model_checkpoint(path)
        self.assertEqual(restored.diffusion.topology_hard_start_t, 1)
        for key, value in m.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_public_inspector_uses_paper_iso(self):
        for partner, expected in [('ASN', True), ('ASP', True), ('GLU', False)]:
            directory = self.root / partner; directory.mkdir()
            names = ['LYS', 'ALA', partner]
            blocks = [Block(r, [Atom(n, [i * 3. + j * .2, 0., 0.], n[0]) for j, n in enumerate(['N', 'CA', 'C', 'O'])],
                            id=(i + 1, ' ')) for i, r in enumerate(names)]
            list_blocks_to_pdb([blocks], ['L'], str(directory / 'fixture.pdb'))
            row = dict(pdb='fixture.pdb', peptide_chain='L', pep_len=3, pair_hat=[0, 2],
                       pair_hat_resnames=['LYS', partner], type_hat='ISO', sequence='test')
            (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            self.assertEqual(inspect(directory)['files'][0]['pair_residue_compatibility'], expected)

    def test_rosetta_retains_total_score_and_does_not_substitute_dg(self):
        p = self.root / 'scores.jsonl'
        p.write_text(json.dumps({'decoy': 'sample_0001', 'dg': -4.2, 'total_score': -700}) + '\n')
        self.assertEqual(read_score_rows(p, {'sample'}), [{'id': 'sample', 'rosetta_dG': -4.2, 'rosetta_total_score': -700.}])
        p.write_text(json.dumps({'decoy': 'sample_0001', 'dg': -4.2}) + '\n')
        self.assertIsNone(read_score_rows(p, {'sample'})[0]['rosetta_total_score'])

    def test_evaluation_filters_then_averages_by_receptor(self):
        rows = [dict(id='a', receptor_id='r1', cyclization_distance=3, rosetta_dG=-1, rosetta_total_score=0),
                dict(id='b', receptor_id='r1', cyclization_distance=8, rosetta_dG=-1, rosetta_total_score=20),
                dict(id='c', receptor_id='r1', cyclization_distance=8.1, rosetta_dG=-1, rosetta_total_score=900),
                dict(id='d', receptor_id='r2', cyclization_distance=4, rosetta_dG=-1, rosetta_total_score=100)]
        conventions = {'cyclization_distance': 'supplied test distances', 'rosetta_total_score': 'synthetic values for aggregation test'}
        report = evaluate(rows, conventions, metrics=['rosetta_total_score'])
        self.assertEqual(report['aggregate']['rosetta_total_score'], 55.)
        self.assertEqual(report['receptor_reports']['r1']['valid_ids'], ['a', 'b'])
        self.assertAlmostEqual(report['aggregate']['cyclization_success'], 5 / 6)
        rows[-1]['rosetta_dG'] = 0
        report = evaluate(rows, conventions, metrics=['rosetta_total_score'])
        self.assertEqual(report['aggregate']['rosetta_total_score'], 10.)
        self.assertEqual(report['coverage']['rosetta_total_score']['eligible_receptors'], 1)
        self.assertEqual(report['coverage']['cyclization_success']['eligible_receptors'], 2)

    def test_rosetta_total_requires_protocol_metadata(self):
        rows = [dict(id='a', receptor_id='r', cyclization_distance=4, rosetta_dG=-1, rosetta_total_score=0)]
        report = evaluate(rows, {'cyclization_distance': 'supplied'}, metrics=['rosetta_total_score'])
        self.assertIsNone(report['aggregate']['rosetta_total_score'])
        self.assertTrue(any('scoring protocol metadata required' in item for item in report['errors']))

    def test_single_linkage_transitivity_and_cramers_v(self):
        distance = np.array([[0, .3, .6], [.3, 0, .3], [.6, .3, 0]])
        self.assertEqual(single_linkage_labels(distance, .4), [0, 0, 0])
        self.assertEqual(cramers_v([0, 0, 1, 1], [4, 4, 9, 9]), 1.)
        self.assertEqual(cramers_v([0, 0, 1, 1], [0, 1, 0, 1]), 0.)
        self.assertIsNone(cramers_v([0], [0]))

    def test_distance_ids_are_joined_and_unknown_conventions_are_rejected(self):
        rows = [dict(id=k, receptor_id='r', cyclization_distance=4, rosetta_dG=-1) for k in 'abc']
        rows[-1]['rosetta_dG'] = 0
        pairwise = dict(format='apcyc.evaluation_distances.v1', receptors={'r': dict(ids=['c', 'b', 'a'],
                            sequence_distance=[[0, .2, .2], [.2, 0, .8], [.2, .8, 0]],
                            ca_rmsd=[[0, 2, 2], [2, 0, 8], [2, 8, 0]])})
        conventions = dict(cyclization_distance='supplied', sequence_distance='supplied', ca_rmsd='supplied')
        result = evaluate(rows, conventions, pairwise, metrics=['rosetta_dG'])
        self.assertEqual(result['aggregate']['D_co'], 1.)
        self.assertEqual(result['aggregate']['cramers_v'], 1.)
        with self.assertRaisesRegex(ValueError, 'convention'):
            evaluate(rows, {'cyclization_distance': 'supplied'}, pairwise)
        pairwise['receptors']['r']['ids'] = ['a', 'b', 'bad']
        with self.assertRaisesRegex(ValueError, 'IDs'):
            evaluate(rows, conventions, pairwise)


if __name__ == '__main__':
    unittest.main()
