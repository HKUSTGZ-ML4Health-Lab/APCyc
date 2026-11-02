"""Independent AE scoring and decoded guidance without parameter updates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from apcyc.checkpoint import sha256
from apcyc.model import APCycLDMPepDesign
from apcyc.property_predictor import PropertyPredictor
from apcyc.property_features import feature_signature
from apcyc.property_scorer import export_scorer, load_scorer, decoded_scores, FrozenPropertyScorer
from apcyc.sample import sampling_options
from utils.checkpoint_utils import save_model_checkpoint


class PropertyScorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(autoencoder_ckpt=dict(embed_size=16, hidden_size=16, latent_size=8,
                        n_channel=14, latent_n_channel=1, n_layers=1),
                        autoencoder_no_randomness=True, hidden_size=16, num_steps=6,
                        n_layers=1, dist_rbf=2, n_rbf=2)
        torch.manual_seed(31)
        self.generator = APCycLDMPepDesign(**self.cfg).eval()
        ae = self.root / 'ae.pt'
        save_model_checkpoint(self.generator.autoencoder,
                              {'model': {'class': 'AutoEncoder', **self.cfg['autoencoder_ckpt']}}, ae)
        payload = torch.load(ae, weights_only=True)
        payload['provenance'] = {'source_sha256': 'original-ae'}
        torch.save(payload, ae)
        head_cfg = dict(latent_size=8, hidden_size=16, n_channel=14, n_layers=1,
                        edge_size=2 + self.generator.diffusion.pair_edge_dim, n_rbf=2,
                        n_properties=1, use_attn_pool=False, use_pair_stats=False)
        heads = {}
        for key in ['affinity', 'Sfinal']:
            head = PropertyPredictor(**head_cfg)
            path = self.root / (key + '.pt')
            torch.save(dict(model_cfg=head_cfg, state_dict=head.state_dict(), property_keys=[key],
                            property_mean=[2.], property_std=[3.]), path)
            heads[key] = path
        context = dict(referenced_ae={'sha256': 'original-ae'}, optimizer_steps=0,
                       pair_channels_all_zero=True, context_model_config=self.cfg,
                       projection_state_dict=self.generator.hidden2latent.state_dict(),
                       distance_rbf_state_dict=self.generator.diffusion.dist_rbf.state_dict(),
                       pair_channels=self.generator.diffusion.pair_edge_dim,
                       vocabulary_config={}, seed=31,
                       heads={k: {'sha256': sha256(p)} for k, p in heads.items()})
        self.context = self.root / 'context.pt'; torch.save(context, self.context)
        self.ae, self.head_paths = ae, heads
        self.path = self.root / 'scorer.pt'
        self.payload = export_scorer(ae, self.context, heads, self.path)
        self.scorer = load_scorer(self.path)
        X = torch.randn(9, 14, 3) + torch.arange(9)[:, None, None] * .5
        self.batch = dict(X=X, S=torch.full((9,), 3, dtype=torch.long),
                          mask=torch.tensor([False] * 5 + [True] * 4),
                          position_ids=torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4]),
                          lengths=torch.tensor([9]), atom_mask=torch.ones(9, 14, dtype=torch.bool),
                          L=torch.eye(3).unsqueeze(0))

    def tearDown(self):
        self.tmp.cleanup()

    def test_deterministic_scoring_gradients_and_exact_portability(self):
        first = self.scorer(self.batch)
        state = torch.get_rng_state().clone()
        loaded = load_scorer(self.path)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, loaded(self.batch)))
        self.assertTrue(torch.equal(self.scorer.raw_scores(first), first * 3 + 2))
        X = self.batch['X'].clone().requires_grad_()
        value = loaded({**self.batch, 'X': X}).sum()
        gradient, = torch.autograd.grad(value, X)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[self.batch['mask']].norm()), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in loaded.parameters()))
        for key, path in self.head_paths.items():
            index = loaded.property_keys.index(key)
            expected = torch.load(path, weights_only=True)['state_dict']
            self.assertTrue(all(torch.equal(v, loaded.heads[index].state_dict()[k]) for k, v in expected.items()))

    def test_export_rejects_wrong_ae_or_head(self):
        payload = torch.load(self.ae, weights_only=True)
        payload['provenance']['source_sha256'] = 'different-ae'
        torch.save(payload, self.ae)
        with self.assertRaisesRegex(ValueError, 'AE source hash'):
            export_scorer(self.ae, self.context, self.head_paths, self.root / 'bad.pt')
        payload['provenance']['source_sha256'] = 'original-ae'; torch.save(payload, self.ae)
        bad = self.root / 'wrong.pt'; torch.save({}, bad)
        with self.assertRaisesRegex(ValueError, 'Head hash'):
            export_scorer(self.ae, self.context, {'affinity': bad}, self.root / 'bad.pt')
        with self.assertRaises(FileExistsError):
            export_scorer(self.ae, self.context, self.head_paths, self.path)

    def test_rejects_invalid_normalization_or_dimension(self):
        for field, value in [('property_std', [0.]), ('property_mean', [float('nan')])]:
            bad = deepcopy(self.payload); bad['heads']['affinity'][field] = value
            with self.assertRaises(ValueError):FrozenPropertyScorer(bad)
        bad = deepcopy(self.payload); bad['pair_channels'] += 1
        with self.assertRaisesRegex(ValueError, 'Pair-channel'):FrozenPropertyScorer(bad)

    def test_reconstructed_scorer_rejects_generator_bound_heads(self):
        bad = deepcopy(self.payload)
        bad['heads']['affinity']['feature_signature'] = feature_signature('generator', 10)
        with self.assertRaisesRegex(ValueError, 'Generator-feature heads'):
            FrozenPropertyScorer(bad)
        path = self.head_paths['affinity']
        torch.save(bad['heads']['affinity'], path)
        context = torch.load(self.context, map_location='cpu', weights_only=True)
        context['heads']['affinity']['sha256'] = sha256(path)
        torch.save(context, self.context)
        output = self.root / 'incompatible.pt'
        with self.assertRaisesRegex(ValueError, 'Generator-feature heads'):
            export_scorer(self.ae, self.context, self.head_paths, output)
        self.assertFalse(output.exists())

    def test_decoded_bridge_has_both_input_gradients(self):
        mask = self.batch['mask']; model = self.generator
        H = torch.randn(9, 8, requires_grad=True)
        X = self.batch['X'].clone().requires_grad_()
        bids = model.diffusion._get_batch_ids(mask, self.batch['lengths'])
        norm, centers = model.diffusion._normalize_position(X, bids, mask, self.batch['atom_mask'], self.batch['L'])
        output = decoded_scores(model, self.scorer, H, norm, H * .1, norm * .1,
                                self.batch, centers, 2)
        gh, gx = torch.autograd.grad(output.sum(), (H, X))
        for gradient in [gh, gx]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[mask].norm()), 0)

    def test_zero_weight_sampling_and_active_guidance(self):
        kwargs = {**self.batch, 'return_tensor': True, 'return_pair': True, 'optimize_sidechain': False}
        base = dict(energy_func=None, autoencoder_n_iter=1)
        torch.manual_seed(47)
        plain = self.generator.sample(**kwargs, sample_opt=base)
        options = dict(base, use_cg_guidance=True, property_guidance_mode='decoded_structure',
                       property_scorer=str(self.path), guidance_w_map={'affinity': 0., 'Sfinal': 0.},
                       lambda_max=.001, t_guidance_start=3, t_guidance_end=2,
                       guidance_grad_norm_balance=False)
        torch.manual_seed(47)
        zero = self.generator.sample(**kwargs, sample_opt=options)
        for index in range(3):self.assertTrue(torch.equal(plain[index], zero[index]))
        options['guidance_w_map']['affinity'] = 1.
        torch.manual_seed(47)
        guided = self.generator.sample(**kwargs, sample_opt=options)
        self.assertTrue(torch.isfinite(guided[0]).all())
        trace = guided[3]['property_guidance']
        self.assertEqual([row['t'] for row in trace], [2])
        self.assertGreater(trace[0]['latent_gradient_norm'], 0)
        self.assertGreater(trace[0]['coordinate_gradient_norm'], 0)
        self.assertGreater(trace[0]['latent_update_norm'], 0)
        self.assertGreater(trace[0]['coordinate_update_norm'], 0)

    def test_cli_resolves_scorer_path(self):
        cfg = dict(sample_opt=dict(property_scorer='scorer.pt', property_guidance_mode='decoded_structure',
                                   use_cg_guidance=True, lambda_max=.01, t_guidance_start=3))
        self.assertEqual(sampling_options(cfg, self.root)['property_scorer'], str(self.path))

    def test_guidance_rejects_nonfinite_and_ineffective_options(self):
        from apcyc.guidance import validate_guidance_options
        base = dict(use_cg_guidance=True, property_guidance_mode='decoded_structure',
                    property_scorer=str(self.path), lambda_max=.01,
                    t_guidance_start=3, t_guidance_end=2)
        invalid = [{'lambda_max': value} for value in [float('nan'), float('inf'), -1., 0., True]]
        invalid += [dict(t_guidance_start=1), dict(t_guidance_start=2, t_guidance_end=1),
                    dict(t_guidance_start=2.5), dict(t_guidance_end=4),
                    dict(guidance_grad_norm_eps=0), dict(guidance_grad_norm_eps=float('nan')),
                    dict(guidance_grad_norm_ref='typo'), dict(guidance_w_min=2., guidance_w_max=1.)]
        for change in invalid:
            with self.subTest(change=change):
                options = {**base, **change}
                with self.assertRaises(ValueError):validate_guidance_options(options, 6)
                with self.assertRaises(ValueError):sampling_options({'sample_opt': options}, self.root)
                with self.assertRaises(ValueError):
                    self.generator.sample(**self.batch, return_tensor=True, sample_opt=options)
        with self.assertRaisesRegex(ValueError, 'diffusion steps'):
            validate_guidance_options({**base, 't_guidance_start': 7}, 6)
        with self.assertRaises(ValueError):
            self.generator.diffusion.sample(H=torch.zeros(9, 8), X=self.batch['X'],
                position_embedding=torch.zeros(9, 8), mask_generate=self.batch['mask'],
                lengths=self.batch['lengths'], atom_embeddings=torch.zeros(9, 14, 4),
                atom_mask=self.batch['atom_mask'], S=self.batch['S'], use_cg_guidance=True,
                property_predictor=self.scorer, lambda_max=float('nan'), t_guidance_start=3)

    def test_each_graph_and_covariance_must_be_valid(self):
        invalid = [{k: v for k, v in self.batch.items() if k != 'L'},
                   {**self.batch, 'lengths': torch.tensor([4, 5]), 'L': torch.eye(3).repeat(2, 1, 1)},
                   {**self.batch, 'L': torch.ones(1, 3, 3)},
                   {**self.batch, 'L': torch.full((1, 3, 3), float('nan'))},
                   {**self.batch, 'lengths': torch.tensor([8])},
                   {**self.batch, 'position_ids': torch.full((9,), 3000, dtype=torch.long)}]
        for batch in invalid:
            with self.assertRaises(ValueError):self.scorer(batch)

    def structure(self):
        from data.format import Atom, Block
        from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
        chains = []
        for count, offset in [(5, 0.), (4, 12.)]:
            chains.append([Block('ALA', [Atom(name, [i * 3. + j * .2 + offset, float(i % 2), float(i % 3)], name[0])
                           for j, name in enumerate(['N', 'CA', 'C', 'O', 'CB'])], id=(i + 1, ' ')) for i in range(count)])
        path = self.root / 'candidate.pdb'
        list_blocks_to_pdb(chains, ['R', 'L'], str(path))
        return path

    def test_scoring_rejects_multiple_models_and_incomplete_backbones(self):
        from pipelines.predict_surrogates import structure_batch
        pdb = self.structure(); original = pdb.read_text()
        records = '\n'.join(line for line in original.splitlines() if line.startswith(('ATOM', 'TER'))) + '\n'
        multi = self.root / 'multi.pdb'
        multi.write_text('MODEL        1\n' + records + 'ENDMDL\nMODEL        2\n' + records + 'ENDMDL\nEND\n')
        with self.assertRaisesRegex(ValueError, 'one PDB model'):
            structure_batch({'pdb': str(multi)}, self.root)
        incomplete = '\n'.join(line for line in original.splitlines() if not
                     (line.startswith('ATOM') and line[21] == 'L' and int(line[22:26]) == 2 and line[12:16].strip() == 'O'))
        pdb.write_text(incomplete + '\n')
        with self.assertRaisesRegex(ValueError, 'Incomplete backbone'):
            structure_batch({'pdb': str(pdb)}, self.root)

    def test_scoring_ids_and_sequence_are_explicit(self):
        from pipelines.predict_surrogates import read_scoring_manifest, structure_batch
        path = self.root / 'manifest.jsonl'
        for invalid in [0, False, None, '', ' ']:
            path.write_text(json.dumps({'id': invalid, 'pdb': 'candidate.pdb'}) + '\n')
            with self.assertRaises(ValueError):read_scoring_manifest(path)
        path.write_text(json.dumps({'id': 'NA', 'pdb': 'candidate.pdb'}) + '\n')
        self.assertEqual(read_scoring_manifest(path)[0]['id'], 'NA')
        pdb = self.structure()
        with self.assertRaisesRegex(ValueError, 'sequence differs'):
            structure_batch({'pdb': str(pdb), 'sequence': 'GGGG'}, self.root)
        self.assertEqual(int(structure_batch({'pdb': str(pdb), 'sequence': 'AAAA'}, self.root)[0]['mask'].sum()), 4)

    def test_same_path_scorer_replacement_reloads_cached_parameters(self):
        options = dict(use_cg_guidance=True, property_guidance_mode='decoded_structure',
                       property_scorer=str(self.path), guidance_w_map={'affinity': 0., 'Sfinal': 0.},
                       lambda_max=.01, t_guidance_start=3, t_guidance_end=2, energy_func=None)
        self.generator.sample(**self.batch, return_tensor=True, optimize_sidechain=False, sample_opt=options)
        old = self.generator.property_predictor.heads[0].heads[0].bias.clone()
        payload = torch.load(self.path, weights_only=True)
        payload['heads']['affinity']['state_dict']['heads.0.bias'] += 7
        torch.save(payload, self.path)
        self.generator.sample(**self.batch, return_tensor=True, optimize_sidechain=False, sample_opt=options)
        self.assertTrue(torch.equal(self.generator.property_predictor.heads[0].heads[0].bias, old + 7))

    def test_native_head_load_preserves_sampling_rng(self):
        from apcyc.guidance import load_property_predictor
        before = torch.get_rng_state().clone()
        loaded = load_property_predictor(self.head_paths['affinity'], 'cpu')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(loaded.property_keys, ['affinity'])


    def test_guidance_cache_preserves_generator_checkpoint_roundtrip(self):
        from apcyc.checkpoint import load_bundle
        before = {key: value.clone() for key, value in self.generator.state_dict().items()}
        options = dict(use_cg_guidance=True, property_guidance_mode='decoded_structure',
                       property_scorer=str(self.path), guidance_w_map={'affinity': 0., 'Sfinal': 0.},
                       lambda_max=.01, t_guidance_start=3, t_guidance_end=2, energy_func=None)
        self.generator.sample(**self.batch, return_tensor=True, optimize_sidechain=False, sample_opt=options)
        actual = self.generator.state_dict()
        self.assertEqual(set(before), set(actual))
        self.assertTrue(all(torch.equal(value, actual[key]) for key, value in before.items()))
        self.assertNotIn('property_predictor', dict(self.generator.named_modules()))
        path = self.root/'generator.ckpt'
        save_model_checkpoint(self.generator, {'model': {'class': 'APCycLDMPepDesign', **self.cfg}}, path)
        restored, _ = load_bundle(path)
        self.assertEqual(set(before), set(restored.state_dict()))
        self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in before.items()))
        self.assertIsNone(restored.property_predictor)

if __name__ == '__main__':
    unittest.main()
