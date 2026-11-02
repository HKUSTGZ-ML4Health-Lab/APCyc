"""Regression checks for public inference, serialization and property boundaries."""
import csv
import json
import os
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

from apcyc.guidance import sigma2_from_transition
from apcyc.sample import peptide_blocks, sampling_options, prepare_input
from data.format import Atom, Block
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from pipelines.collect_properties import collect, expand, merge_stage
from pipelines.score_properties import fit_normalizer, score


class ReleaseRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def structure(self):
        chains = []
        for count, offset in [(5, 0), (4, 10)]:
            blocks = []
            for i in range(count):
                xyz = np.array([i * 3., i % 2, i % 3]) + offset
                blocks.append(Block('ALA', [Atom(a, (xyz + j * .3).tolist(), a[0]) for j, a in enumerate(['N', 'CA', 'C', 'O', 'CB'])], id=(i + 10, 'A' if i == 0 else ' ')))
            chains.append(blocks)
        path = self.root / 'structure.pdb'
        list_blocks_to_pdb(chains, ['R', 'L'], str(path))
        return path

    def test_pdb_atom_and_residue_identity_roundtrip(self):
        pdb = self.structure()
        model = PDBParser(QUIET=True).get_structure('x', str(pdb))[0]
        self.assertEqual(set(a.id for a in model['R'].child_list[0]), {'N', 'CA', 'C', 'O', 'CB'})
        self.assertEqual(model['R'].child_list[0].id, (' ', 10, 'A'))

    def test_generated_tokens_and_one_based_pdb_residues(self):
        blocks, seq = peptide_blocks(['A', 'G', 'C', 'K'], np.zeros((4, 14, 3)))
        self.assertEqual(seq, 'AGCK')
        self.assertEqual([b.id[0] for b in blocks], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            peptide_blocks(['A'], np.full((1, 14, 3), np.nan))

    def test_conditioning_ignores_target_ligand(self):
        pdb = self.structure()
        pocket = self.root / 'pocket.json'
        pocket.write_text(json.dumps([['R', [i + 10, 'A' if i == 0 else ' ']] for i in range(5)]))
        rec = dict(sample_id='x', complex_pdb=pdb, pocket_json=pocket)
        _, before = prepare_input(rec, 6, 'cpu')
        pdb.write_text('\n'.join(line for line in pdb.read_text().splitlines() if not (line.startswith('ATOM  ') and line[21] == 'L')) + '\n')
        _, after = prepare_input(rec, 6, 'cpu')
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertEqual(int(after['mask'].sum()), 6)

    def test_model_import_in_fresh_process(self):
        result = subprocess.run([sys.executable, '-c', 'from apcyc.model import APCycLDMPepDesign'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_receptor_conditioning_matches_historical_dataset(self):
        from apcyc.dataset import CPCoreDataset
        pdb = self.structure()
        pocket = self.root / 'structure.json'
        pocket.write_text(json.dumps([['R', [i + 10, 'A' if i == 0 else ' ']] for i in range(5)]))
        index = self.root / 'index.txt'
        index.write_text('structure\n')
        metadata = self.root / 'metadata.jsonl'
        metadata.write_text(json.dumps({'sample_id':'structure', 'peptide_length':4, 'c_star':'HT', 'gt_pair':[0,3]}) + '\n')
        dataset = CPCoreDataset(str(self.root), str(index), str(metadata), pocket_json_dir=str(self.root),
                                pocket_json_suffix='.json', require_pocket=True, use_covariance_matrix=True,
                                return_supervision=False)
        old = dataset[0]
        _, new = prepare_input(dict(sample_id='structure', complex_pdb=pdb, pocket_json=pocket), 4, 'cpu')
        for key in ['X', 'S', 'atom_mask']:
            self.assertTrue(torch.equal(old[key][~old['mask']], new[key][~new['mask']]), key)
        for key in ['position_ids', 'mask', 'L']:
            self.assertTrue(torch.equal(old[key], new[key]), key)

    def test_sigma_scalar_broadcast(self):
        transition = SimpleNamespace(var_sched=SimpleNamespace(sigmas=torch.tensor([0., .5, 1.])))
        result = sigma2_from_transition(transition, 1, torch.tensor([0, 0, 1, 1, 1]), (5, 1, 1))
        self.assertEqual(result.shape, (5, 1, 1))
        self.assertTrue(torch.all(result == .25))

    def test_covalent_edges_keep_attributes_aligned(self):
        import models  # register mutually referenced historical model classes
        from apcyc.diffusion.apcyc_dpm import append_covalent_edges
        ctx = torch.tensor([[0, 1], [1, 0]])
        inter = torch.tensor([[0], [2]])
        cov = torch.tensor([[2], [3]])
        ctx2, edges, types = append_covalent_edges(ctx, inter, torch.tensor([0, 0, 1]), cov, torch.tensor([3]))
        self.assertEqual(edges.tolist(), [[0, 1, 2, 0], [1, 0, 3, 2]])
        self.assertEqual(types.tolist(), [0, 0, 3, 1])
        self.assertTrue(torch.equal(edges[:, :3], ctx2))

    def test_property_compatibility_rejects_dimensions_and_missing_context(self):
        from apcyc.guidance import validate_property_compatibility
        diffusion = SimpleNamespace(dist_rbf_dim=16, pair_edge_dim=53)
        with self.assertRaisesRegex(ValueError, 'expects 85'):
            validate_property_compatibility(SimpleNamespace(edge_size=85, n_properties=1, property_keys=['affinity']), diffusion, 'bundle')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_property_compatibility(SimpleNamespace(edge_size=69, n_properties=1, property_keys=['affinity']), diffusion, 'bundle')
        from apcyc.property_features import feature_signature
        head = SimpleNamespace(edge_size=69, n_properties=1, property_keys=['affinity'], feature_signature=feature_signature('bundle', 10))
        validate_property_compatibility(head, diffusion, 'bundle')

    def test_guidance_computes_gradients_inside_no_grad_sampler(self):
        from apcyc.diffusion.apcyc_dpm import APCycFullDPM
        class DifferentiableProbe(torch.nn.Module):
            n_properties = 1
            def forward(self, H_noisy, X_noisy, mask_generate, **kwargs):
                return (H_noisy[mask_generate].mean() + X_noisy[mask_generate].square().mean()).reshape(1, 1)
        torch.manual_seed(7)
        dpm = APCycFullDPM(latent_size=8, hidden_size=16, n_channel=14, num_steps=3, n_layers=1).eval()
        H = torch.randn(5, 8)
        X = torch.randn(5, 14, 3)
        arguments = dict(H=H, X=X, position_embedding=torch.zeros(5, 8),
                         mask_generate=torch.tensor([False, False, False, True, True]),
                         lengths=torch.tensor([5]), atom_embeddings=torch.zeros(5, 14, 4),
                         atom_mask=torch.ones(5, 14, dtype=torch.bool), S=torch.tensor([4,4,4,3,3]),
                         position_ids=torch.tensor([1,2,3,1,2]), L=torch.eye(3).unsqueeze(0))
        torch.manual_seed(42)
        plain, _ = dpm.sample(**arguments)
        torch.manual_seed(42)
        guided, _ = dpm.sample(**arguments, use_cg_guidance=True, property_predictor=DifferentiableProbe(),
                               t_guidance_start=3, t_guidance_end=1, lambda_max=.3,
                               guidance_grad_norm_balance=True)
        self.assertTrue(torch.isfinite(guided[0][0]).all())
        self.assertFalse(torch.allclose(plain[0][1][3:], guided[0][1][3:]))
        self.assertTrue(all(p.grad is None for p in dpm.parameters()))

    def test_sampling_rejects_unknown_or_inactive_guidance(self):
        with self.assertRaises(ValueError):
            sampling_options({'sample_opt': {'guidance_typo': 1}}, self.root)
        with self.assertRaises(ValueError):
            sampling_options({'sample_opt': {'use_cg_guidance': True}}, self.root)

    def test_normalization_uses_only_fit_ids(self):
        df = pd.DataFrame({'id': ['train1', 'train2', 'test'], 'logP': [1., 3., 100.], 'GRAVY': [1., 3., 100.], 'TPSA': [1., 3., 1.], 'rTPSA': [1., 3., 1.], 'rosetta_dG': [-2., -4., 99.], 'vina_score': [-1., -3., 5.]})
        norm = fit_normalizer(df, {'train1', 'train2'})
        self.assertEqual(norm['statistics']['logP']['mean'], 2.)
        out = score(df, norm)
        self.assertTrue(np.isfinite(out.loc[2, 'affinity']))  # finite energies remain eligible
        legacy = {**norm, 'format': 'apcyc.property_normalizer.v1'}
        self.assertTrue(np.isnan(score(df, legacy).loc[2, 'affinity']))
        self.assertNotIn('tox_score', out.columns)
        self.assertNotIn('agg_risk', out.columns)
        self.assertEqual(norm['fit_ids'], ['train1', 'train2'])

    def test_environment_expansion_precedes_placeholders(self):
        with patch.dict(os.environ, {'APCYC_TEST_BIN': '/tmp/tool with space'}):
            self.assertEqual(expand('${APCYC_TEST_BIN}/{file}', {'file': 'x'}), '/tmp/tool with space/x')
        with self.assertRaises(ValueError):
            expand('${APCYC_UNSET_TEST}/x', {})

    def test_stage_nonfinite_values_are_missing(self):
        p = self.root / 'values.csv'
        p.write_text('id,score\na,nan\nb,4.2\n')
        spec = {'columns': {'camsol_score': 'score'}}
        self.assertEqual(merge_stage(p, spec, {'a', 'b'}), {'a': {'camsol_score': ''}, 'b': {'camsol_score': '4.2'}})
        p.write_text('id,score\na,1\na,2\n')
        with self.assertRaises(ValueError):
            merge_stage(p, spec, {'a'})

    def test_collection_import_and_explicit_missingness(self):
        pdb = self.structure()
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text(json.dumps(dict(id='a', pdb=str(pdb), peptide_chain='L')) + '\n')
        raw = self.root / 'camsol.csv'
        raw.write_text('id,score\na,nan\n')
        config = self.root / 'config.json'
        config.write_text(json.dumps({'stages': {'camsol': {'import': str(raw), 'columns': {'camsol_score': 'score'}}}}))
        out = self.root / 'result'
        args = SimpleNamespace(config=config, manifest=manifest, out_dir=out, only=None, run_external=False, strict=False)
        collect(args)
        status = json.loads((out / 'status.json').read_text())
        self.assertEqual(status['stages']['camsol']['status'], 'partial')
        self.assertEqual(status['missing_counts']['camsol_score'], 1)
        self.assertEqual(status['missing_counts']['GRAVY'], 0)
        args.strict = True
        with self.assertRaises(SystemExit):
            collect(args)


if __name__ == '__main__':
    unittest.main()
