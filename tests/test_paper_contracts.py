"""Method-level checks for paper defaults and strict topology preservation."""
from types import SimpleNamespace
from pathlib import Path
import unittest
import torch
import yaml

from apcyc.cyclization_heads import CyclizationHeads, _valid_mask_iso
from apcyc.property_features import feature_signature
from apcyc.guidance import validate_property_compatibility
from data.format import VOCAB
from pipelines.postprocess import closure_spec


class PaperContracts(unittest.TestCase):
    def test_cluster_audit_detects_leakage_despite_disjoint_sample_ids(self):
        from scripts.data.cluster_split import audit
        owner = {'a': 'c1', 'b': 'c1', 'c': 'c2'}
        self.assertFalse(audit(owner, {'a'}, {'b','c'})['cluster_independent'])
        self.assertTrue(audit(owner, {'a','b'}, {'c'})['cluster_independent'])
        with self.assertRaisesRegex(ValueError, 'lack cluster membership'):
            audit(owner, {'a','unknown'}, {'c'})

    def test_rosetta_decoys_resolve_only_to_known_inputs(self):
        from pipelines.adapters.rosetta import normalize_id
        self.assertEqual(normalize_id('target_0001', {'target'}), 'target')
        self.assertEqual(normalize_id('target_0001.pdb', {'target_0001'}), 'target_0001')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            normalize_id('unknown_0001', {'target'})

    def test_distance_features_preserve_pair_graph_and_coordinate_gradients(self):
        from apcyc.diffusion.apcyc_dpm import APCycFullDPM
        x = torch.randn(5, 3, 3, requires_grad=True)
        before = x.detach().clone()
        pair_loss = torch.cdist(x[:, 1], x[:, 1]).square().sum()
        mask = torch.ones(5, 3, dtype=torch.bool)
        mask[2:, 1] = False
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        distance = APCycFullDPM._get_edge_dist(None, x, edges, mask)
        self.assertTrue(torch.equal(x.detach(), before))
        self.assertTrue(distance.requires_grad)
        gradient, = torch.autograd.grad(pair_loss + distance.sum(), x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[2:, 0].abs().sum().item(), 0)

    def test_chemistry_masks_use_paper_iso_and_exclude_impossible_types(self):
        mask = _valid_mask_iso(['K', 'A', 'N', 'E'], torch.device('cpu'), paper=True)
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[0, 3])
        head = CyclizationHeads(4, chemistry_profile='paper', relax_residue_constraints=False)
        types, _, pair = head.forward_single(torch.randn(4,4,4),
                                             torch.tensor([VOCAB.symbol_to_idx('A')]*4),
                                             type_logits=torch.tensor([0., 100., 200.]))
        self.assertEqual(types.argmax().item(), 0)
        self.assertEqual(pair[0,3].item(), 1.)
        self.assertTrue(torch.isfinite(pair).all())

    def test_postprocess_preserves_sequence_and_explicit_sites(self):
        res = [SimpleNamespace(resname=n) for n in ['LYS','ALA','ASN','GLU']]
        row = dict(pair_index_base=0, pair_hat=[0,2], type_hat='ISO')
        self.assertEqual(closure_spec(row,res)['atoms'], ['NZ','CG'])
        with self.assertRaisesRegex(ValueError, 'requires_Lys_Asp_or_Asn'):
            closure_spec({**row,'pair_hat':[0,3]},res)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            closure_spec({**row,'type_hat':'HT'},res)
        with self.assertRaisesRegex(ValueError, 'explicitly_zero'):
            closure_spec({**row,'pair_index_base':1},res)

    def test_guidance_rejects_changed_schedule_or_implementation(self):
        diffusion = SimpleNamespace(dist_rbf_dim=16, pair_edge_dim=53)
        head = SimpleNamespace(edge_size=69, n_properties=1, property_keys=['affinity'],
                               feature_signature=feature_signature('bundle',10))
        validate_property_compatibility(head,diffusion,'bundle',10)
        with self.assertRaisesRegex(ValueError,'contract'):
            validate_property_compatibility(head,diffusion,'bundle',5)
        head.feature_signature['implementation_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError,'contract'):
            validate_property_compatibility(head,diffusion,'bundle',10)

    def test_public_training_configs_require_new_training(self):
        root = Path(__file__).resolve().parents[1]
        ae = yaml.safe_load((root/'configs/training/autoencoder.yaml').read_text())
        ldm = yaml.safe_load((root/'configs/training/diffusion.yaml').read_text())
        self.assertEqual(ae['model']['loss_mode'],'paper')
        self.assertTrue(ae['model']['enable_enlarged_vocab'])
        self.assertNotIn('load_ckpt',ae)
        self.assertNotIn('load_ckpt',ldm)
        self.assertGreater(ldm['model']['loss_opt']['w_type'],0)
        self.assertFalse(ldm['model']['pair_opt']['relax_residue_constraints'])
        self.assertNotIn('w_ring',ldm['model']['loss_opt'])


if __name__ == '__main__':
    unittest.main()
