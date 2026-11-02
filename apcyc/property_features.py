"""Shared frozen-generator features for surrogate training and classifier guidance."""
import hashlib
from pathlib import Path
import torch

FEATURE_CONTRACT = 'apcyc.generator_features.v2'


def feature_signature(generator_sha256, t_hard):
    root = Path(__file__).resolve().parent
    paths = ['property_features.py', 'pair_init.py', 'cyclization_heads.py',
             'cyclic_injection.py', 'chemistry.py', 'diffusion/apcyc_dpm.py', 'model.py', 'property_predictor.py']
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode() + (root / name).read_bytes())
    return dict(contract=FEATURE_CONTRACT, generator_bundle_sha256=generator_sha256,
                implementation_sha256=digest.hexdigest(), t_hard=int(t_hard),
                pair_features='generator', context_features='generator',
                peptide_identity='AE_decoded_for_paper_masked_for_historical', time_sampling='shared_uniform_1_to_T',
                target_direction='higher_is_better')


def property_features(diffusion, H, X, S, mask_generate, lengths, atom_mask,
                      position_ids, batch_ids, centers, L, t, t_hard, atom_embeddings):
    # A shared diffusion time per batch matches inference and makes hard topology
    # selection identical. It retains the uniform marginal time distribution.
    if torch.is_tensor(t):
        if not torch.all(t == t.reshape(-1)[0]):
            raise ValueError('Property features require one shared diffusion time per batch.')
        t = int(t.reshape(-1)[0])
    X_unorm = diffusion._unnormalize_position(X, centers, batch_ids, L)
    pair_out = diffusion._pair_forward(
        H, X_unorm, S, mask_generate, batch_ids, position_ids, type_labels=None, atom_mask=atom_mask)
    features = diffusion.topology_edge_inputs(
        pair_out, X_unorm, atom_mask, mask_generate, batch_ids, lengths, t, t_hard=t_hard)
    edge_attr = torch.cat([features['ctx_edge_attr'], features['inter_edge_attr']], dim=0)
    return dict(H_noisy=H, X_noisy=X, mask_generate=mask_generate, atom_mask=atom_mask,
                position_ids=position_ids, ctx_edges=features['ctx_edges'], inter_edges=features['inter_edges'],
                edge_attr=edge_attr, edge_gate=features['edge_gate'], edge_bias=features['edge_bias'],
                edge_type=features['edge_type'], pair_struct=pair_out['pair_struct'],
                pair_cyc=pair_out['pair_cyc'], type_probs=features['type_probs'], s_ij=pair_out['s_ij_full'],
                t=features['t'], batch_ids=batch_ids, atom_embeddings=atom_embeddings)
