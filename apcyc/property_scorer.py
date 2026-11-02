"""Frozen property scoring in its own AE representation and decoded guidance."""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
from torch import nn

from apcyc.checkpoint import sha256
from apcyc.guidance import predictor_keys
from apcyc.property_predictor import PropertyPredictor
from utils.checkpoint_utils import configure_vocabulary, load_autoencoder_checkpoint

FORMAT = 'apcyc.frozen_property_scorer.v1'


def _require_reconstructed_head(payload):
    if payload.get('feature_signature') is not None:
        raise ValueError('Generator-feature heads require their original generator and '
                         'configs/inference/guided.yaml; they cannot use a reconstructed scorer context.')


def export_scorer(autoencoder, context, heads, output):
    """Bind exact head tensors to an explicit AE and saved context projection."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ctx = torch.load(context, map_location='cpu', weights_only=True)
    ae_payload = torch.load(autoencoder, map_location='cpu', weights_only=True)
    expected_ae = ctx['referenced_ae']['sha256']
    if ae_payload.get('provenance', {}).get('source_sha256') != expected_ae:
        raise ValueError('Exported AE source hash differs from the context reconstruction.')
    if not ctx.get('pair_channels_all_zero') or ctx.get('optimizer_steps') != 0:
        raise ValueError('Context must declare frozen reconstruction with zero pair channels.')
    projection = ctx['projection_state_dict']
    projection_hash = hashlib.sha256(projection['weight'].contiguous().numpy().tobytes()
                                     + projection['bias'].contiguous().numpy().tobytes()).hexdigest()
    if ctx.get('projection_sha256', projection_hash) != projection_hash:
        raise ValueError('Reconstructed projection checksum mismatch.')
    ae = load_autoencoder_checkpoint(autoencoder)
    payloads = {}
    for key, path in heads.items():
        if key not in ctx['heads'] or sha256(path) != ctx['heads'][key]['sha256']:
            raise ValueError(f'Head hash differs from the reconstructed context: {key}')
        value = torch.load(path, map_location='cpu', weights_only=True)
        _require_reconstructed_head(value)
        if value.get('property_keys') != [key]:
            raise ValueError(f'Head target differs from its configured name: {key}')
        payloads[key] = value
    if not payloads:
        raise ValueError('Provide at least one property head.')
    from data.format import VOCAB
    cfg = deepcopy(ctx['context_model_config'])
    cfg.pop('class', None)
    cfg['autoencoder_ckpt'] = deepcopy(ae._checkpoint_model_cfg)
    bundle = dict(format=FORMAT, model_config=cfg,
                  autoencoder_state_dict=ae.state_dict(),
                  projection_state_dict=ctx['projection_state_dict'],
                  distance_rbf_state_dict=ctx['distance_rbf_state_dict'],
                  pair_channels=ctx['pair_channels'], heads=payloads,
                  cyc_config=deepcopy(ctx['vocabulary_config']),
                  vocabulary=[VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))],
                  provenance=dict(context_origin='reconstructed', seed=ctx['seed'],
                                  original_extractor_archived=False,
                                  context_sha256=sha256(context), ae_source_sha256=expected_ae,
                                  ae_bundle_sha256=sha256(autoencoder),
                                  head_sha256={k: sha256(p) for k, p in heads.items()}))
    # Validate every component before exposing the destination.
    with torch.random.fork_rng(devices=[]):
        FrozenPropertyScorer(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        torch.save(bundle, handle)
    return bundle


class FrozenPropertyScorer(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        if bundle.get('format') != FORMAT:
            raise ValueError('Expected a frozen property-scorer bundle.')
        configure_vocabulary(bundle)
        from apcyc.model import APCycLDMPepDesign
        self.features = APCycLDMPepDesign(**bundle['model_config'])
        if (not self.features.train_sequence or not self.features.train_structure
                or self.features.autoencoder.mode != 'codesign'):
            raise ValueError('Frozen property scoring requires a codesign feature extractor.')
        self.features.autoencoder.load_state_dict(bundle['autoencoder_state_dict'], strict=True)
        self.features.hidden2latent.load_state_dict(bundle['projection_state_dict'], strict=True)
        self.features.diffusion.dist_rbf.load_state_dict(bundle['distance_rbf_state_dict'], strict=True)
        self.pair_channels = int(bundle['pair_channels'])
        if self.pair_channels != self.features.diffusion.pair_edge_dim:
            raise ValueError('Pair-channel count differs from the feature architecture.')
        self.property_keys = list(bundle['heads'])
        self.n_properties = len(self.property_keys)
        if not self.n_properties:
            raise ValueError('Empty property-scorer bundle.')
        self.heads = nn.ModuleList()
        means, stds = [], []
        for key, payload in bundle['heads'].items():
            _require_reconstructed_head(payload)
            head = PropertyPredictor(**payload['model_cfg'])
            head.property_keys = payload['property_keys']
            if predictor_keys(head) != [key]:
                raise ValueError('Scorer requires one correctly named output per head.')
            if head.edge_size != self.features.diffusion.dist_rbf_dim + self.pair_channels:
                raise ValueError('Head edge dimension differs from its own feature extractor.')
            head.load_state_dict(payload['state_dict'], strict=True)
            mean = torch.as_tensor(payload['property_mean'], dtype=torch.float).reshape(-1)
            std = torch.as_tensor(payload['property_std'], dtype=torch.float).reshape(-1)
            if (mean.numel() != 1 or std.numel() != 1 or not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all() or not (std > 0).all()):
                raise ValueError('Each head needs one finite mean and positive standard deviation.')
            means.append(mean[0]); stds.append(std[0])
            self.heads.append(head)
        self.register_buffer('property_mean', torch.stack(means))
        self.register_buffer('property_std', torch.stack(stds))
        self.provenance = bundle['provenance']
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def predict_latents(self, H, X, batch, atoms, centers, t):
        diffusion = self.features.diffusion
        mask, lengths, atom_mask = [batch[k] for k in ['mask', 'lengths', 'atom_mask']]
        bids = diffusion._get_batch_ids(mask, lengths)
        absolute = diffusion._unnormalize_position(X, centers, bids, batch.get('L'))
        anchor = torch.where(atom_mask[:, 1, None], absolute[:, 1], absolute[:, 0])
        ctx, inter = diffusion._get_edges(mask, bids, lengths)
        edges = torch.cat([ctx, inter], dim=1)
        distances = (anchor[edges[0]] - anchor[edges[1]]).norm(dim=-1)
        radial = diffusion.dist_rbf(distances)
        attributes = torch.cat([radial, radial.new_zeros(len(distances), self.pair_channels)], dim=-1)
        common = dict(mask_generate=mask, atom_mask=atom_mask, position_ids=batch['position_ids'],
                      ctx_edges=ctx, inter_edges=inter, edge_attr=attributes,
                      t=t, batch_ids=bids, atom_embeddings=atoms)
        return torch.cat([head(H, X, **common) for head in self.heads], dim=1)

    def forward(self, batch):
        """Score physical coordinates using deterministic AE means at clean time 0."""
        X, S, mask, positions, lengths, atom_mask = [batch[k] for k in
                                                   ['X', 'S', 'mask', 'position_ids', 'lengths', 'atom_mask']]
        n = X.shape[0]
        if (lengths.ndim != 1 or not len(lengths) or lengths.dtype not in {torch.int32, torch.int64}
                or (lengths <= 0).any() or int(lengths.sum()) != n or mask.shape != (n,)
                or mask.dtype != torch.bool or S.shape != (n,) or positions.shape != (n,)
                or X.shape != (n, self.features.autoencoder.n_channel, 3)
                or atom_mask.shape != X.shape[:2] or atom_mask.dtype != torch.bool):
            raise ValueError('Scoring batch shapes, lengths and masks must describe the same residues.')
        if not torch.isfinite(X).all() or not mask.any() or not (~mask).any():
            raise ValueError('Scoring requires finite peptide and receptor coordinates.')
        bids = torch.repeat_interleave(torch.arange(len(lengths), device=mask.device), lengths.to(mask.device))
        context = ~mask & atom_mask[:, self.features.autoencoder.ca_channel_idx]
        if ((torch.bincount(bids[mask], minlength=len(lengths)) == 0).any()
                or (torch.bincount(bids[context], minlength=len(lengths)) == 0).any()):
            raise ValueError('Every scoring graph requires peptide residues and visible receptor CA atoms.')
        L = batch.get('L')
        if (not torch.is_tensor(L) or L.shape != (len(lengths), 3, 3)
                or L.dtype != X.dtype or not torch.isfinite(L).all()
                or (L.diagonal(dim1=-2, dim2=-1) <= 0).any() or (L.triu(1) != 0).any()):
            raise ValueError('Scoring requires a finite positive lower-triangular receptor covariance factor L.')
        position_limit = min(head.pos_embed.num_embeddings for head in self.heads)
        ae_positions = self.features.autoencoder.aa_feature.aa_embedding.res_pos_embedding
        if isinstance(ae_positions, nn.Embedding):
            position_limit = min(position_limit, ae_positions.num_embeddings)
        if positions.min() < 0 or positions.max() >= position_limit:
            raise ValueError('Residue position exceeds the scorer or AE position vocabulary.')
        model = self.features
        H, Z, _, _ = model.autoencoder.encode(X, S, mask, positions, lengths, atom_mask,
                                             no_randomness=True)
        masked = S.clone(); masked[mask] = model.latent_idx
        context, (atoms, _) = model.autoencoder.aa_feature(masked, positions)
        H0 = model.hidden2latent(context).clone(); H0[mask] = H
        X0 = X.clone(); X0[mask] = model.autoencoder._fill_latent_channels(Z, S[mask])
        latent_mask = atom_mask.clone()
        latent_mask[mask] = False
        latent_mask[mask, :model.autoencoder.latent_n_channel] = True
        batch = {**batch, 'atom_mask': latent_mask}
        bids = model.diffusion._get_batch_ids(mask, lengths)
        normalized, centers = model.diffusion._normalize_position(X0, bids, mask, latent_mask, batch.get('L'))
        times = torch.zeros(len(lengths), dtype=torch.long, device=X.device)
        result = self.predict_latents(H0, normalized, batch, atoms, centers, times)
        if not torch.isfinite(result).all():
            raise ValueError('Property scorer produced a nonfinite value.')
        return result

    def raw_scores(self, normalized):
        return normalized * self.property_std + self.property_mean


def load_scorer(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    # Construction must not alter the sampling RNG, including zero-weight controls.
    with torch.random.fork_rng(devices=[]):
        scorer = FrozenPropertyScorer(payload)
    return scorer.to(device).eval()


def decoded_scores(generator, scorer, H, X, eps_H, eps_X, batch, centers, t):
    """Map predicted clean generator states through structure to the scoring AE."""
    mask, lengths = batch['mask'], batch['lengths']
    diffusion = generator.diffusion
    def clean(value, noise, transition):
        schedule = getattr(transition, 'var_sched', None)
        if schedule is None:
            raise ValueError('Decoded property guidance requires DDPM transitions.')
        alpha = schedule.alpha_bars[t]
        if alpha <= 1e-6:
            raise ValueError('Start decoded guidance after the terminal high-noise steps.')
        estimate = (value - torch.sqrt(1 - alpha) * noise) / torch.sqrt(alpha)
        expand = mask.reshape(-1, *([1] * (value.ndim - 1)))
        return torch.where(expand, estimate, value)
    H0 = clean(H, eps_H, diffusion.trans_h)
    X0 = clean(X, eps_X, diffusion.trans_x)
    bids = diffusion._get_batch_ids(mask, lengths)
    absolute = diffusion._unnormalize_position(X0, centers, bids, batch.get('L'))
    ae = generator.autoencoder
    logits, coordinates = ae.decode(absolute, batch['S'], H0[mask],
                                    absolute[mask, :ae.latent_n_channel], mask,
                                    batch['position_ids'], lengths, batch['atom_mask'],
                                    teacher_forcing=False, allow_unk=False)
    tokens = batch['S'].clone(); tokens[mask] = ae.s_remap[logits.argmax(dim=-1)]
    physical = absolute.clone(); physical[mask] = coordinates
    _, (_, visibility) = ae.aa_feature(tokens, batch['position_ids'])
    atom_mask = batch['atom_mask'].clone(); atom_mask[mask] = visibility[mask].bool()
    # Argmax tokens are fixed within each gradient evaluation. Coordinates and
    # both AE transforms retain their graphs, as does the diffusion denoiser.
    return scorer({**batch, 'X': physical, 'S': tokens, 'atom_mask': atom_mask})
