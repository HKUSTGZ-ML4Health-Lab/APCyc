#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB

from models.autoencoder.model import AutoEncoder


def _strip_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_autoencoder(ckpt_path: str, strict: bool) -> AutoEncoder:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, AutoEncoder):
        return ckpt
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        if any(k.startswith('autoencoder.') for k in state_dict):
            state_dict = _strip_prefix(state_dict, 'autoencoder.')
        cfg = ckpt.get('model_cfg')
        if cfg is None:
            raise ValueError('Autoencoder checkpoint missing model_cfg for strict load.')
        autoencoder = AutoEncoder(**cfg)
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder
    raise TypeError(f'Unsupported autoencoder checkpoint type: {type(ckpt)}')
from apcyc.diffusion.apcyc_dpm import APCycFullDPM
from apcyc.constants import CYCLE_TYPES, TYPE2IDX


@R.register('APCycLDMPepDesign')
class APCycLDMPepDesign(nn.Module):
    def __init__(
        self,
        autoencoder_ckpt,
        autoencoder_no_randomness,
        hidden_size,
        num_steps,
        n_layers,
        dist_rbf=0,
        dist_rbf_cutoff=7.0,
        n_rbf=0,
        cutoff=1.0,
        max_gen_position=30,
        mode='codesign',
        h_loss_weight=None,
        diffusion_opt={},
        pair_opt={},
        loss_opt={},
        autoencoder_strict_load: bool = True,
    ):
        super().__init__()
        self.autoencoder_no_randomness = autoencoder_no_randomness
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)

        self.autoencoder: AutoEncoder = _load_autoencoder(autoencoder_ckpt, autoencoder_strict_load)
        # Backward-compat for older checkpoints without anchor-channel fields.
        if not hasattr(self.autoencoder, 'enable_latent_anchor_channels'):
            self.autoencoder.enable_latent_anchor_channels = False
        if not hasattr(self.autoencoder, 'latent_anchor_atoms'):
            self.autoencoder.latent_anchor_atoms = None
        if not hasattr(self.autoencoder, 'latent_anchor_pad_atom'):
            self.autoencoder.latent_anchor_pad_atom = 'CA'
        if not hasattr(self.autoencoder, 'latent_anchor_strict'):
            self.autoencoder.latent_anchor_strict = False
        if not hasattr(self.autoencoder, 'anchor_channel_map'):
            self.autoencoder.anchor_channel_map = None
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False

        latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size

        self.abs_position_encoding = nn.Embedding(max_gen_position, latent_size)
        self.diffusion = APCycFullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            ca_idx=self.autoencoder.ca_channel_idx,
            **diffusion_opt,
            **pair_opt,
        )
        self.property_predictor = None
        if self.train_sequence:
            self.hidden2latent = nn.Linear(hidden_size, self.autoencoder.latent_size)
            if h_loss_weight is None:
                self.h_loss_weight = self.autoencoder.latent_n_channel * 3 / self.autoencoder.latent_size
            else:
                self.h_loss_weight = h_loss_weight
        if self.train_structure:
            self.consec_dist_mean, self.consec_dist_std = None, None

        self.enable_type_loss = loss_opt.get('enable_type_loss', True)
        self.w_type = loss_opt.get('w_type', 1.0)
        self.w_pair = loss_opt.get('w_pair', 1.0)
        self.pair_loss_requires_type_match = loss_opt.get('pair_loss_requires_type_match', False)
        self.ignore_ht_pair_loss = loss_opt.get('ignore_ht_pair_loss', True)

    def _map_type_label(self, c_star: torch.Tensor) -> torch.Tensor:
        if c_star.dtype == torch.long:
            return c_star
        # if given as string indices in tensor, fallback
        return c_star.long()

    def _pair_loss(
        self,
        s_ij_full: torch.Tensor,
        pep_indices: List[torch.Tensor],
        gt_pair: torch.Tensor,
        type_logits: torch.Tensor,
        type_labels: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        losses = []
        bs = len(pep_indices)
        pred_types = torch.argmax(type_logits, dim=-1)
        for b in range(bs):
            pep_idx = pep_indices[b]
            if pep_idx.numel() == 0:
                continue
            if self.ignore_ht_pair_loss and CYCLE_TYPES[int(type_labels[b].item())] == 'HT':
                continue
            if self.pair_loss_requires_type_match and pred_types[b] != type_labels[b]:
                continue
            local_pair = gt_pair[b]
            i, j = int(local_pair[0].item()), int(local_pair[1].item())
            if i >= pep_idx.numel() or j >= pep_idx.numel():
                continue
            gi, gj = pep_idx[i], pep_idx[j]
            prob = s_ij_full[gi, gj]
            losses.append(-torch.log(prob + eps))
        if len(losses) == 0:
            return torch.tensor(0.0, device=s_ij_full.device)
        return torch.stack(losses).mean()

    def _pair_metrics(
        self,
        s_ij_full: torch.Tensor,
        pep_indices: List[torch.Tensor],
        gt_pair: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        top1, top5, top10 = [], [], []
        for b, pep_idx in enumerate(pep_indices):
            if pep_idx.numel() == 0:
                continue
            s_pp = s_ij_full[pep_idx][:, pep_idx].view(-1)
            k = min(s_pp.numel(), 10)
            topk = torch.topk(s_pp, k=k).indices
            target = gt_pair[b]
            target_flat = int(target[0].item()) * pep_idx.numel() + int(target[1].item())
            top1.append((topk[:1] == target_flat).any())
            top5.append((topk[: min(5, k)] == target_flat).any())
            top10.append((topk[: min(10, k)] == target_flat).any())
        if len(top1) == 0:
            return {'pair_top1': torch.tensor(0.0), 'pair_top5': torch.tensor(0.0), 'pair_top10': torch.tensor(0.0)}
        return {
            'pair_top1': torch.stack(top1).float().mean(),
            'pair_top5': torch.stack(top5).float().mean(),
            'pair_top10': torch.stack(top10).float().mean(),
        }

    @oom_decorator
    def forward(
        self,
        X,
        S,
        mask,
        position_ids,
        lengths,
        atom_mask,
        L=None,
        c_star=None,
        gt_pair=None,
        peptide_length=None,
        **kwargs,
    ):
        S_true = S
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)

        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H

        if self.train_structure:
            X = X.clone()
            S_pep = S_true[mask] if S_true is not None else None
            try:
                X[mask] = self.autoencoder._fill_latent_channels(Z, S_pep)
            except TypeError:
                X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, : self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel :] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        type_labels = self._map_type_label(c_star) if c_star is not None else None

        loss_dict, pair_out = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            S=S_true,
            position_ids=position_ids,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence,
            type_labels=type_labels,
        )

        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        if type_labels is not None and gt_pair is not None:
            pair_loss = self._pair_loss(pair_out['s_ij_full'], pair_out['pep_indices'], gt_pair, pair_out['type_logits'], type_labels)
            loss = loss + self.w_pair * pair_loss
            loss_dict['Cyc/Pair'] = pair_loss

            if self.enable_type_loss:
                type_loss = F.cross_entropy(pair_out['type_logits'], type_labels)
                loss = loss + self.w_type * type_loss
                loss_dict['Cyc/Type'] = type_loss

            type_acc = (torch.argmax(pair_out['type_logits'], dim=-1) == type_labels).float().mean()
            loss_dict['Cyc/TypeAcc'] = type_acc
            pair_metrics = self._pair_metrics(pair_out['s_ij_full'], pair_out['pep_indices'], gt_pair)
            loss_dict['Cyc/PairTop1'] = pair_metrics['pair_top1']
            loss_dict['Cyc/PairTop5'] = pair_metrics['pair_top5']
            loss_dict['Cyc/PairTop10'] = pair_metrics['pair_top10']

        return loss, loss_dict

    def set_consec_dist(self, mean: float, std: float):
        self.consec_dist_mean = mean
        self.consec_dist_std = std

    def latent_geometry_guidance(self, X, mask_generate, batch_ids, tolerance=3, **kwargs):
        from models.LDM.energies.dist import dist_energy
        assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
               'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
        return dist_energy(
            X, mask_generate, batch_ids,
            self.consec_dist_mean, self.consec_dist_std,
            tolerance=tolerance, **kwargs
        )

    @torch.no_grad()
    def sample(
        self,
        X,
        S,
        mask,
        position_ids,
        lengths,
        atom_mask,
        L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0,
            'autoencoder_n_iter': 1,
            't_hard': 0,
            'return_pair_trajectory': False,
        },
        return_tensor=False,
        optimize_sidechain=True,
        return_pair: bool = False,
        **kwargs,
    ):
        self.autoencoder.eval()
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = 0

        if self.train_structure:
            X = X.clone()
            X[mask] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, : self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel :] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt = dict(sample_opt)
        autoencoder_n_iter = sample_opt.pop('autoencoder_n_iter', 1)
        t_hard = sample_opt.pop('t_hard', 0)
        return_pair_trajectory = sample_opt.pop('return_pair_trajectory', False)
        use_cg_guidance = sample_opt.pop('use_cg_guidance', False)
        property_ckpt = sample_opt.pop('property_ckpt', None)
        property_ckpts = sample_opt.pop('property_ckpts', None)
        guidance_w = sample_opt.pop('guidance_w', None)
        guidance_w_map = sample_opt.pop('guidance_w_map', None)
        guidance_w_min = sample_opt.pop('guidance_w_min', None)
        guidance_w_max = sample_opt.pop('guidance_w_max', None)
        lambda_max = sample_opt.pop('lambda_max', 0.0)
        t_guidance_start = sample_opt.pop('t_guidance_start', 0)
        t_guidance_end = sample_opt.pop('t_guidance_end', 0)
        guidance_grad_norm_balance = sample_opt.pop('guidance_grad_norm_balance', False)
        guidance_grad_norm_eps = sample_opt.pop('guidance_grad_norm_eps', 1e-6)
        guidance_grad_norm_ref = sample_opt.pop('guidance_grad_norm_ref', 'mean')

        if 'energy_func' in sample_opt:
            if sample_opt['energy_func'] is None:
                pass
            elif sample_opt['energy_func'] == 'default':
                sample_opt['energy_func'] = self.latent_geometry_guidance

        if property_ckpts is None:
            property_ckpts = property_ckpt

        ckpt_keys = None
        ckpt_paths = None
        if property_ckpts:
            if isinstance(property_ckpts, str):
                if ',' in property_ckpts:
                    ckpt_paths = [p.strip() for p in property_ckpts.split(',') if p.strip()]
                else:
                    ckpt_paths = [property_ckpts]
            elif isinstance(property_ckpts, (list, tuple)):
                ckpt_paths = list(property_ckpts)
            elif isinstance(property_ckpts, dict):
                ckpt_keys = list(property_ckpts.keys())
                ckpt_paths = [property_ckpts[k] for k in ckpt_keys]
            else:
                raise TypeError(f'Unsupported property_ckpts type: {type(property_ckpts)}')

        if use_cg_guidance and ckpt_paths is not None:
            from apcyc.guidance import load_property_predictor, load_property_predictors
            ckpt_tag = tuple(ckpt_paths)
            if self.property_predictor is None or getattr(self, '_property_predictor_tag', None) != ckpt_tag:
                if len(ckpt_paths) == 1:
                    self.property_predictor = load_property_predictor(ckpt_paths[0], device=X.device)
                else:
                    self.property_predictor = load_property_predictors(ckpt_paths, device=X.device, keys=ckpt_keys)
                self._property_predictor_tag = ckpt_tag

        if guidance_w is None and guidance_w_map is not None and use_cg_guidance:
            keys = getattr(self.property_predictor, 'property_keys', None)
            if keys:
                guidance_w = [guidance_w_map.get(k, 1.0) for k in keys]
            else:
                guidance_w = [guidance_w_map[k] for k in guidance_w_map]

        if isinstance(guidance_w, str):
            guidance_w = [float(x) for x in guidance_w.split(',') if x.strip() != '']

        traj, pair_state = self.diffusion.sample(
            H_0, X, position_embedding, mask, lengths, atom_embeddings, atom_mask, S,
            position_ids=position_ids, L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence,
            t_hard=t_hard,
            return_pair_trajectory=return_pair_trajectory,
            use_cg_guidance=use_cg_guidance,
            property_predictor=self.property_predictor,
            guidance_w=guidance_w,
            guidance_w_min=guidance_w_min,
            guidance_w_max=guidance_w_max,
            lambda_max=lambda_max,
            t_guidance_start=t_guidance_start,
            t_guidance_end=t_guidance_end,
            guidance_grad_norm_balance=guidance_grad_norm_balance,
            guidance_grad_norm_eps=guidance_grad_norm_eps,
            guidance_grad_norm_ref=guidance_grad_norm_ref,
            **sample_opt,
        )
        X_0, H_0 = traj[0]
        X_0, H_0 = X_0[mask][:, : self.autoencoder.latent_n_channel], H_0[mask]

        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_laten_H=H_0, given_latent_X=X_0, return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain,
            n_iter=autoencoder_n_iter
        )

        if not return_pair:
            return batch_X, batch_S, batch_ppls

        type_logits = pair_state['type_logits']
        s_ij_full = pair_state['s_ij_full']
        pep_indices = pair_state['pep_indices']
        type_hat = torch.argmax(type_logits, dim=-1)
        pair_hat = []
        for b, pep_idx in enumerate(pep_indices):
            if pep_idx.numel() == 0:
                pair_hat.append(torch.tensor([0, 0], device=s_ij_full.device))
                continue
            s_pp = s_ij_full[pep_idx][:, pep_idx]
            flat_idx = torch.argmax(s_pp.view(-1))
            i = int(flat_idx // s_pp.size(1))
            j = int(flat_idx % s_pp.size(1))
            pair_hat.append(torch.tensor([i, j], device=s_ij_full.device))
        pair_hat = torch.stack(pair_hat, dim=0)

        return batch_X, batch_S, batch_ppls, {
            'type_logits': type_logits,
            'type_hat': type_hat,
            'pair_hat': pair_hat,
            'pair_traj': pair_state.get('pair_traj'),
        }
