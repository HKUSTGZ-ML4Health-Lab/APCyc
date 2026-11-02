"""Tensor-only training checkpoints shared by AE and diffusion entrypoints."""
from copy import deepcopy
from pathlib import Path

import torch

TRAINING_FORMAT = 'apcyc.training_state_dict.v1'


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def configure_vocabulary(payload):
    from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
    from data.format import VOCAB
    cfg = payload.get('cyc_config', {})
    set_config(cfg)
    apply_enlarged_vocab(cfg)
    actual = [VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))]
    if 'vocabulary' in payload and actual != payload['vocabulary']:
        raise ValueError('Checkpoint vocabulary mismatch; load vocabulary variants in separate processes.')


def autoencoder_config(model, configured=None):
    cfg = deepcopy(configured if configured is not None else getattr(model, '_checkpoint_model_cfg', None))
    if cfg is None:
        raise ValueError('AE architecture is missing; supply the original AE model configuration for export.')
    cfg.pop('class', None)
    # Preserve the decoder actually constructed, independently of the global input vocabulary.
    for key in ['enable_enlarged_vocab', 'enable_latent_anchor_channels']:
        cfg[key] = bool(getattr(model, key, False))
    return cfg


def model_checkpoint(model, save_config, progress=None):
    from data.format import VOCAB
    model = unwrap_model(model)
    cfg = deepcopy(save_config['model'])
    name = cfg.pop('class')
    if name != type(model).__name__:
        raise ValueError(f'Checkpoint class mismatch: {name} versus {type(model).__name__}')
    if name == 'AutoEncoder':
        cfg = autoencoder_config(model, cfg)
    elif name == 'APCycLDMPepDesign':
        cfg['autoencoder_ckpt'] = autoencoder_config(model.autoencoder)
    else:
        raise ValueError(f'Unsupported portable training checkpoint class: {name}')
    return dict(
        format=TRAINING_FORMAT, model_class=name, model_cfg=cfg,
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        cyc_config=deepcopy(save_config.get('cyc_enhance', {})),
        vocabulary=[VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))],
        geometry={k: getattr(model, k, None) for k in ['consec_dist_mean', 'consec_dist_std']},
        training_config=deepcopy(save_config), progress=progress or {},
    )


def save_model_checkpoint(model, save_config, path, progress=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(model_checkpoint(model, save_config, progress), temporary)
    temporary.replace(path)


def restore_geometry(model, checkpoint):
    geometry = checkpoint.get('geometry', {}) if isinstance(checkpoint, dict) else {
        key: getattr(checkpoint, key, None) for key in ['consec_dist_mean', 'consec_dist_std']}
    for key in ['consec_dist_mean', 'consec_dist_std']:
        if hasattr(model, key) and key in geometry:
            setattr(model, key, geometry[key])


def load_model_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != TRAINING_FORMAT:
        raise ValueError('Expected an APCyc tensor-only training checkpoint.')
    configure_vocabulary(payload)
    import models  # noqa: F401
    import apcyc.model  # noqa: F401
    from utils import register as R
    cfg = deepcopy(payload['model_cfg'])
    model = R.construct({'class': payload['model_class'], **cfg})
    model.load_state_dict(payload['state_dict'], strict=True)
    if payload['model_class'] == 'AutoEncoder':
        model._checkpoint_model_cfg = cfg
    restore_geometry(model, payload)
    return model.to(device), payload


def load_autoencoder_checkpoint(path, strict=True):
    """Accept a saved AE or extract its exact state from a full APCyc bundle."""
    from models.autoencoder.model import AutoEncoder
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise ValueError('Expected a tensor-only AE checkpoint; convert trusted legacy files with scripts/models/export_autoencoder.py.')
    configure_vocabulary(payload)
    cfg = payload.get('model_config', payload.get('model_cfg'))
    if not isinstance(cfg, dict):
        raise ValueError('AE checkpoint requires model_cfg (or a full APCyc model_config).')
    state = payload['state_dict']
    if 'autoencoder_ckpt' in cfg:
        cfg = cfg['autoencoder_ckpt']
        if not isinstance(cfg, dict):
            raise ValueError('Full checkpoint must embed its AE architecture, not an external path.')
        state = {k.removeprefix('autoencoder.'): v for k, v in state.items() if k.startswith('autoencoder.')}
    cfg = deepcopy(cfg)
    cfg.pop('class', None)
    model = AutoEncoder(**cfg)
    model.load_state_dict(state, strict=strict)
    model._checkpoint_model_cfg = cfg
    return model
