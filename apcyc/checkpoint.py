"""Portable checkpoint bundles: tensor state, explicit architecture and vocabulary."""

import torch
from utils.file_utils import sha256

FORMAT = 'apcyc.state_dict.v1'


def load_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    from utils.checkpoint_utils import TRAINING_FORMAT, load_model_checkpoint
    if isinstance(bundle, dict) and bundle.get('format') == TRAINING_FORMAT:
        if bundle.get('model_class') != 'APCycLDMPepDesign':
            raise ValueError('Inference requires an APCyc LDM checkpoint, not a standalone AE.')
        model, payload = load_model_checkpoint(path, device)
        model._bundle_sha256 = sha256(path)
        model.eval()
        metadata = {k: v for k, v in payload.items() if k != 'state_dict'}
        metadata['provenance'] = {'source': TRAINING_FORMAT, 'progress': payload.get('progress', {})}
        return model, metadata
    if not isinstance(bundle, dict) or bundle.get('format') != FORMAT:
        raise ValueError('Use an exported APCyc state_dict bundle; legacy pickles require scripts/models/export_checkpoint.py.')
    from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
    from data.format import VOCAB
    from apcyc.model import APCycLDMPepDesign
    cfg = bundle.get('cyc_config', {})
    set_config(cfg)
    apply_enlarged_vocab(cfg)
    vocabulary = [VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))]
    if vocabulary != bundle['vocabulary']:
        raise ValueError('Vocabulary differs from the checkpoint. Load different vocabulary variants in separate processes.')
    model = APCycLDMPepDesign(**bundle['model_config'])
    model.load_state_dict(bundle['state_dict'], strict=True)
    for key in ['consec_dist_mean', 'consec_dist_std']:
        setattr(model, key, bundle['geometry'].get(key))
    model._bundle_sha256 = sha256(path)
    model.to(device).eval()
    return model, {k: v for k, v in bundle.items() if k != 'state_dict'}
