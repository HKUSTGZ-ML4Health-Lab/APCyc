#!/usr/bin/env python3
"""Export an AE for diffusion training without changing any model tensors."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import yaml

from utils.checkpoint_utils import load_autoencoder_checkpoint, save_model_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--config', type=Path, help='AE YAML required for trusted historical Python objects.')
    p.add_argument('--trust-legacy-pickle', action='store_true')
    args = p.parse_args()
    if args.out.exists() or args.out.resolve() == args.checkpoint.resolve():
        raise FileExistsError('Export destination must be a new file.')
    if args.trust_legacy_pickle:
        if not args.config:
            p.error('--trust-legacy-pickle requires the original --config AE YAML')
        import models  # noqa: F401: historical pickle classes
        from utils.checkpoint_utils import configure_vocabulary
        config = yaml.safe_load(args.config.read_text())
        configure_vocabulary({'cyc_config': config.get('cyc_enhance', {})})
        original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        ae = getattr(original, 'autoencoder', original)
        if not isinstance(ae, models.AutoEncoder):
            raise TypeError('Legacy checkpoint must contain an AutoEncoder or an APCyc generator.')
    else:
        ae = load_autoencoder_checkpoint(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        config = {'model': {'class': 'AutoEncoder', **ae._checkpoint_model_cfg},
                  'cyc_enhance': payload.get('cyc_config', {})}
    from apcyc.checkpoint import sha256
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.apcyc-ae-export-', dir=args.out.parent) as directory:
        temporary = Path(directory) / 'autoencoder.pt'
        save_model_checkpoint(ae, config, temporary)
        payload = torch.load(temporary, map_location='cpu', weights_only=True)
        payload['provenance'] = {'source_sha256': sha256(args.checkpoint)}
        torch.save(payload, temporary)
        restored = load_autoencoder_checkpoint(temporary)
        if any(not torch.equal(v, restored.state_dict()[k]) for k, v in ae.state_dict().items()):
            raise RuntimeError('AE export failed exact tensor verification.')
        os.link(temporary, args.out)
    print(f'Exported and verified {len(ae.state_dict())} AE tensors: {args.out}')


if __name__ == '__main__':
    main()
