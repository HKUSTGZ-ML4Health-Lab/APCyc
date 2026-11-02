#!/usr/bin/env python3
"""One-time export of a trusted historical model; never changes its tensors."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import yaml
import models  # legacy pickle class definitions
from apcyc.checkpoint import FORMAT, load_bundle, sha256
from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
from data.format import VOCAB


def export(args):
    if not args.trust_legacy_pickle:
        raise ValueError('The source is a Python pickle. Pass --trust-legacy-pickle only for a trusted checkpoint.')
    if args.out.exists() or args.out.resolve() == args.legacy.resolve():
        raise FileExistsError('Export destination must be a new file.')
    sidecar = args.out.with_suffix('.json')
    if args.out.suffix.lower() == '.json':
        raise ValueError('Checkpoint output and JSON sidecar must be different files.')
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError('Export JSON sidecar must be a new file.')
    config = yaml.safe_load(args.train_config.read_text())
    ae_config = dict(yaml.safe_load(args.ae_config.read_text())['model'])
    ae_config.pop('class', None)
    cyc = config.get('cyc_enhance', {})
    set_config(cyc)
    apply_enlarged_vocab(cyc)
    legacy = torch.load(args.legacy, map_location='cpu', weights_only=False)
    if not isinstance(legacy, models.APCycLDMPepDesign):
        raise TypeError('Source is not an APCyc LDM checkpoint.')
    # Preserve historical decoder capability instead of inventing new token outputs.
    ae_config['enable_enlarged_vocab'] = bool(getattr(legacy.autoencoder, 'enable_enlarged_vocab', False))
    ae_config['enable_latent_anchor_channels'] = bool(getattr(legacy.autoencoder, 'enable_latent_anchor_channels', False))
    model_config = dict(config['model'])
    model_config.pop('class', None)
    model_config['autoencoder_ckpt'] = ae_config
    state = {k: v.detach().cpu().clone() for k, v in legacy.state_dict().items()}
    nonfinite = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if nonfinite:
        raise ValueError(f'Non-finite checkpoint tensors: {nonfinite}')
    bundle = dict(format=FORMAT, model_config=model_config, cyc_config=cyc, state_dict=state,
                  vocabulary=[VOCAB.idx_to_symbol(i) for i in range(len(VOCAB))],
                  decoder_symbols=[VOCAB.idx_to_symbol(int(i)) for i in legacy.autoencoder.s_remap],
                  geometry={k: float(getattr(legacy, k)) if getattr(legacy, k, None) is not None else None
                            for k in ['consec_dist_mean', 'consec_dist_std']},
                  provenance=dict(legacy_name=args.legacy.name, legacy_sha256=sha256(args.legacy),
                                  train_config_sha256=sha256(args.train_config), ae_config_sha256=sha256(args.ae_config),
                                  retrained=False, selection=args.selection))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.apcyc-export-', dir=args.out.parent) as directory:
        temporary = Path(directory) / 'model.pt'
        torch.save(bundle, temporary)
        restored, info = load_bundle(temporary)
        actual = restored.state_dict()
        if set(actual) != set(state) or any(not torch.equal(state[k], actual[k]) for k in state):
            raise ValueError('Reconstructed model failed exact tensor equality.')
        report = dict(path=args.out.name, sha256=sha256(temporary), bytes=temporary.stat().st_size,
                      tensors=len(state), parameters=sum(p.numel() for p in restored.parameters()),
                      tensor_equality=True, finite_tensors=True, input_tokens=len(bundle['vocabulary']),
                      decoder_classes=len(bundle['decoder_symbols']), geometry=bundle['geometry'],
                      provenance=bundle['provenance'])
        metadata = Path(directory) / 'model.json'
        metadata.write_text(json.dumps(report, indent=2) + '\n')
        os.link(temporary, args.out)
        os.link(metadata, sidecar)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--ae-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', required=True, help='Evidence for selecting this checkpoint, not a claim of optimality.')
    parser.add_argument('--trust-legacy-pickle', action='store_true')
    export(parser.parse_args())
