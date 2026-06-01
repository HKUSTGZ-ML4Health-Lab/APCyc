#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Check checkpoint compatibility against a model config.
Prints missing/unexpected/mismatched keys and a brief summary.
"""
import argparse
import json
import os
import sys
from typing import Dict, Tuple

import torch
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.logger import print_log
from utils import register as R
from apcyc.cyc_enhance import apply_enlarged_vocab, set_config
import models  # noqa: F401  register modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check checkpoint vs model config.")
    parser.add_argument("--config", required=True, help="Model config yaml.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--autoencoder_ckpt", default=None, help="Override autoencoder_ckpt path.")
    parser.add_argument("--show", type=int, default=20, help="Show up to N keys for each list.")
    parser.add_argument("--out", type=str, default=None, help="Optional json output.")
    return parser.parse_args()


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _extract_state_dict(model, trained):
    if isinstance(trained, dict):
        if "state_dict" in trained:
            state_dict = trained["state_dict"]
        elif "model" in trained:
            state_dict = trained["model"]
        else:
            state_dict = trained
    else:
        if hasattr(trained, "module"):
            trained = trained.module
        if hasattr(trained, "state_dict"):
            state_dict = trained.state_dict()
        else:
            state_dict = trained

    if isinstance(model, models.AutoEncoder):
        if any(k.startswith("autoencoder.") for k in state_dict):
            state_dict = _strip_prefix(state_dict, "autoencoder.")
    return state_dict


def _compare_state_dicts(
    model_state: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, int], Dict[str, list]]:
    missing = []
    unexpected = []
    mismatched = []
    matched = []

    for key, val in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(val.shape):
            mismatched.append((key, tuple(val.shape), tuple(model_state[key].shape)))
            continue
        matched.append(key)

    for key in model_state.keys():
        if key not in state_dict:
            missing.append(key)

    summary = {
        "matched": len(matched),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "mismatched": len(mismatched),
    }
    details = {
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "matched": matched,
    }
    return summary, details


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    if args.autoencoder_ckpt:
        config.setdefault("model", {})
        config["model"]["autoencoder_ckpt"] = args.autoencoder_ckpt
    cyc_cfg = config.get("cyc_enhance", {})
    if cyc_cfg:
        set_config(cyc_cfg)
        apply_enlarged_vocab(cyc_cfg)

    model = R.construct(config["model"])
    model_state = model.state_dict()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = _extract_state_dict(model, ckpt)

    summary, details = _compare_state_dicts(model_state, state_dict)

    print_log(f"Matched: {summary['matched']}")
    print_log(f"Missing: {summary['missing']}")
    print_log(f"Unexpected: {summary['unexpected']}")
    print_log(f"Mismatched: {summary['mismatched']}")

    show_n = max(0, args.show)
    if show_n:
        if details["missing"]:
            print_log(f"Missing (first {show_n}): {details['missing'][:show_n]}")
        if details["unexpected"]:
            print_log(f"Unexpected (first {show_n}): {details['unexpected'][:show_n]}")
        if details["mismatched"]:
            print_log(f"Mismatched (first {show_n}): {details['mismatched'][:show_n]}")

    if args.out:
        payload = {
            "summary": summary,
            "missing": details["missing"],
            "unexpected": details["unexpected"],
            "mismatched": details["mismatched"],
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print_log(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()
