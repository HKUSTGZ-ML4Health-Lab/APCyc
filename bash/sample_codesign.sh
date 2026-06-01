#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-checkpoints/codesign.ckpt}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-outputs/codesign}"

python -m api.detect_pocket \
  --pdb assets/1ssc_A_B.pdb \
  --target_chains A \
  --ligand_chains B \
  --out assets/1ssc_A_pocket.json

CUDA_VISIBLE_DEVICES="${GPU}" python -m api.run \
  --mode codesign \
  --pdb assets/1ssc_A_B.pdb \
  --pocket assets/1ssc_A_pocket.json \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --length_min 8 \
  --length_max 15 \
  --n_samples 10
