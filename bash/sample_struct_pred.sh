#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-checkpoints/fixseq.ckpt}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-outputs/struct_pred}"
PEPTIDE_SEQ="${PEPTIDE_SEQ:-PYVPVHFDASV}"

python -m api.detect_pocket \
  --pdb assets/1ssc_A_B.pdb \
  --target_chains A \
  --ligand_chains B \
  --out assets/1ssc_A_pocket.json

CUDA_VISIBLE_DEVICES="${GPU}" python -m api.run \
  --mode struct_pred \
  --pdb assets/1ssc_A_B.pdb \
  --pocket assets/1ssc_A_pocket.json \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --peptide_seq "${PEPTIDE_SEQ}" \
  --n_samples 10
