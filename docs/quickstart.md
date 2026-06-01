# Quickstart

This page keeps the runnable commands short and explicit. All commands assume
that they are executed from the repository root.

## 1. Prepare The Environment

```bash
conda env create -f env.yaml
conda activate APCyc
```

For source-level development:

```bash
pip install -e .
pip install -r requirements/dev.txt
```

## 2. Prepare Checkpoints

Place released model weights under `checkpoints/`:

```text
checkpoints/codesign.ckpt
checkpoints/fixseq.ckpt
```

The directory is git-kept, but checkpoint files are ignored by design.

## 3. Extract A Pocket

```bash
python -m api.detect_pocket \
  --pdb assets/1ssc_A_B.pdb \
  --target_chains A \
  --ligand_chains B \
  --out assets/1ssc_A_pocket.json
```

## 4. Generate Cyclic Peptides

```bash
CUDA_VISIBLE_DEVICES=0 python -m api.run \
  --mode codesign \
  --pdb assets/1ssc_A_B.pdb \
  --pocket assets/1ssc_A_pocket.json \
  --ckpt checkpoints/codesign.ckpt \
  --out_dir outputs/codesign \
  --length_min 8 \
  --length_max 15 \
  --n_samples 10
```

## 5. Run Source Checks

```bash
make syntax
```

This only checks Python syntax. It does not load model weights, datasets, or
CUDA-dependent libraries.
