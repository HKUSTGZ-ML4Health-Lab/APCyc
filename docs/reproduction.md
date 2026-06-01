# Reproduction Notes

APCyc keeps the PepGLAD-compatible training stages:

1. Train the autoencoder.
2. Train the latent diffusion model.
3. Estimate latent-space distance statistics.
4. Generate structures and compute metrics.

The integrated runner is:

```bash
GPU=0 bash scripts/run_exp_pipe.sh \
  <experiment_name> \
  <autoencoder_config> \
  <ldm_config> \
  <latent_distance_config> \
  <test_config> \
  [mode]
```

`mode` is a four-character switch for:

```text
train_autoencoder / train_ldm / generate / evaluate
```

For example, `0011` skips training and only runs generation plus evaluation.

## PepBench

Download:

```bash
mkdir -p datasets
wget https://zenodo.org/records/13373108/files/train_valid.tar.gz?download=1 -O datasets/train_valid.tar.gz
wget https://zenodo.org/records/13373108/files/LNR.tar.gz?download=1 -O datasets/LNR.tar.gz
wget https://zenodo.org/records/13373108/files/ProtFrag.tar.gz?download=1 -O datasets/ProtFrag.tar.gz
```

Process:

```bash
tar zxvf datasets/train_valid.tar.gz -C datasets
tar zxvf datasets/LNR.tar.gz -C datasets
tar zxvf datasets/ProtFrag.tar.gz -C datasets

python -m scripts.data_process.process --index datasets/train_valid/all.txt --out_dir datasets/train_valid/processed
python -m scripts.data_process.process --index datasets/LNR/test.txt --out_dir datasets/LNR/processed
python -m scripts.data_process.process --index datasets/ProtFrag/all.txt --out_dir datasets/ProtFrag/processed

python -m scripts.data_process.split \
  --train_index datasets/train_valid/train.txt \
  --valid_index datasets/train_valid/valid.txt \
  --processed_dir datasets/train_valid/processed
```

## PepBDB

Download and process:

```bash
mkdir -p datasets/pepbdb
wget http://huanglab.phys.hust.edu.cn/pepbdb/db/download/pepbdb-20200318.tgz -O datasets/pepbdb.tgz
tar zxvf datasets/pepbdb.tgz -C datasets/pepbdb

python -m scripts.data_process.pepbdb \
  --index datasets/pepbdb/peptidelist.txt \
  --out_dir datasets/pepbdb/processed

python -m scripts.data_process.split \
  --train_index datasets/pepbdb/train.txt \
  --valid_index datasets/pepbdb/valid.txt \
  --test_index datasets/pepbdb/test.txt \
  --processed_dir datasets/pepbdb/processed
```

## APCyc Configs

Core APCyc configs:

```text
configs/apcyc/apcyc_ae.yaml
configs/apcyc/apcyc_ldm.yaml
configs/apcyc/apcyc_ldm_sample.yaml
configs/apcyc/property/*.yaml
```

Router configs:

```text
configs/router/paradigm/*.yaml
```

Use `scripts/check_ckpt_compat.py` before long sampling jobs when swapping
checkpoints across model variants.
