# Training

Install [dependencies](INSTALL.md) and copy the required recipe before changing paths.

| Component | Entry point | Recipes under `configs/training/` |
| --- | --- | --- |
| Autoencoder | `train.py` | `autoencoder.yaml` |
| Diffusion and cyclization | `train.py` | `diffusion.yaml` |
| Property heads | `python -m apcyc.train_property` | `property.yaml` |

The configuration selects the model. The AE and diffusion recipes use cyclization-aware vocabulary, sequence/coordinate reconstruction with physical geometry losses, invariant KL and equivariant CA anchoring. Physical loss weights are 1 for CA/bond terms and 0.5 for sidechain chi angles. Diffusion uses noise MSE plus type/pair classification, training times 1…T and visible peptide latent/receptor CA anchors. Hard topology features are shared by training and sampling; final sites are computed from the returned sequence.

## Training from scratch

1. Train the randomly initialized AE with `autoencoder.yaml`.
2. Set `model.autoencoder_ckpt` in `diffusion.yaml` to the selected AE checkpoint from step 1. Train the randomly initialized diffusion and topology modules with that AE frozen.
3. Set `property_train.generator_bundle` in `property.yaml` to the selected generator checkpoint from step 2. Train randomly initialized property heads with the generator frozen.

The supplied recipes do not set `load_ckpt` or `predictor_ckpt`. Leave them unset for this workflow. The AE and generator paths connect the training stages; they do not require an existing pretrained APCyc model. Collect and normalize property labels as described in [Evaluation](EVALUATION.md) before training the heads.

## Data

Set `dataset.{train,valid}.mmap_dir`, `jsonl_path` and `specify_index` in AE/LDM YAML. Mmap indexes contain IDs and byte offsets. `CPCoreDataset` instead accepts PDB files and plain ID lists. Both return flat variable-length batches with concatenated residue tensors and per-complex `lengths`; PDB complexity includes the selected receptor and peptide.

Metadata IDs must be unique and agree with the indexes and property CSV; dots are preserved. `peptide_length` must match the structure. Supervision uses `HT`, `SS` or `ISO` and two distinct, in-range, zero-based peptide-local `gt_pair` sites. Property tables use complete cases for the selected targets; duplicate IDs fail and both filtered splits must remain nonempty.

Export stored structures and pockets without changing mmap data:

```bash
python scripts/data/export_cpcore.py --mmap-dir /path/to/data/cpcore/mmap \
  --ids /path/to/data/ids.txt --out-dir outputs/exported
```

For split audits, pass existing training/validation ID lists to `scripts/data/cluster_split.py`.

## Training commands

```bash
python train.py --config /path/to/ae.yaml --gpus 0
python train.py --config /path/to/ldm.yaml --gpus 0
python -m apcyc.train_property --config /path/to/property.yaml   --mmap_dir /path/to/data/cpcore/mmap --jsonl_path /path/to/data/metadata.jsonl   --train_index /path/to/data/train_index.txt --valid_index /path/to/data/valid_index.txt   --property_csv /path/to/data/scored.csv --gpus 0
```

Set `load_ckpt: null` for fresh initialization. Warm starts load parameters and saved geometry, not optimizer/scheduler/RNG state. Vocabulary must match. `strict: true` requires exact tensor shapes and cannot combine with `skip_mismatch: true`; partial initialization explicitly uses `strict: false`.

Recipe `batch_size_scope: per_device` applies to each GPU; a dynamic loader item is a packed set bounded by `ubound_per_batch`. Packing preserves the selected epoch list, uses `type_seed + epoch` and rejects oversized/invalid costs. The distributed sampler refreshes partition sizes and retains PyTorch's training padding policy. Global batch sizes must divide the world size.

`grad_interval` averages microbatches, including the actual final partial window; batch schedulers step per optimizer update. OOM/nonfinite checks stop the affected update. Validation uses a sample-weighted metric reduced across ranks; default validation is replicated, LDM restores CPU/CUDA RNG, and its selection criterion is `Loss`. Early stopping tracks the best validation metric and stops within the current epoch when patience is exhausted. A final-step validation also satisfies the epoch validation.

CLI property arguments override YAML, including explicit parser-default values; unknown keys fail. AE/LDM dotted overrides preserve types, for example `--model.autoencoder_no_randomness=false`. The property entrypoint supports one GPU with separate heads or one multitask head.

## Property heads

Set `property_train.generator_bundle` to the exact frozen generator. Heads retain target order, mean/std, generator SHA-256, feature implementation SHA-256 and `t_hard`; inference requires the same feature contract. The paper chemical mask uses AE-decoded identities and the stored hard-topology schedule.

Default targets are `affinity, cycpep_proteolysis_sumscore, Pproxy, solubility_camsol_score, Sfinal`. Custom targets use `--property_keys`. Missing/duplicate columns fail. Standard deviations use population variance and a positive floor; existing checkpoints retain saved statistics. MAE/RMSE aggregate sample errors. Supplied separate checkpoints must all exist, use distinct filenames and match their targets; validation-only mode requires a complete set.

## Zero-update preflight

Use small copied configurations and new output directories. Each rank needs the requested number of batches.

```bash
python train.py --config /path/to/small-ae.yaml --gpus -1   --preflight-batches 2 --preflight-out outputs/preflight-ae
python train.py --config /path/to/small-ldm.yaml --gpus -1   --preflight-batches 2 --preflight-out outputs/preflight-ldm
python -m apcyc.train_property --config /path/to/property.yaml   --mmap_dir /path/to/data/cpcore/mmap --jsonl_path /path/to/data/metadata.jsonl   --train_index /path/to/data/train_index.txt --valid_index /path/to/data/valid_index.txt   --property_csv /path/to/data/scored.csv --gpus -1   --preflight-batches 2 --preflight-out outputs/preflight-properties
```

For multi-GPU AE/LDM checks, launch `torch.distributed.run --standalone --nproc_per_node=2` with `--gpus 0 1`. Preflight checks finite gradients, unchanged parameters and exact save/reload with `optimizer_steps: 0`. Its checkpoints are diagnostic, not trained models; execution checks do not establish convergence or generation quality.

## Model export

AE/LDM saves use `apcyc.training_state_dict.v1`. Diffusion checkpoints embed their AE and can be passed directly to inference. Models without latent-distance calibration use `energy_func: null`.

```bash
python scripts/models/export_autoencoder.py --checkpoint /path/to/models/apcyc.pt \
  --out /path/to/models/autoencoder.pt
```

Trusted Python-object generators use `scripts/models/export_checkpoint.py` with their original LDM/AE YAML and explicit `--trust-legacy-pickle`; trusted AE objects also require their original `--config`. Ordinary loading uses `weights_only=True`. Export destinations and generator JSON sidecars must be new files; exact tensors are verified before publication.

The reconstructed frozen scorer requires its scoring AE, an explicit zero-pair context projection and matching heads. Heads trained with `apcyc.train_property` carry a generator-feature contract and use `guided.yaml` with their original generator; they cannot be exported into a reconstructed context.

```bash
python scripts/models/export_property_scorer.py \
  --autoencoder /path/to/models/scoring-ae.pt --context /path/to/models/context.pt \
  --head affinity=/path/to/models/affinity.pt \
  --head cycpep_proteolysis_sumscore=/path/to/models/proteolysis.pt \
  --head Pproxy=/path/to/models/permeability.pt \
  --head solubility_camsol_score=/path/to/models/solubility.pt \
  --head Sfinal=/path/to/models/immunogenicity.pt \
  --out /path/to/models/frozen-property-scorer.pt
```

The context artifact binds AE-source and head hashes. The scorer embeds its own AE, projection, 32 distance features, 53 zero pair features, exact head tensors and normalization. This representation is separate from generator-feature heads. See [Inference](INFERENCE.md) for both guidance routes and [Evaluation](EVALUATION.md) for score normalization.
