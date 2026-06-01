# APCyc Evaluation

This module provides offline evaluation for target-conditioned cyclic peptide generation. It aggregates metrics per target (receptor) and then summarizes across targets.

## Quick Start

```bash
python -m apcyc.evaluation.eval_runner \
  --manifest /path/to/apcyc_predictions.jsonl \
  --outdir results/eval_run1 \
  --config apcyc/evaluation/configs/eval_default.yaml
```

`apcyc_predictions.jsonl` is produced by `apcyc_sample.py` and includes:
- `receptor_id`
- `sample_id`
- `type_hat`
- `pair_hat`
- `pdb`

## Outputs

- `metrics_per_sample.csv`
- `metrics_per_target.csv`
- `summary.json`
- `failures.json`
- `config_snapshot.json`

## Standalone MD Runner

Use `apcyc.evaluation.md_runner` with `apcyc/evaluation/configs/md_default.yaml` to generate Amber inputs or run MD separately from the main evaluator.

## MD Report (Figure/Table)

After MD finishes, you can generate RMSD trajectory plots and a Table 4-style CSV:

```bash
python -m apcyc.evaluation.md_report \
  --md_outdir results/md_run1 \
  --outdir results/md_report1 \
  --config apcyc/evaluation/configs/md_report.yaml
```

## External Dependencies (Optional)

Rosetta and FoldX are optional. When not available, the evaluator will skip them and record a failure reason.

### TM-score
- Provide `tm_score_bin` in the config, or place `TMscore`/`USalign` in PATH.
- The code also checks `ppflow/bin/TMscore/TMscore` if present.
- TM-score is computed on CA-only peptide chain PDBs generated on the fly.

### Rosetta (pyrosetta)
- Install PyRosetta and make sure it can be imported in the same environment.
- `rosetta.enabled` is `true` by default; it will be skipped if `pyrosetta` is missing.
- `rosetta.require_receptor` controls whether a receptor chain must exist to score.
- `rosetta_total_score` is computed as peptide-only by default (`total_score_scope: peptide`).
- `energy_success` is based on interface ΔG (< 0), not the total score.

### FoldX
- Install FoldX and set `foldx.bin_path` to the FoldX binary.
- `foldx.enabled` is `true` by default; it will be skipped if the binary is missing.
- FoldX requires both peptide and receptor chains; missing chains are reported as failures.


## KL Reference Sets

KL metrics are computed on the peptide chain only. Configure `kl.ref_sets`:
- `pepglad_train_valid` and `pepglad_lnr` use the peptide chain id from the index file (`chain_source: index_peptide_id`).
- `cpcore_all` uses a fixed peptide chain id (default `L`).

Precomputed reference histograms are cached under `kl.cache_dir` (default: `<outdir>/ref_cache`).

## Configuration Notes

- `aggregation`: choose `mean_over_valid` and/or `best_of_k`.
- `validity_filter`: when `true`, energy/diversity metrics are computed only on valid samples.
- `cyclization`: Cβ–Cβ distance window for cyclization success. If Cβ is missing, the fallback atom is used.
