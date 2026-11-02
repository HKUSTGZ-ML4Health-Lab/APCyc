# Inference

A generator bundle embeds its AE, architecture, vocabulary and geometry calibration. Supply it with `--bundle`; no separate AE file is needed for sampling. Tensor-only formats `apcyc.state_dict.v1` and `apcyc.training_state_dict.v1` load with strict tensor checks. `/path/to/...` values below are placeholders.

## Download from Hugging Face

From the repository root, install the download client and fetch a generator bundle:

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="qiaoqian0214/APCyc",
    filename="weights/apcyc-legacy-vocab.pt",
    local_dir="outputs/models",
)
PY
python scripts/make_example.py --out outputs/input
python -m apcyc.sample \
  --bundle outputs/models/weights/apcyc-legacy-vocab.pt \
  --input outputs/input/manifest.jsonl \
  --config configs/inference/default.yaml \
  --out-dir outputs/sample --device cpu --seed 2021
python scripts/inspect_predictions.py outputs/sample
```

Use new output directories when repeating the example. The input is a synthetic interface fixture; replace it with real receptor pockets for design experiments. Set `revision` in `hf_hub_download` to a Hub commit SHA to pin a model version. The optional download client does not replace the installation requirements in [Installation](INSTALL.md).

## Inputs and sampling

Each input JSONL row specifies a receptor PDB, pocket and requested peptide length:

```json
{"sample_id":"target","complex_pdb":"target.pdb","pocket_json":"pocket.json","peptide_length":12,"receptor_chain":"R","peptide_chain":"L"}
```

Paths resolve from the manifest. Pocket entries are `[chain, [residue_number, insertion_code]]`, for example `["R", [42, " "]]`. Chains default to R/L; only receptor atoms condition generation. Length must be at least four and fit the checkpoint's position vocabulary. Supplied pockets and lengths are prior information.

```bash
python -m apcyc.sample --bundle /path/to/models/apcyc.pt \
  --input /path/to/data/inputs.jsonl --config configs/inference/default.yaml \
  --out-dir outputs/sample --device cuda --num-samples 1 --seed 2021
python scripts/inspect_predictions.py outputs/sample
```

Use `--device cpu` for CPU inference. To create a deterministic synthetic PDB, pocket and manifest, run `python scripts/make_example.py --out outputs/input`. This fixture tests the interface and is not a biological benchmark.

For the paper's Base/no-guidance ablation, use the same trained APCyc bundle with `configs/inference/default.yaml`; property guidance is disabled. The guided profiles use that bundle with trained compatible heads and different objective weights. A comparison with another generator checkpoint can also use `default.yaml` with its own `--bundle`.

The output directory must be empty. It receives PDBs, `predictions.jsonl` and `run.json` with seeds, options and code/asset hashes. `pair_hat` is zero-based and peptide-local; peptide PDB numbering starts at one. Expanded tokens are retained in JSON and mapped to base residue names in the PDB. Outputs are marked `raw_prediction_not_relaxed`.

## Property guidance

Copy a configuration and edit its checkpoint paths. Paths inside YAML resolve from that YAML's directory. Use `energy_func: null` when the bundle has no latent-distance calibration.

| Route | Configuration | Required model |
| --- | --- | --- |
| Generator features | `guided.yaml` | Heads bound to the exact generator SHA-256, feature implementation and `t_hard` |
| Decoded structure | `decoded_guidance.yaml` | Self-contained `property_scorer` with its own AE and heads |

Sampling configurations are under `configs/inference/`. For generator-feature heads, target names, normalization, edge dimensions and `apcyc.generator_features.v2` must match. `guidance_w_map` explicitly lists every loaded target, including zero weights. Changing the feature implementation requires new compatible heads.

The decoded route sets `property_guidance_mode: decoded_structure` and `property_scorer`, without `property_ckpts`. It requires codesign AEs, DDPM transitions and receptor covariance. It decodes a predicted clean latent state, re-encodes the structure with the scoring AE and differentiates through that path. Argmax residue identities remain fixed during each gradient evaluation. Guided manifests record normalized scores and update norms in scorer target order.

Guidance uses `z_next = z_base - lambda(t) * posterior_variance(t) * grad(E)`, with `E = -sum(w_k * score_k)`, optional gradient balancing and fixed context. Strengths must be finite; the window must include an effective update above the final zero-variance transition. Zero objective weights preserve ordinary sampling. Runtime scorer caches stay outside generator checkpoint tensors and refresh when file contents or the sampling device changes. The mean update follows [guided-diffusion](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py); clean-prediction guidance is motivated by [Universal Guidance](https://arxiv.org/abs/2302.07121).

Set `guidance_w_map` in a copy of `guided.yaml` for the desired objective. The four paper profiles use this target order: affinity, proteolysis, permeability, solubility, immunogenicity.

| Profile | Weights |
| --- | --- |
| affinity | 10, 0, 0, 0, 0 |
| protease | 8, 8, 0, 2, 0 |
| permeability | 8, 2, 10, 2, 0 |
| multi | 6, 4, 4, 3, 5 |

## Frozen surrogate scoring

```bash
python -m pipelines.predict_surrogates   --scorer /path/to/models/frozen-property-scorer.pt   --manifest /path/to/data/predictions.jsonl --out-dir outputs/scores --device cpu
```

Scoring rows require `pdb`; optional fields include `id`, `sequence`, R/L chain IDs, `pocket_json` and `residue_tokens`. IDs default to the PDB stem and must be unique nonempty strings. Sequences/tokens must match the PDB. Use one PDB model with complete N/CA/C/O backbones and finite coordinates. Without `pocket_json`, the full receptor is used. The CLI computes covariance `L`; Python scoring batches must supply it and include a peptide and visible receptor CA atoms in each graph.

The scorer uses deterministic AE means at clean time zero. Its CSV contains raw and `__normalized` predictions; `run.json` records provenance. These predictions are separate from [external property measurements](EVALUATION.md). Scorer export is described in [Training](TRAINING.md#model-export).

## Relaxation

```bash
python -m pipelines.postprocess --predictions outputs/sample/predictions.jsonl   --out-dir outputs/relaxed --relax-python "$RELAX_PYTHON"   --cpsea-relax-dir "$CPSEA_RELAX_DIR" --strict
```

Install CPSea `PostProcess/Relax`, its force-field XML, OpenMM and PDBFixer in the specified environment. Without `--relax-python`, only topology eligibility is checked and no accepted relaxed manifest is emitted.

HT uses terminal C–N; SS requires Cys–Cys separated by at least three sequence positions; ISO requires Lys–Asp/Asn. The sequence, type and sites are preserved. Acceptance checks complete backbones, finite coordinates, residue identity/numbering, original receptor coordinates, explicit CONECT bonds, CA handedness and severe heavy-atom contacts below 1 Å. Closure bounds are 1.20–1.50 Å for amide/isopeptide and 1.85–2.20 Å for disulfide. This is a conservative chemical check, not exhaustive stereochemical validation.

Only accepted structures enter the output manifest, with resolved pocket paths. Failures remain in `status.json`; `--strict` exits nonzero for rejected or unrelaxed samples. Choose raw or relaxed structures consistently for subsequent scoring and evaluation.

## Repeated-run check

```bash
export APCYC_BUNDLE=/path/to/models/apcyc.pt
export APCYC_INPUT=/path/to/data/inputs.jsonl
bash scripts/smoke.sh cpu outputs/smoke
```

`APCYC_CONFIG` selects another configuration. Optional comparison settings are `APCYC_COMPARISON_BUNDLE` and `APCYC_COMPARISON_CONFIG`. Reports record byte equality; CPU requires it by default, while CUDA reports differences from nondeterministic scatter reductions. Override with `APCYC_REQUIRE_BITWISE=1` or `0`. A passed execution check does not establish chemical validity or property improvement.
