# Properties and evaluation

Use the same raw or relaxed structure stage for all measurements in an evaluation run.

## Collect external properties

The collector accepts inference/export manifests or JSONL rows with `id`, `pdb` and `peptide_chain`. Paths resolve from the manifest. IDs must be unique filename-safe strings; sequences must match PDBs. Structures require one model, complete peptide N/CA/C/O backbones and finite coordinates. Prepared copies, FASTA, peptide PDBs/ZIPs, logs and `status.json` go into a separate output directory. Invalid structures retain missing rows and never enter tool inputs.

| Stage | Outputs | Required installation |
| --- | --- | --- |
| Basic | sequence, GRAVY | Biopython |
| CamSol | camsol_score | CamSol account, Firefox/geckodriver |
| Proteolysis | cycpep_proteolysis_sumscore | ProsperousPlus and its models |
| Immunogenicity | immune_el, immune_im | BigMHC FASTA wrapper and models |
| Hydrophobicity | logP, TPSA, rTPSA | CPSea `water_or_oil.py`, RDKit |
| Vina | vina_score | CPSea wrapper, Vina, MGLTools |
| Rosetta | rosetta_dG, rosetta_total_score | CPSea evaluation runner, Rosetta binary |

Set tool paths/environment names from `configs/evaluation/properties.yaml` and `.env.example`. Tools can use separate environments; credentials are not stored in manifests.

```bash
python -m pipelines.preflight_properties --config configs/evaluation/properties.yaml \
  --out outputs/dependencies.json
python -m pipelines.collect_properties --manifest /path/to/data/inputs.jsonl \
  --config configs/evaluation/properties.yaml --out-dir outputs/properties --only basic
# Run configured external stages:
python -m pipelines.collect_properties --manifest /path/to/data/inputs.jsonl \
  --config configs/evaluation/properties.yaml --out-dir outputs/properties-full \
  --run-external --strict
```

Dependency preflight checks executable/files/environment names, not service authentication or prediction. Unselected/disabled stages are skipped. Completed outputs are cached only when source/prepared inputs, commands, configuration, dependencies and artifact hashes match. Partial stages retry. Unexpected/duplicate IDs fail; missing, invalid or nonfinite values stay missing. `--strict` makes incomplete selected stages fail.

To import a CSV, replace a stage command with `import: /path/to/result.csv`, preserving its column mapping. Source results are copied and hashed; rows from rejected structures stay missing. [Frozen surrogate predictions](INFERENCE.md#frozen-surrogate-scoring) are a separate route.

ProsperousPlus uses five proteases, 8-mer windows, stride 2, final-window inclusion and at most 30 windows. BigMHC uses HLA-A*02:01, direct 8–11 scoring and top-three window aggregation for longer peptides. Rosetta uses bundled `configs/evaluation/rosetta.xml`: ref2015 full-complex total score plus interface `dg`, without repacking or relaxation. Its `.protocol.json` records scoring function, XML/executable/structure hashes and scores. Total score is retained separately from the five surrogate objectives; aggregation and toxicity are excluded.

External stages use isolated process groups with timeout and child cleanup on Linux/macOS. Wrappers must not detach workers; Windows cleanup covers only the direct process. Service-backed availability requires a successful live run.

## Normalize scores

```bash
python -m pipelines.score_properties --input /path/to/data/raw.csv   --fit-ids /path/to/data/train_ids.txt --save-normalizer outputs/normalizer.json   --output outputs/scored.csv
python -m pipelines.score_properties --input /path/to/data/generated_raw.csv   --normalizer outputs/normalizer.json --output outputs/generated_scored.csv
```

| Higher-is-better objective | Definition |
| --- | --- |
| Solubility | CamSol unchanged |
| Proteolysis | Negative raw sumscore |
| Affinity | Negative mean of z-scored Rosetta dG and Vina |
| Pproxy | z(logP) + z(GRAVY) - z(TPSA) - z(rTPSA) |
| Sfinal | -log(EL) - log(IM) |

Fit only on explicit training IDs and reuse the saved statistics. Variance uses `ddof=0`; zero variance maps finite values to zero. Missing terms propagate. Inputs, scored CSV and saved normalizer must use distinct paths, including links. Applying a normalizer does not accept `--save-normalizer`.

## Measure and evaluate

```bash
python -m pipelines.measure_paper --manifest outputs/sample/predictions.jsonl   --properties outputs/scored.csv   --rosetta-protocol outputs/properties-full/stages/rosetta/results.protocol.json   --out-dir outputs/measurements
python -m pipelines.evaluate_paper --measurements outputs/measurements/measurements.csv   --conventions outputs/measurements/conventions.json --distances outputs/measurements/distances.json   --output outputs/evaluation.json --require-complete
```

Candidate IDs must match exactly between the manifest and CSV; absent manifest IDs use the PDB stem. Each candidate needs `receptor_id` (or `sample_id`), `pdb`, `peptide_chain`, ordered distinct `pair_hat` and `pair_index_base: 0`. Sequences must agree across PDB, manifest and CSV. Measurements require finite CA coordinates, one peptide length per receptor, and identical receptor CA identities/coordinates within 0.003 Å. Rosetta metadata must bind the same IDs, structure hashes and scores. Outputs cannot overwrite inputs.

| Metric | Definition |
| --- | --- |
| Cyclization success | Selected-residue CA distance in [3, 8] Å |
| Energy success | Interface Rosetta dG < 0 |
| Valid sample | Both success conditions pass |
| Sequence distance | 1 - global BLOSUM62 score / geometric mean of self-scores; zero gap scores |
| Structure distance | Equal-length, residue-order peptide CA RMSD in the shared receptor frame; no peptide superposition or cyclic permutation |
| Clusters | Single linkage with inclusive cuts 0.4 for sequence distance and 4 Å for RMSD |
| D_seq / D_struct | Cluster count / valid sample count |
| D_co | sqrt(D_seq * D_struct) |
| Consistency | Uncorrected Cramér's V; both single-cluster partitions score 1, only one scores 0 |

Sequence/clustering conventions follow [PepGLAD](https://github.com/THUNLP-MT/PepGLAD/blob/main/evaluation/diversity.py). CA proximity is a paper statistic; chemical bond checks use [relaxation](INFERENCE.md#relaxation).

Success rates include every candidate and receptor. Property means first average valid samples within each receptor, then weight eligible receptors equally. Property metrics require at least one valid sample; diversity/consistency require two. Coverage reports eligible/reported/total receptors. An empty eligible set yields `null`; missing values on an eligible receptor make the aggregate missing and `--require-complete` fail.

Default property metrics are `rosetta_total_score`, `vina_score`, `rosetta_dG`, `solubility_camsol_score`, `cycpep_proteolysis_sumscore`, `Pproxy` and `Sfinal`; select others with `--metrics`. Total score requires scoring protocol metadata. Evaluation never refits normalization.

Precomputed evaluation CSVs use unique `id`, `receptor_id`, `cyclization_distance` and `rosetta_dG`. Distance JSON uses `apcyc.evaluation_distances.v1` with receptor-keyed `ids`, `sequence_distance` and `ca_rmsd`. Matrices cover all IDs exactly and must be finite, square, symmetric, nonnegative and zero-diagonal; sequence distances lie in [0, 1]. Clustering uses valid samples only. Reports retain conventions and input hashes.
