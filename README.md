# APCyc: Property-Informed Design of Cyclic Peptides via Automated Cyclization

<div align="center">

Yifan Zhao, Lang Qin, Jintai Chen<sup>†</sup>

The Hong Kong University of Science and Technology (Guangzhou)

[![Project Page](https://img.shields.io/badge/Project%20Page-APCyc-blueviolet?logo=githubpages&style=flat-square)](https://hkustgz-ml4health-lab.github.io/APCyc/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader&style=flat-square)](https://hkustgz-ml4health-lab.github.io/APCyc/assets/apcyc/paper.pdf)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/qiaoqian0214/APCyc)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-yellow?logo=python&style=flat-square)](docs/INSTALL.md)

</div>

Receptor-conditioned cyclic-peptide generation with autoencoder and diffusion training, cyclization prediction, five property objectives and optional relaxation.

Explore the method and design examples on the [project page](https://hkustgz-ml4health-lab.github.io/APCyc/), or read the [paper](https://hkustgz-ml4health-lab.github.io/APCyc/assets/apcyc/paper.pdf). The project page is maintained in [`website/`](https://github.com/HKUSTGZ-ML4Health-Lab/APCyc/tree/main/website) and deployed with GitHub Pages.

## Installation and guides

The tested environment is Linux with Python 3.12 and PyTorch 2.5.1. Clone the repository, then follow [Installation](docs/INSTALL.md) to install matching CPU or CUDA wheels and the unified requirements.

```bash
git clone https://github.com/HKUSTGZ-ML4Health-Lab/APCyc.git
cd APCyc
```

| Task | Guide |
| --- | --- |
| Install dependencies | [Installation](docs/INSTALL.md) |
| Generate, guide, score and relax peptides | [Inference](docs/INFERENCE.md) |
| Prepare training data, train and export models | [Training](docs/TRAINING.md) |
| Collect properties and evaluate candidates | [Evaluation](docs/EVALUATION.md) |
| Component licenses and external tools | [Third-party components](docs/THIRD_PARTY.md) |

## Model checkpoints

Download pretrained generator bundles and property-scoring models from [Hugging Face: qiaoqian0214/APCyc](https://huggingface.co/qiaoqian0214/APCyc).

Generator bundles embed their AE, architecture, vocabulary and geometry calibration. Download a generator bundle before running the quick start. For decoded-structure guidance, use the supplied frozen scorer; generator-feature guidance requires heads matched to the exact generator and feature implementation. See [Inference](docs/INFERENCE.md#property-guidance) for both routes.

## Quick start

Follow the CPU or CUDA setup in [Installation](docs/INSTALL.md), then run:

```bash
python scripts/make_example.py --out outputs/input
python -m apcyc.sample --bundle /path/to/models/apcyc-legacy-vocab.pt \
  --input outputs/input/manifest.jsonl --config configs/inference/default.yaml \
  --out-dir outputs/sample --device cuda --seed 2021
python scripts/inspect_predictions.py outputs/sample
```

The example script creates synthetic input for checking the interface. For design experiments, supply a manifest with real receptor pockets as described in [Inference](docs/INFERENCE.md#inputs-and-sampling).

The five objectives are affinity, proteolytic stability, permeability proxy, solubility and immunogenicity proxy. The source includes plain sampling, compatible generator-feature guidance, and frozen scoring with decoded-structure guidance.

## Training and evaluation

To train from scratch, follow [Training](docs/TRAINING.md): train the AE, freeze it while training the diffusion model, then freeze the selected generator while training the five property heads. The provided training configurations start these stages from random initialization unless a checkpoint is explicitly supplied.

[Evaluation](docs/EVALUATION.md) describes property collection, normalization and the paper evaluation protocol. Relaxation and external predictors require separate installations; configure them through `.env.example` and the evaluation YAML.

## Repository layout

```text
apcyc/                         APCyc model, sampling, datasets and property guidance
models/                        Shared autoencoder, denoiser and geometric layers
data/                          Structure formats, conversion, mmap access and batching
trainer/                       Training loops, validation and checkpoint saving
pipelines/                     Relaxation, property collection, scoring and evaluation
utils/                         Shared configuration, tensors, checkpoints and logging
configs/
  training/                    Autoencoder, diffusion and property-head training
  inference/                   Plain, generator-guided and decoded-guided sampling
  evaluation/                  External property tools and Rosetta protocol
scripts/                       Synthetic input, output inspection and source packaging
  data/                        Stored-structure export and split tools
  models/                      Generator, autoencoder and scorer export
tests/                         Regression tests with synthetic fixtures
docs/                          Installation, inference, training, evaluation and attribution
website/                       Project page, paper and public design demos
```

`apcyc/` builds the cyclic-peptide model from components in `models/`; both are used during APCyc training and inference. Configuration files contain runtime options, while model architecture and vocabulary are embedded in exported bundles.

## Development

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

## License and acknowledgements

The code is distributed under the [MIT license](LICENSE) and builds on [PepGLAD](https://github.com/THUNLP-MT/PepGLAD). See [Third-party components](docs/THIRD_PARTY.md) for component attribution and external tool requirements.
