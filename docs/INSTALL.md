# Installation

The tested environment is Linux, Python 3.12 and PyTorch 2.5.1. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install matching PyTorch and torch-scatter wheels for one platform.

CUDA 12.4:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=:all: torch-scatter==2.1.2   -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

CPU:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: torch-scatter==2.1.2   -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

Then install the unified dependency list and check the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

`requirements.txt` covers inference, training, tests, local property utilities and browser adapters. Install the matching wheels first to avoid compiling torch-scatter from source. macOS and other Python/PyTorch versions have not been validated.

External predictors, Rosetta, CPSea, Firefox/geckodriver and model files are installed separately. Configure their paths and credentials using `.env.example`; the programs do not automatically source it. See [Inference](INFERENCE.md) and [Evaluation](EVALUATION.md).
