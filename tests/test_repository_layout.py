from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_entrypoints_exist():
    for path in [
        'README.md',
        'env.yaml',
        'setup.py',
        'api/run.py',
        'api/detect_pocket.py',
        'scripts/train.sh',
        'scripts/run_exp_pipe.sh',
        'apcyc/evaluation/configs/eval_default.yaml',
    ]:
        assert (ROOT / path).exists(), path


def test_example_assets_exist():
    for path in [
        'assets/1ssc_A_B.pdb',
        'assets/1ssc_A_pocket.json',
        'assets/teaser.png',
    ]:
        assert (ROOT / path).exists(), path
