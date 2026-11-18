from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
FEATURES_DIR = RESULTS_DIR / "features"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_results_dirs():
    for directory in (CHECKPOINT_DIR, METRICS_DIR, FEATURES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
