"""Shared seeding and path helpers so every run is reproducible."""
import os
import random
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "kode" / "results"
FIG_DIR = ROOT / "laporan" / "figures"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
