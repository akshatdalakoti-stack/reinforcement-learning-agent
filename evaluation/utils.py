"""Reproducibility and device helpers shared by every algorithm."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG this project touches.

    Gymnasium environments are seeded separately, at ``env.reset(seed=...)``,
    because Gymnasium keeps its own ``np_random`` per environment instance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Kept opt-in: it costs throughput on CUDA but makes runs bit-comparable.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` into a concrete torch device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(preference)


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(*parts: str) -> str:
    """Build a file path relative to the repository root, creating its parent."""
    path = os.path.join(project_root(), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def resolve_dir(*parts: str) -> str:
    """Build a directory path relative to the repository root, creating it."""
    path = os.path.join(project_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


LOG_DIR = ("results", "logs")
PLOT_DIR = ("results", "plots")
