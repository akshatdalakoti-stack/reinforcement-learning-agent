"""Shared plumbing for the experiment scripts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from evaluation.metrics import MetricsRecorder, moving_average
from evaluation.utils import LOG_DIR, resolve_path


def banner(title: str, subtitle: str = "") -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 74)


def save_summary(name: str, payload: dict[str, Any]) -> str:
    path = resolve_path(*LOG_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n  summary -> {path}")
    return path


