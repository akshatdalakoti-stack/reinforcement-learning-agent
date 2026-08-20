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


def steps_to_threshold(
    recorder: MetricsRecorder, threshold: float, window: int = 50
) -> float | None:
    """Environment steps until the moving-average reward first reaches ``threshold``.

    This is the sample-efficiency number.  It is more informative than final
    reward when several variants eventually reach the same score -- and it is
    ``None`` for the ones that never get there, which is itself the result.
    """
    smoothed = recorder.smoothed_rewards(window)
    env_steps = recorder.series("env_steps")
    hits = np.flatnonzero(smoothed >= threshold)
    if hits.size == 0:
        return None
    return float(env_steps[hits[0]])


def stability_score(recorder: MetricsRecorder, window: int = 50) -> float:
    """Mean absolute episode-to-episode change in the smoothed reward.

    A learning curve that climbs steadily has a small value here; one that
    oscillates or collapses and recovers has a large one.  This is how we put a
    number on the word "unstable" instead of pointing at a wobbly plot.
    """
    smoothed = recorder.smoothed_rewards(window)
    if smoothed.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(smoothed))))


