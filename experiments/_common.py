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


def collapse_count(recorder: MetricsRecorder, window: int = 50, drop: float = 0.4) -> int:
    """How many times the smoothed reward fell by more than ``drop`` from a peak.

    Catastrophic forgetting and performance collapse both show up as a sharp
    fall after a high-water mark; counting those events separates "learned
    slowly" from "learned then fell apart".
    """
    smoothed = recorder.smoothed_rewards(window)
    if smoothed.size < 2:
        return 0
    peak, events, in_collapse = smoothed[0], 0, False
    for value in smoothed[1:]:
        peak = max(peak, value)
        if peak > 1e-9 and value < peak * (1.0 - drop):
            if not in_collapse:
                events += 1
                in_collapse = True
        else:
            in_collapse = False
    return events


def aggregate_curves(recorders: list[MetricsRecorder], window: int = 25):
    """Stack per-seed reward curves to a common length; return ``(mean, std, n)``."""
    n = min(len(r.episodes) for r in recorders)
    stack = np.stack([moving_average(r.rewards[:n], window) for r in recorders])
    return stack.mean(axis=0), stack.std(axis=0), n


def describe(name: str, values: list[float]) -> str:
    """One aligned 'mean +/- std (n)' line.

    The precision adapts to the magnitude: a KL of 0.0076 and a reward of 488.6
    both appear in these tables, and a fixed ``.1f`` would silently print the
    former as ``0.0``.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return f"    {name:<34} never reached"
    scale = np.abs(arr).max()
    if scale >= 100:
        fmt = "10.1f", ".1f"
    elif scale >= 1:
        fmt = "10.2f", ".2f"
    elif scale >= 0.01:
        fmt = "10.4f", ".4f"
    else:
        fmt = "10.2e", ".1e"
    return (f"    {name:<34} {arr.mean():{fmt[0]}}  +/- {arr.std():{fmt[1]}}"
            f"   (n={arr.size})")
