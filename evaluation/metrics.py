"""Per-episode metric recording.

Every algorithm in this repository logs through :class:`MetricsRecorder` so that
learning curves from Q-learning and PPO can be compared with the same code.

The metric set is fixed by the project brief:

* episode reward           -- the return actually collected
* moving-average reward    -- the smoothed learning curve
* episode length           -- steps until termination/truncation
* training loss            -- where the algorithm has one
* policy entropy           -- where the policy is stochastic
* environment steps        -- the sample-efficiency x-axis
* training time            -- the wall-clock x-axis
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def moving_average(values, window: int = 50) -> np.ndarray:
    """Trailing moving average, same length as ``values``.

    The first ``window - 1`` entries average over however many samples exist,
    so the curve starts at episode 0 instead of at episode ``window``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


@dataclass
class MetricsRecorder:
    """Collects one row per episode plus run-level metadata."""

    run_name: str
    config: dict[str, Any] = field(default_factory=dict)
    episodes: list[dict[str, float]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.perf_counter, repr=False)
    total_env_steps: int = 0

    # ------------------------------------------------------------------ record
    def record_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        env_steps: int,
        loss: float | None = None,
        entropy: float | None = None,
        **extra: float,
    ) -> None:
        self.total_env_steps = env_steps
        row: dict[str, float] = {
            "episode": float(episode),
            "reward": float(reward),
            "length": float(length),
            "env_steps": float(env_steps),
            "elapsed_sec": self.elapsed,
        }
        if loss is not None and np.isfinite(loss):
            row["loss"] = float(loss)
        if entropy is not None and np.isfinite(entropy):
            row["entropy"] = float(entropy)
        row.update({k: float(v) for k, v in extra.items()})
        self.episodes.append(row)

    # ------------------------------------------------------------------ access
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def series(self, key: str) -> np.ndarray:
        """Column ``key`` across episodes; NaN where an episode lacks it."""
        return np.array([ep.get(key, np.nan) for ep in self.episodes], dtype=np.float64)

    @property
    def rewards(self) -> np.ndarray:
        return self.series("reward")

    def smoothed_rewards(self, window: int = 50) -> np.ndarray:
        return moving_average(self.rewards, window)

    def final_performance(self, window: int = 50) -> float:
        """Mean reward over the last ``window`` training episodes."""
        rewards = self.rewards
        if rewards.size == 0:
            return float("nan")
        return float(np.mean(rewards[-window:]))

    def summary(self, window: int = 50) -> dict[str, Any]:
        rewards = self.rewards
        lengths = self.series("length")
        return {
            "run_name": self.run_name,
            "config": self.config,
            "episodes": len(self.episodes),
            "env_steps": int(self.total_env_steps),
            "training_time_sec": round(self.elapsed, 2),
            "best_episode_reward": float(np.max(rewards)) if rewards.size else float("nan"),
            "final_performance_train": round(self.final_performance(window), 3),
            "mean_episode_length_final": (
                float(np.mean(lengths[-window:])) if lengths.size else float("nan")
            ),
            "steps_per_second": (
                round(self.total_env_steps / max(self.elapsed, 1e-9), 1)
            ),
        }

    # -------------------------------------------------------------------- save
    def save(self, directory: str) -> tuple[str, str]:
        """Write ``<run_name>.csv`` (per-episode) and ``<run_name>.json`` (summary)."""
        os.makedirs(directory, exist_ok=True)
        csv_path = os.path.join(directory, f"{self.run_name}.csv")
        json_path = os.path.join(directory, f"{self.run_name}.json")

        keys: list[str] = []
        for ep in self.episodes:
            for k in ep:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(keys) + "\n")
            for ep in self.episodes:
                fh.write(",".join("" if k not in ep else f"{ep[k]:.6g}" for k in keys) + "\n")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2)
        return csv_path, json_path

    @staticmethod
    def load_csv(path: str) -> dict[str, np.ndarray]:
        """Read a recorder CSV back into column arrays (used by plotting scripts)."""
        with open(path, "r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            cols: dict[str, list[float]] = {k: [] for k in header}
            for line in fh:
                if not line.strip():
                    continue
                for k, v in zip(header, line.rstrip("\n").split(",")):
                    cols[k].append(float(v) if v else np.nan)
        return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
