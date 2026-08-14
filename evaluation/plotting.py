"""Matplotlib helpers for learning curves and ablation comparisons."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # scripts run headless; figures go to results/plots/
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.metrics import MetricsRecorder, moving_average  # noqa: E402

PALETTE = ["#2f6f9f", "#c1543a", "#4a8a5c", "#8a6bb1", "#c9922e", "#5b5f66"]


def save_figure(fig, path: str) -> str:
    """Tighten, save at print resolution, close, and report the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved plot -> {path}")
    return path


def plot_learning_curve(
    recorder: MetricsRecorder,
    path: str,
    window: int = 50,
    title: str | None = None,
) -> str:
    """Reward + moving average, episode length, loss and entropy in one figure.

    Panels for loss/entropy are dropped when the algorithm does not report them
    (tabular Q-learning has no loss curve in the supervised sense, for example).
    """
    rewards = recorder.rewards
    lengths = recorder.series("length")
    losses = recorder.series("loss")
    entropies = recorder.series("entropy")

    panels: list[tuple[str, np.ndarray, bool]] = [
        ("Episode reward", rewards, True),
        ("Episode length", lengths, True),
    ]
    if np.isfinite(losses).any():
        panels.append(("Training loss", losses, False))
    if np.isfinite(entropies).any():
        panels.append(("Policy entropy (nats)", entropies, False))

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(rewards))

    for ax, (label, values, smooth) in zip(axes, panels):
        ax.plot(x, values, color=PALETTE[0], alpha=0.28, linewidth=0.9, label="per episode")
        if smooth:
            ax.plot(
                x,
                moving_average(values, window),
                color=PALETTE[1],
                linewidth=2.0,
                label=f"moving avg ({window})",
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Episode")
    fig.suptitle(title or recorder.run_name, fontsize=12)
    fig.subplots_adjust(top=0.94)
    return save_figure(fig, path)


def plot_comparison(
    recorders: dict[str, MetricsRecorder],
    path: str,
    window: int = 50,
    title: str = "Comparison",
    x_axis: str = "episode",
) -> str:
    """Overlay smoothed reward curves from several runs (the ablation figure).

    ``x_axis`` may be ``episode`` or ``env_steps``; the latter is the honest
    axis when the algorithms consume samples at different rates.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, rec) in enumerate(recorders.items()):
        colour = PALETTE[i % len(PALETTE)]
        y = rec.smoothed_rewards(window)
        x = rec.series("env_steps") if x_axis == "env_steps" else np.arange(len(y))
        ax.plot(x, y, color=colour, linewidth=2.0, label=label)
        ax.plot(x, rec.rewards, color=colour, alpha=0.15, linewidth=0.7)
    ax.set_xlabel("Environment steps" if x_axis == "env_steps" else "Episode")
    ax.set_ylabel(f"Reward (moving avg, window={window})")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return save_figure(fig, path)


