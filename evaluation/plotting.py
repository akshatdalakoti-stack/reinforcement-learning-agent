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


def plot_variance_bands(
    curves: dict[str, list[np.ndarray]],
    path: str,
    title: str,
    window: int = 25,
    ylabel: str = "Episode reward",
) -> str:
    """Mean +/- 1 std across seeds -- how we visualise gradient variance.

    ``curves[label]`` is a list of per-seed reward arrays.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, seed_runs) in enumerate(curves.items()):
        colour = PALETTE[i % len(PALETTE)]
        n = min(len(r) for r in seed_runs)
        stack = np.stack([moving_average(r[:n], window) for r in seed_runs])
        mean, std = stack.mean(axis=0), stack.std(axis=0)
        x = np.arange(n)
        # The seed count belongs in the title, not on every label -- appending it
        # here turns "Actor-Critic (n=32)" into "Actor-Critic (n=32) (n=4)".
        ax.plot(x, mean, color=colour, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return save_figure(fig, path)


def plot_gridworld_policy(
    env,
    q_table: np.ndarray,
    path: str,
    title: str = "Learned greedy policy",
) -> str:
    """Arrow map of argmax_a Q(s,a) over the grid, shaded by state value V(s)."""
    rows, cols = env.rows, env.cols
    values = q_table.max(axis=1).reshape(rows, cols)
    greedy = q_table.argmax(axis=1).reshape(rows, cols)

    masked = np.ma.array(values, mask=~env.walkable_mask())
    fig, ax = plt.subplots(figsize=(1.15 * cols + 2.0, 1.15 * rows + 1.2))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#3a3a3a")
    im = ax.imshow(masked, cmap=cmap)
    fig.colorbar(im, ax=ax, label="V(s) = max_a Q(s,a)", fraction=0.046)

    # Arrow directions come straight from the environment's (dr, dc) deltas, so
    # the picture cannot drift out of sync with the action ordering.
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell in env.walls:
                ax.text(c, r, "wall", ha="center", va="center", color="white", fontsize=7)
                continue
            if cell in env.terminals:
                ax.text(
                    c, r, f"{env.terminals[cell]:+.0f}",
                    ha="center", va="center", fontsize=12, fontweight="bold", color="black",
                )
                continue
            dr, dc = env.action_deltas[int(greedy[r, c])]
            dy, dx = 0.3 * dr, 0.3 * dc
            ax.arrow(c - dx, r - dy, 1.6 * dx, 1.6 * dy,
                     head_width=0.16, head_length=0.13, fc="black", ec="black", linewidth=1.2)
            # Flip the label to white on the dark end of the colormap, or the
            # highest-value cells -- the interesting ones -- become unreadable.
            span = float(masked.max() - masked.min()) or 1.0
            shade = (values[r, c] - float(masked.min())) / span
            ax.text(c, r + 0.42, f"{values[r, c]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if shade > 0.6 else "#222222")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title)
    return save_figure(fig, path)


def plot_q_convergence(deltas, path: str, title: str = "Q-table convergence") -> str:
    """max |Q_{t+1} - Q_t| per episode, on a log axis."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(deltas, color=PALETTE[2], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"max$_{s,a}$ |$\Delta Q$|")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    return save_figure(fig, path)
