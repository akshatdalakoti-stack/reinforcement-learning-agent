"""Experiment 3 -- the target-update frequency trade-off.

Claim
-----
The target network trades *speed* for *stability*, and the update frequency is
the dial between them.

Why a target network is needed at all
-------------------------------------
Without one, the regression target is built from the very network being
updated::

    L = ( Q_θ(s,a) - [r + gamma * max_a' Q_θ(s',a')] )^2

Since ``s`` and ``s'`` are neighbouring states, a gradient step that raises
``Q(s,a)`` also raises ``Q(s',a')`` -- which raises the target -- which raises
``Q(s,a)`` again.  The fixed point can run away.  This is the *deadly triad*:
function approximation + bootstrapping + off-policy updates.  Any two are safe;
all three together have no convergence guarantee.

Freezing the target for ``k`` gradient steps turns the problem into a sequence
of ``k``-step **supervised regressions against a stationary target**, restoring
the assumption gradient descent needs.

The trade-off
-------------
* ``k`` too small -> the target moves almost as fast as the estimate; barely
  different from having no target network.
* ``k`` too large -> the targets are stale, so value information propagates
  backwards through the state space very slowly.

Design
------
Sweep ``k`` in gradient steps, plus a no-target-network control, across seeds.
Track final performance, sample efficiency, instability, and the mean predicted
Q-value -- the last of which is where divergence shows up first.

    python -m experiments.exp3_target_network_frequency --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from dqn.train import main as dqn_main
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import PALETTE, plot_variance_bands, save_figure
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import (
    banner,
    collapse_count,
    describe,
    save_summary,
    stability_score,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--max-env-steps", type=int, default=120_000)
    p.add_argument("--frequencies", type=int, nargs="+", default=[1, 50, 250, 1000, 5000],
                   help="target sync period in GRADIENT steps")
    p.add_argument("--threshold", type=float, default=195.0)
    p.add_argument("--eval-every", type=int, default=5_000)
    args = p.parse_args(argv)

    banner(
        "Experiment 3: target-network update frequency",
        f"DQN + replay on CartPole-v1, {args.seeds} seeds per setting",
    )

    variants: dict[str, list[str]] = {"no target network": ["--no-target-network"]}
    for k in args.frequencies:
        variants[f"sync every {k} grad steps"] = ["--target-update-freq", str(k)]

    all_runs: dict[str, list[MetricsRecorder]] = {}
    evaluations: dict[str, list[float]] = {}
    steps_to: dict[str, list] = {}

    for label, flags in variants.items():
        all_runs[label], evaluations[label], steps_to[label] = [], [], []
        for seed in range(args.seeds):
            print(f"\n  >> {label}  seed={seed}")
            _, recorder, _, evaluator = dqn_main([
                *flags,
                "--episodes", str(args.episodes),
                "--max-env-steps", str(args.max_env_steps),
                "--eval-every", str(args.eval_every),
                "--seed", str(seed),
                "--run-name", f"exp3_{label.replace(' ', '_')}_s{seed}",
                "--no-plots", "--no-save", "--quiet",
            ])
            all_runs[label].append(recorder)
            # Best periodic greedy evaluation, not end-of-training training
            # reward: see the note in exp2 on why the latter is the wrong metric.
            evaluations[label].append(evaluator.best)
            steps_to[label].append(evaluator.steps_to(args.threshold))

    print("\n\n  --- comparison ------------------------------------------")
    summary = {}
    for label, recorders in all_runs.items():
        finals = evaluations[label]
        to_threshold = steps_to[label]
        stability = [stability_score(r) for r in recorders]
        collapses = [float(collapse_count(r)) for r in recorders]
        peak_q = [float(np.nanmax(r.series("mean_q"))) for r in recorders]

        print(f"\n  {label}")
        print(describe("best greedy reward", finals))
        print(describe(f"env steps to greedy>={args.threshold:.0f}", to_threshold))
        print(describe("instability (training reward)", stability))
        print(describe("peak mean predicted Q", peak_q))

        summary[label] = {
            "eval_reward_mean": float(np.mean(finals)),
            "eval_reward_std": float(np.std(finals)),
            "reached_threshold": int(sum(t is not None for t in to_threshold)),
            "steps_to_threshold_mean": (
                float(np.mean([t for t in to_threshold if t is not None]))
                if any(t is not None for t in to_threshold) else None
            ),
            "instability_mean": float(np.mean(stability)),
            "collapses_mean": float(np.mean(collapses)),
            "peak_mean_q": float(np.mean(peak_q)),
        }

    plot_variance_bands(
        {label: [r.rewards for r in recs] for label, recs in all_runs.items()},
        resolve_path(*PLOT_DIR, "exp3_target_frequency_curves.png"),
        title=f"Target-update frequency on CartPole ({args.seeds} seeds, mean +/- 1 std)",
        window=25,
    )

    # ---- the trade-off, as a single figure -------------------------------
    import matplotlib.pyplot as plt

    freq_labels = [f"sync every {k} grad steps" for k in args.frequencies]
    x = np.arange(len(args.frequencies))
    rewards = [summary[l]["eval_reward_mean"] for l in freq_labels]
    reward_err = [summary[l]["eval_reward_std"] for l in freq_labels]
    peak_q_values = [summary[l]["peak_mean_q"] for l in freq_labels]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    bar = ax1.errorbar(x, rewards, yerr=reward_err, color=PALETTE[0], marker="o",
                       linewidth=2, capsize=4, label="best greedy reward")
    control = ax1.axhline(
        summary["no target network"]["eval_reward_mean"],
        color=PALETTE[5], linestyle="--", linewidth=1.5,
        label="no target network (control)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in args.frequencies])
    ax1.set_xlabel("Target sync period (gradient steps)")
    ax1.set_ylabel("Best greedy evaluation reward", color=PALETTE[0])
    ax1.grid(alpha=0.25)

    # Peak predicted Q on a log axis is the diagnostic that explains *both*
    # failure modes at once: it runs away at small k and never grows at large k.
    ax2 = ax1.twinx()
    q_line, = ax2.plot(x, peak_q_values, color=PALETTE[1], marker="s", linewidth=2,
                       linestyle=":", label="peak mean predicted Q")
    ax2.set_yscale("log")
    ax2.set_ylabel("Peak mean predicted Q (log scale)", color=PALETTE[1])
    ax2.axhline(99.3, color=PALETTE[1], linestyle="-", linewidth=0.8, alpha=0.4)
    ax2.text(len(x) - 1, 110, "true max ~99", color=PALETTE[1], fontsize=7,
             ha="right", va="bottom", alpha=0.8)

    # Build the legend from named artists only; errorbar containers contribute
    # unlabelled cap/bar lines that would otherwise show up as "_nolegend_".
    ax1.legend([bar, control, q_line],
               [a.get_label() for a in (bar, control, q_line)],
               loc="lower center", fontsize=8)
    ax1.set_title("Target-update frequency: too fast diverges, too slow never learns")
    save_figure(fig, resolve_path(*PLOT_DIR, "exp3_target_frequency_tradeoff.png"))

    best = max(freq_labels, key=lambda l: summary[l]["eval_reward_mean"])
    print(f"\n  Reading: best setting here is '{best}' "
          f"({summary[best]['eval_reward_mean']:.0f} reward).")
    print("    Very frequent syncs approach the no-target-network control; very rare")
    print("    syncs slow value propagation. The curve is non-monotonic in both directions.")

    save_summary("exp3_target_network_frequency", summary)
    return summary


