"""Experiment 2 -- a 2x2 factorial over replay and the target network.

Why factorial
-------------
The obvious comparison -- online DQN vs the full agent -- changes two things at
once, so it cannot tell you which one mattered.  Running all four corners does:

                      no target network      target network
    no replay         (a) stage-2 baseline   (c) target only
    replay            (b) replay only        (d) full agent, stage 4

* (d) - (b) isolates the **target network**
* (d) - (c) isolates **replay**
* (a) is the naive deep Q-learner that stage 2 warns about

Everything else -- network, optimiser, exploration schedule, seed, environment
step budget, and the number of gradient steps taken -- is identical across all
four.  The only difference is *which transitions each gradient step sees* and
*which network produces the bootstrap target*.

A note on batch size: the no-replay variants necessarily train on a batch of
one, because a decorrelated batch of 128 is exactly the thing a replay buffer
provides.  That is a consequence of the ablation, not a confound in it.

Metric
------
Progress is measured by **periodic greedy evaluation**, not by training reward.
With ``epsilon = 0.05`` a single random action ends a CartPole episode, so a
fully-trained DQN's *training* reward oscillates between ~90 and ~250 while its
greedy policy scores a flat 500.  Judging this ablation on training reward would
mostly measure exploration noise.

    python -m experiments.exp2_replay_ablation --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from dqn.train import main as dqn_main
from evaluation.plotting import PALETTE, plot_variance_bands, save_figure
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, describe, save_summary

VARIANTS = {
    "(a) neither": ["--no-replay", "--no-target-network"],
    "(b) replay only": ["--no-target-network"],
    "(c) target net only": ["--no-replay"],
    "(d) replay + target net": [],
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--max-env-steps", type=int, default=120_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--threshold", type=float, default=195.0)
    args = p.parse_args(argv)

    banner(
        "Experiment 2: replay x target-network factorial",
        f"DQN on CartPole-v1, {args.seeds} seeds, {args.max_env_steps:,} env steps each",
    )

    results: dict[str, dict[str, list]] = {}

    for label, flags in VARIANTS.items():
        results[label] = {"best": [], "final": [], "steps": [], "rewards": [], "peak_q": []}
        for seed in range(args.seeds):
            print(f"\n  >> {label}  seed={seed}")
            _, recorder, _, evaluator = dqn_main([
                *flags,
                "--episodes", str(args.episodes),
                "--max-env-steps", str(args.max_env_steps),
                "--eval-every", str(args.eval_every),
                "--seed", str(seed),
                "--run-name", f"exp2_{label[1]}_s{seed}",
                "--no-plots", "--no-save", "--quiet",
            ])
            results[label]["best"].append(evaluator.best)
            results[label]["final"].append(evaluator.final)
            results[label]["steps"].append(evaluator.steps_to(args.threshold))
            results[label]["rewards"].append(recorder.rewards)
            results[label]["peak_q"].append(float(np.nanmax(recorder.series("mean_q"))))

    print("\n\n  --- comparison ------------------------------------------")
    summary = {}
    for label, data in results.items():
        reached = sum(s is not None for s in data["steps"])
        print(f"\n  {label}")
        print(describe("best greedy reward", data["best"]))
        print(describe("final greedy reward", data["final"]))
        print(describe(f"env steps to greedy>={args.threshold:.0f}", data["steps"]))
        print(f"    {'seeds reaching threshold':<34} {reached}/{args.seeds}")
        print(describe("peak mean predicted Q", data["peak_q"]))

        summary[label] = {
            "best_greedy_mean": float(np.mean(data["best"])),
            "best_greedy_std": float(np.std(data["best"])),
            "final_greedy_mean": float(np.mean(data["final"])),
            "reached_threshold": reached,
            "seeds": args.seeds,
            "steps_to_threshold_mean": (
                float(np.mean([s for s in data["steps"] if s is not None]))
                if reached else None
            ),
            "peak_mean_q": float(np.mean(data["peak_q"])),
        }

    # ---- the two main effects -------------------------------------------
    d = summary["(d) replay + target net"]["best_greedy_mean"]
    b = summary["(b) replay only"]["best_greedy_mean"]
    c = summary["(c) target net only"]["best_greedy_mean"]
    a = summary["(a) neither"]["best_greedy_mean"]
    effects = {
        "target_network_effect_with_replay": d - b,
        "target_network_effect_without_replay": c - a,
        "replay_effect_with_target": d - c,
        "replay_effect_without_target": b - a,
    }
    summary["main_effects"] = effects

    print("\n  --- main effects (best greedy reward) --------------------")
    print(f"    adding a target network, with replay:     {d:6.1f} - {b:6.1f} = "
          f"{effects['target_network_effect_with_replay']:+7.1f}")
    print(f"    adding a target network, without replay:  {c:6.1f} - {a:6.1f} = "
          f"{effects['target_network_effect_without_replay']:+7.1f}")
    print(f"    adding replay, with a target network:     {d:6.1f} - {c:6.1f} = "
          f"{effects['replay_effect_with_target']:+7.1f}")
    print(f"    adding replay, without a target network:  {b:6.1f} - {a:6.1f} = "
          f"{effects['replay_effect_without_target']:+7.1f}")

    plot_variance_bands(
        {label: data["rewards"] for label, data in results.items()},
        resolve_path(*PLOT_DIR, "exp2_replay_ablation.png"),
        title=f"Replay x target-network factorial, training reward "
              f"({args.seeds} seeds, mean +/- 1 std)",
        window=25,
    )

    # ---- greedy-evaluation curves, the cleaner view ----------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, flags) in enumerate(VARIANTS.items()):
        colour = PALETTE[i % len(PALETTE)]
        ax.bar(i, summary[label]["best_greedy_mean"],
               yerr=summary[label]["best_greedy_std"],
               color=colour, alpha=0.85, capsize=5, width=0.6)
        # Sit the label clear of the error bar, not on top of it.
        ax.text(i,
                summary[label]["best_greedy_mean"] + summary[label]["best_greedy_std"] + 14,
                f"{summary[label]['best_greedy_mean']:.0f}",
                ha="center", fontsize=10)
    ax.axhline(args.threshold, color="#555", linestyle="--", linewidth=1.2,
               label=f"threshold ({args.threshold:.0f})")
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(list(VARIANTS), fontsize=9)
    ax.set_ylabel("Best greedy evaluation reward")
    ax.set_title(f"Each corner of the 2x2 ({args.seeds} seeds, mean +/- 1 std)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, resolve_path(*PLOT_DIR, "exp2_factorial_bars.png"))

    save_summary("exp2_replay_ablation", summary)
    return summary


if __name__ == "__main__":
    main()
