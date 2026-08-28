"""Experiment 5 -- what does PPO's clipping actually prevent?

Claim
-----
Clipping is what makes multi-epoch reuse of a rollout safe.  Without it, taking
many gradient steps on the same data drives the policy far from the one that
collected it, the importance ratios blow up, and performance collapses.

Design
------
The clean control is not "PPO vs A2C" -- that changes several things at once.
It is **the same PPO agent, same rollouts, same number of epochs, with the clip
range set so wide it never binds** (``--clip-epsilon 10``).  That variant is
exactly "importance-weighted policy gradient with 10 epochs of reuse", which is
the thing clipping was invented to fix.

Settings compared
-----------------
* ``clip=0.2, 10 epochs``   -- standard PPO
* ``clip=inf, 10 epochs``   -- no trust region: the failure case
* ``clip=0.2,  1 epoch``    -- clipping present but nothing to protect against
* ``clip=0.05, 10 epochs``  -- an over-tight region: stable but slow

Diagnostics
-----------
``approx_kl`` (how far the policy moved), ``clip_fraction`` (how often the
bound bound), and policy entropy (whether the policy is collapsing).

    python -m experiments.exp5_ppo_clipping --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_variance_bands
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, collapse_count, describe, save_summary
from ppo.train import main as ppo_main

# slug -> (display label, flags).  The slug keys the per-run log filenames, so
# it must be unique: two variants share the label prefix "PPO clip=0.2".
VARIANTS = {
    "clip0.2_ep10": ("PPO clip=0.2, 10 epochs", ["--clip-epsilon", "0.2", "--n-epochs", "10"]),
    "noclip_ep10": ("no clipping, 10 epochs", ["--clip-epsilon", "10.0", "--n-epochs", "10"]),
    "clip0.2_ep1": ("PPO clip=0.2, 1 epoch", ["--clip-epsilon", "0.2", "--n-epochs", "1"]),
    "clip0.05_ep10": ("PPO clip=0.05, 10 epochs", ["--clip-epsilon", "0.05", "--n-epochs", "10"]),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=80_000)
    args = p.parse_args(argv)

    banner(
        "Experiment 5: the clipped surrogate objective",
        f"PPO on {args.env}, {args.seeds} seeds x {args.total_steps} steps",
    )

    runs: dict[str, list[MetricsRecorder]] = {}
    evals: dict[str, list[float]] = {}

    for slug, (label, flags) in VARIANTS.items():
        runs[label], evals[label] = [], []
        for seed in range(args.seeds):
            print(f"\n  >> {label}  seed={seed}")
            _, recorder, result = ppo_main([
                *flags,
                "--env", args.env,
                "--total-steps", str(args.total_steps),
                "--seed", str(seed),
                "--run-name", f"exp5_{slug}_s{seed}",
                "--no-plots", "--no-save", "--quiet",
            ])
            runs[label].append(recorder)
            evals[label].append(result["mean_reward"])

    print("\n\n  --- comparison ------------------------------------------")
    summary = {}
    for label, recorders in runs.items():
        finals = evals[label]
        kl = [float(np.nanmean(r.series("approx_kl"))) for r in recorders]
        max_kl = [float(np.nanmax(r.series("approx_kl"))) for r in recorders]
        clip_frac = [float(np.nanmean(r.series("clip_fraction"))) for r in recorders]
        entropy = [float(r.series("entropy")[-1]) for r in recorders]
        collapses = [float(collapse_count(r)) for r in recorders]

        print(f"\n  {label}")
        print(describe("greedy eval reward", finals))
        print(describe("mean approx KL per update", kl))
        print(describe("max approx KL", max_kl))
        print(describe("clip fraction", clip_frac))
        print(describe("final policy entropy", entropy))
        print(describe("collapse events", collapses))

        summary[label] = {
            "eval_reward_mean": float(np.mean(finals)),
            "eval_reward_std": float(np.std(finals)),
            "approx_kl_mean": float(np.mean(kl)),
            "approx_kl_max": float(np.max(max_kl)),
            "clip_fraction_mean": float(np.mean(clip_frac)),
            "final_entropy_mean": float(np.mean(entropy)),
            "collapses_mean": float(np.mean(collapses)),
        }

    plot_variance_bands(
        {name: [r.rewards for r in recs] for name, recs in runs.items()},
        resolve_path(*PLOT_DIR, "exp5_ppo_clipping.png"),
        title=f"PPO clipping ablation on {args.env} ({args.seeds} seeds, mean +/- 1 std)",
        window=25,
    )

    clipped = summary["PPO clip=0.2, 10 epochs"]
    unclipped = summary["no clipping, 10 epochs"]
    print("\n  Reading:")
    print(f"    with clipping   reward {clipped['eval_reward_mean']:7.1f}, "
          f"max KL {clipped['approx_kl_max']:.4f}")
    print(f"    without         reward {unclipped['eval_reward_mean']:7.1f}, "
          f"max KL {unclipped['approx_kl_max']:.4f}")
    ratio = unclipped["approx_kl_max"] / max(clipped["approx_kl_max"], 1e-9)
    print(f"    Removing the clip lets the policy move {ratio:.0f}x further per update.")
    print("    Same data, same epochs, same optimiser -- only the trust region differs.")

    save_summary("exp5_ppo_clipping", summary)
    return summary


