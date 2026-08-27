"""Experiment 4 -- variance: Monte-Carlo returns vs a learned baseline.

Claim
-----
REINFORCE and Actor-Critic optimise the same objective by the same theorem.
They differ in what multiplies ``grad log pi(a|s)``, and that difference is
almost entirely a variance story.

Two measurements
----------------
1. **Direct gradient variance.**  Hold the parameters fixed, estimate the
   gradient many times from independent rollouts, and measure how much the
   estimates agree (mean pairwise cosine similarity, and
   ``||mean grad|| / mean ||grad||``).  This measures the estimator itself,
   with learning dynamics factored out.

2. **Learning-curve spread across seeds.**  What that variance costs in
   practice: how differently identical configurations behave when only the
   random seed changes.

Variants
--------
* ``REINFORCE (no baseline)``   -- the textbook estimator, weights are raw ``G_t``
* ``REINFORCE (normalised)``    -- constant baseline + scaling
* ``Actor-Critic (n=32)``       -- learned state-dependent baseline ``V(s)``

    python -m experiments.exp4_reinforce_vs_actor_critic --seeds 4
"""

from __future__ import annotations

import argparse

import numpy as np

from actor_critic.train import main as ac_main
from environments.registry import make_env
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_variance_bands
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, describe, save_summary, steps_to_threshold
from reinforce.train import build_agent as build_reinforce
from reinforce.train import main as reinforce_main
from reinforce.train import measure_gradient_variance, parse_args as reinforce_args


def gradient_variance_probe(baseline: str, samples: int, seed: int) -> dict:
    """Measure gradient agreement at a *fixed, untrained* policy.

    Using an untrained policy is deliberate: it removes the confound that a
    better policy produces different-length episodes, so the two baselines are
    compared on the same data distribution.
    """
    args = reinforce_args(["--baseline", baseline, "--seed", str(seed)])
    env = make_env("CartPole-v1", seed=seed)
    from environments.registry import env_spec
    from evaluation.utils import get_device

    agent = build_reinforce(args, env_spec(env), get_device("cpu"))
    stats = measure_gradient_variance(agent, env, samples, seed)
    env.close()
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-env-steps", type=int, default=150_000)
    p.add_argument("--variance-samples", type=int, default=60)
    p.add_argument("--threshold", type=float, default=195.0)
    args = p.parse_args(argv)

    banner(
        "Experiment 4: REINFORCE vs Actor-Critic",
        "Where the variance comes from, and what it costs",
    )

    # ---- part 1: the estimator itself ------------------------------------
    print("\n  Part 1 -- gradient agreement at fixed parameters")
    print(f"  ({args.variance_samples} independent single-episode gradient estimates)\n")
    probes = {}
    for baseline in ("none", "norm"):
        stats = gradient_variance_probe(baseline, args.variance_samples, seed=0)
        probes[baseline] = stats
        print(f"    baseline={baseline:<6}  "
              f"cosine similarity {stats['mean_cosine_similarity']:+.3f}  "
              f"SNR {stats['snr']:.3f}")
    print("\n    Cosine similarity near 0 means two rollouts from the SAME policy")
    print("    produce gradient estimates pointing in nearly unrelated directions.")

    # ---- part 2: what it costs in training -------------------------------
    print("\n\n  Part 2 -- learning curves across seeds")
    runs: dict[str, list[MetricsRecorder]] = {}
    evals: dict[str, list[float]] = {}

    # (label -> (runner kind, log slug, flags)).  The slug must be unique per
    # variant: both REINFORCE rows use the same runner, so keying the log files
    # on the runner name alone would make the second overwrite the first.
    configs = {
        "REINFORCE (no baseline)": ("reinforce", "reinforce_nobase", ["--baseline", "none"]),
        "REINFORCE (normalised)": ("reinforce", "reinforce_norm", ["--baseline", "norm"]),
        "Actor-Critic (n=32)": ("actor_critic", "actor_critic", []),
    }

    for label, (kind, slug, flags) in configs.items():
        runs[label], evals[label] = [], []
        for seed in range(args.seeds):
            print(f"\n  >> {label}  seed={seed}")
            common = [
                "--episodes", str(args.episodes),
                "--max-env-steps", str(args.max_env_steps),
                "--seed", str(seed),
                "--run-name", f"exp4_{slug}_s{seed}",
                "--no-plots", "--no-save", "--quiet",
            ]
            runner = reinforce_main if kind == "reinforce" else ac_main
            _, recorder, result = runner([*flags, *common])
            runs[label].append(recorder)
            evals[label].append(result["mean_reward"])

    print("\n\n  --- comparison ------------------------------------------")
    summary = {"gradient_probe": probes}
    for label, recorders in runs.items():
        finals = evals[label]
        to_threshold = [steps_to_threshold(r, args.threshold) for r in recorders]
        # Spread across seeds at the end of training: the practical cost of variance.
        seed_spread = float(np.std(finals))

        print(f"\n  {label}")
        print(describe("greedy eval reward", finals))
        print(describe(f"env steps to smoothed>={args.threshold:.0f}", to_threshold))
        print(f"    {'across-seed std of final reward':<34} {seed_spread:10.1f}")

        summary[label] = {
            "eval_reward_mean": float(np.mean(finals)),
            "eval_reward_std": seed_spread,
            "eval_rewards": [float(f) for f in finals],
            "reached_threshold": int(sum(t is not None for t in to_threshold)),
            "steps_to_threshold_mean": (
                float(np.mean([t for t in to_threshold if t is not None]))
                if any(t is not None for t in to_threshold) else None
            ),
        }

    plot_variance_bands(
        {label: [r.rewards for r in recs] for label, recs in runs.items()},
        resolve_path(*PLOT_DIR, "exp4_reinforce_vs_actor_critic.png"),
        title=f"REINFORCE vs Actor-Critic on CartPole ({args.seeds} seeds, mean +/- 1 std)",
        window=25,
    )

    print("\n  Reading: the shaded bands are the visible form of the cosine-similarity")
    print("  number from Part 1 -- an estimator whose samples disagree produces runs")
    print("  that disagree.")

    save_summary("exp4_reinforce_vs_actor_critic", summary)
    return summary


if __name__ == "__main__":
    main()
