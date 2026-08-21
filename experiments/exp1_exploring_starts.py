"""Experiment 1 -- what actually limits tabular Q-learning?

Claim
-----
With enough episodes the Bellman update is not the bottleneck; *coverage* is.
Q-learning converges to Q* only for state-action pairs it visits infinitely
often, and a fixed start state means the pairs off the optimal path are visited
almost never.

Design
------
Four configurations on the same GridWorld, several seeds each:

* fixed start,     constant alpha
* fixed start,     1/n(s,a) alpha  (satisfies the Robbins-Monro conditions)
* exploring starts, constant alpha
* exploring starts, 1/n(s,a) alpha

Because GridWorld is small enough to solve exactly with value iteration, we can
report the *true* error rather than a proxy:

* ``|Q - Q*|_max`` over decision states
* how many states the greedy policy gets optimal
* state-action coverage

    python -m experiments.exp1_exploring_starts --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.gridworld import GridWorld, value_iteration
from evaluation.plotting import PALETTE, save_figure
from evaluation.utils import PLOT_DIR, resolve_path, set_seed
from experiments._common import banner, save_summary
from q_learning.agent import QLearningAgent


def run_variant(
    exploring_starts: bool,
    alpha_schedule: str,
    episodes: int,
    seed: int,
    slip_prob: float = 0.1,
    gamma: float = 0.95,
) -> dict:
    set_seed(seed)
    env = GridWorld(slip_prob=slip_prob)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        alpha_schedule=alpha_schedule,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=seed,
    )
    q_star, _ = value_iteration(env, gamma=gamma)
    decision_states = env.decision_states()
    options = {"exploring_start": exploring_starts}

    gap_curve, reward_curve = [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed * 7919 + episode, options=options)
        total = 0.0
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            total += reward
            state = next_state
            if terminated or truncated:
                break
        agent.decay_epsilon()
        gap_curve.append(
            float(np.abs(agent.q_table[decision_states] - q_star[decision_states]).max())
        )
        reward_curve.append(total)

    greedy = agent.greedy_policy()
    optimal_states = sum(
        1 for s in decision_states if q_star[s, greedy[s]] >= q_star[s].max() - 1e-9
    )
    return {
        "final_gap": gap_curve[-1],
        "gap_curve": gap_curve,
        "reward_curve": reward_curve,
        "optimal_states": optimal_states,
        "n_decision_states": len(decision_states),
        "coverage": agent.coverage(decision_states),
        "min_visits": int(agent.visit_counts[decision_states].min()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--slip-prob", type=float, default=0.1)
    args = p.parse_args(argv)

    banner(
        "Experiment 1: exploration coverage and step-size schedule",
        "Tabular Q-learning vs the exact value-iteration solution",
    )

    variants = {
        "fixed start, constant a": (False, "constant"),
        "fixed start, a=1/n": (False, "visit_count"),
        "exploring starts, constant a": (True, "constant"),
        "exploring starts, a=1/n": (True, "visit_count"),
    }

    results: dict[str, list[dict]] = {}
    for label, (exploring, schedule) in variants.items():
        results[label] = [
            run_variant(exploring, schedule, args.episodes, seed, args.slip_prob)
            for seed in range(args.seeds)
        ]

    print(f"\n  {args.episodes} episodes x {args.seeds} seeds, slip_prob={args.slip_prob}\n")
    header = f"  {'variant':<30}{'|Q-Q*|max':>12}{'optimal pi':>13}{'coverage':>11}{'min visits':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary = {}
    for label, runs in results.items():
        gaps = np.array([r["final_gap"] for r in runs])
        optimal = np.array([r["optimal_states"] for r in runs])
        coverage = np.array([r["coverage"] for r in runs])
        min_visits = np.array([r["min_visits"] for r in runs])
        n_states = runs[0]["n_decision_states"]
        print(f"  {label:<30}{gaps.mean():9.3f}   "
              f"{optimal.mean():6.1f}/{n_states:<5} "
              f"{coverage.mean():10.0%} {min_visits.mean():11.0f}")
        summary[label] = {
            "final_gap_mean": float(gaps.mean()),
            "final_gap_std": float(gaps.std()),
            "optimal_states_mean": float(optimal.mean()),
            "n_decision_states": n_states,
            "coverage_mean": float(coverage.mean()),
            "min_visits_mean": float(min_visits.mean()),
        }

    # ---- figure: convergence to Q* --------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, runs) in enumerate(results.items()):
        curves = np.stack([r["gap_curve"] for r in runs])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        x = np.arange(curves.shape[1])
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(x, mean, color=colour, linewidth=1.8, label=label)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15)
    ax.set_yscale("log")
    # A single seed can momentarily land almost exactly on Q*, which on a log
    # axis drags the limits down by three decades and squashes the region where
    # all the real signal lives. Clamp to the informative range.
    floor = max(1e-2, 0.5 * min(c.min() for c in
                                (np.stack([r["gap_curve"] for r in runs]).mean(axis=0)
                                 for runs in results.values())))
    ax.set_ylim(floor, 2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"max$_{s,a}$ |Q - Q*|  (log scale)")
    ax.set_title("Distance to the exact optimum, mean +/- 1 std across seeds")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, resolve_path(*PLOT_DIR, "exp1_exploring_starts.png"))

    best = min(summary.items(), key=lambda kv: kv[1]["final_gap_mean"])
    worst = max(summary.items(), key=lambda kv: kv[1]["final_gap_mean"])
    print(f"\n  Reading: '{worst[0]}' leaves a gap of "
          f"{worst[1]['final_gap_mean']:.3f}; '{best[0]}' gets it to "
          f"{best[1]['final_gap_mean']:.3f}.")
    print("  The update rule is identical in all four -- only which transitions")
    print("  the agent sees, and how it averages them, differ.")

    save_summary("exp1_exploring_starts", summary)
    return summary


if __name__ == "__main__":
    main()
