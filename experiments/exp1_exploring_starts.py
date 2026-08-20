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


