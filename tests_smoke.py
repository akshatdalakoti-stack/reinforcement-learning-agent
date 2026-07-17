"""Fast correctness checks for the hand-written components.

    python tests_smoke.py

Not a substitute for the training runs. These target the places where a silent
bug would be invisible in a learning curve -- a wrong discount, a target network
that is secretly being trained, a GAE recursion that ignores episode
boundaries. Every check below tests something this project implements by hand
rather than something PyTorch or Gymnasium already guarantees.
"""

from __future__ import annotations

import numpy as np
import torch

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_gridworld_transitions():
    """P(s'|s,a) must be a distribution, and step() must actually sample from it."""
    from environments.gridworld import GridWorld

    env = GridWorld(slip_prob=0.2)
    P, _ = env.transition_model()
    check("gridworld: P sums to 1 over next states", np.allclose(P.sum(axis=2), 1.0))
    check("gridworld: P is non-negative", bool((P >= 0).all()))

    s, a = env.to_index((2, 0)), 1  # RIGHT from a free cell
    counts = np.zeros(env.n_states)
    sampler = GridWorld(slip_prob=0.2)
    for i in range(20_000):
        sampler.reset(seed=i)
        sampler._agent = env.to_cell(s)
        nxt, _, _, _, _ = sampler.step(a)
        counts[nxt] += 1
    empirical = counts / counts.sum()
    check("gridworld: sampled dynamics match transition_model",
          np.allclose(empirical, P[s, a], atol=0.02),
          f"max diff {np.abs(empirical - P[s, a]).max():.4f}")


def test_value_iteration_fixed_point():
    """V* must satisfy the Bellman optimality equation it was derived from."""
    from environments.gridworld import GridWorld, value_iteration

    env = GridWorld(slip_prob=0.1)
    gamma = 0.95
    _, V = value_iteration(env, gamma=gamma)
    P, R = env.transition_model()
    residual = max(
        abs(float((R[s] + gamma * P[s] @ V).max()) - float(V[s]))
        for s in env.decision_states()
    )
    check("value iteration: Bellman residual ~ 0", residual < 1e-6, f"residual {residual:.2e}")


def test_q_learning_deterministic_convergence():
    """On a deterministic grid with full coverage, Q must reach Q* almost exactly."""
    from environments.gridworld import GridWorld, value_iteration
    from q_learning.agent import QLearningAgent

    env = GridWorld(slip_prob=0.0, max_steps=60)
    q_star, _ = value_iteration(env, gamma=0.95)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.95,
                           epsilon_start=1.0, epsilon_end=0.2,
                           epsilon_decay=0.999, seed=0)
    for ep in range(4000):
        state, _ = env.reset(seed=ep, options={"exploring_start": True})
        while True:
            a = agent.act(state)
            nxt, r, term, trunc, _ = env.step(a)
            agent.update(state, a, r, nxt, term)
            state = nxt
            if term or trunc:
                break
        agent.decay_epsilon()
    ds = env.decision_states()
    gap = float(np.abs(agent.q_table[ds] - q_star[ds]).max())
    check("q-learning: converges to Q* on a deterministic grid", gap < 0.02, f"gap {gap:.4f}")


def test_discounted_returns():
    """G_t must match the closed form on a known reward sequence."""
    from reinforce.agent import ReinforceAgent

    agent = ReinforceAgent(obs_dim=2, n_actions=2, gamma=0.9, seed=0)
    got = agent.discounted_returns([1.0, 2.0, 3.0]).numpy()
    want = np.array([1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3.0], dtype=np.float32)
    check("reinforce: discounted returns", np.allclose(got, want, atol=1e-5), f"{got} vs {want}")


