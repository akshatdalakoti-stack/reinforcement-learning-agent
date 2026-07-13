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


