"""Evaluate a trained agent, and a CLI to evaluate a saved checkpoint.

Training reward is *not* final performance.  During training the agent is still
exploring (epsilon-greedy, or sampling from a stochastic policy), so the logged
episode reward measures the behaviour policy, not the policy we would deploy.
Everything here runs the greedy/deterministic policy with exploration off, on
seeds the agent did not train against.

    python -m evaluation.evaluate --checkpoint checkpoints/dqn_cartpole.pt \\
        --env CartPole-v1 --episodes 20

Every agent in this repository exposes ``act(state, greedy=True)``, so one
function covers the tabular and the neural cases alike.
"""

from __future__ import annotations

import argparse
from typing import Any, Protocol

import numpy as np


class Actor(Protocol):
    def act(self, state, greedy: bool = False) -> int: ...


def evaluate_agent(
    agent: Actor,
    env,
    episodes: int = 20,
    seed: int = 10_000,
    max_steps: int | None = None,
    render: bool = False,
) -> dict[str, Any]:
    """Roll out the greedy policy and summarise the return distribution.

    The spread matters as much as the mean: an agent averaging 400 on CartPole
    with a standard deviation of 180 has not solved the task, it has learned a
    policy that works from some initial states and fails from others.
    """
    rewards: list[float] = []
    lengths: list[int] = []

    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        total, steps = 0.0, 0
        while True:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if render:
                env.render()
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break
        rewards.append(total)
        lengths.append(steps)

    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "episodes": episodes,
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "min_reward": float(arr.min()),
        "max_reward": float(arr.max()),
        "median_reward": float(np.median(arr)),
        "mean_length": float(np.mean(lengths)),
        "rewards": rewards,
    }


class PeriodicEvaluator:
    """Run a short greedy evaluation every ``every`` environment steps.

    Why this exists: on CartPole with ``epsilon = 0.05``, a single random action
    can end an episode, so the *training* reward of a fully-trained DQN keeps
    oscillating between ~90 and ~250 while its greedy policy scores a flat 500.
    Judging an ablation by end-of-training episode reward therefore measures
    exploration noise as much as learning.  Sampling the greedy policy on a
    fixed schedule gives a signal that reflects what the agent has actually
    learned, and makes ``best`` and ``final`` meaningful summary numbers.

    The evaluation environment is separate from the training one and seeded from
    a disjoint range, so evaluating never perturbs the training RNG stream.
    """

    def __init__(self, env, every: int = 5_000, episodes: int = 5, seed: int = 500_000):
        self.env = env
        self.every = int(every)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self._next_at = self.every
        self.history: list[tuple[int, float]] = []

    def maybe_evaluate(self, agent: Actor, env_steps: int) -> float | None:
        """Evaluate if the step counter has passed the next checkpoint."""
        if self.every <= 0 or env_steps < self._next_at:
            return None
        while self._next_at <= env_steps:
            self._next_at += self.every
        result = evaluate_agent(
            agent, self.env, episodes=self.episodes, seed=self.seed + len(self.history) * 100
        )
        self.history.append((env_steps, result["mean_reward"]))
        return result["mean_reward"]

    @property
    def best(self) -> float:
        return max((r for _, r in self.history), default=float("nan"))

    @property
    def final(self) -> float:
        return self.history[-1][1] if self.history else float("nan")

    def steps_to(self, threshold: float) -> int | None:
        """Environment steps at the *first* evaluation reaching ``threshold``."""
        for steps, reward in self.history:
            if reward >= threshold:
                return steps
        return None

    def steps_to_sustained(self, threshold: float) -> int | None:
        """Environment steps from which *every later* evaluation holds ``threshold``.

        The honest sample-efficiency number.  These learning curves are strongly
        non-monotonic: on CartPole both DQN and Actor-Critic touch 485-500 within
        the first 10k steps, collapse back below 100, and only recover much
        later.  ``steps_to`` would credit them with solving the task almost
        immediately, which is not a claim the run supports.  This asks the
        stricter question -- when did it start working and *stay* working.
        """
        answer = None
        for steps, reward in reversed(self.history):
            if reward < threshold:
                break
            answer = steps
        return answer

    def fraction_above(self, threshold: float) -> float:
        """Share of evaluations at or above ``threshold`` -- a stability measure."""
        if not self.history:
            return float("nan")
        return sum(r >= threshold for _, r in self.history) / len(self.history)


