"""Thin factories around Gymnasium.

Gymnasium supplies the *simulators* (CartPole, LunarLander, ...); every learning
algorithm in this repository is written by hand.  This module exists only so
the training scripts do not each re-implement seeding and space introspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class EnvSpec:
    """The handful of facts an agent needs about an environment."""

    env_id: str
    obs_dim: int
    n_actions: int
    reward_threshold: float | None
    max_episode_steps: int | None


def make_env(
    env_id: str = "CartPole-v1",
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Create a Gymnasium environment and seed its RNGs.

    Seeding happens through ``reset(seed=...)`` plus the action-space seed,
    which is what Gymnasium's own docs prescribe; a later ``reset()`` without a
    seed then continues the seeded stream rather than restarting it.
    """
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


