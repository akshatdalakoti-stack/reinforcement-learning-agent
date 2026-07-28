"""Train REINFORCE on CartPole, and measure its gradient variance.

    python -m reinforce.train --episodes 1500 --baseline norm

The variance analysis is the point of this stage, not an extra.  REINFORCE
usually *does* learn CartPole; what it does badly is learn it *reliably*.  This
script quantifies that in two ways:

* ``--gradient-variance N`` estimates the policy gradient N times from N
  independent single-episode rollouts **at the same parameters**, then reports
  the cosine similarity between those estimates.  If single-trajectory gradients
  pointed the same way, similarity would be near 1.0.  It is not.
* the returned recorder keeps per-episode reward, so
  ``experiments/exp4_reinforce_vs_actor_critic.py`` can plot the across-seed
  spread that this variance produces.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed
from reinforce.agent import ReinforceAgent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REINFORCE on a Gymnasium environment")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--max-env-steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--baseline", choices=["none", "mean", "norm"], default="norm")
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--batch-episodes", type=int, default=1,
                   help="trajectories averaged per update; >1 cuts variance as 1/N")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--gradient-variance", type=int, default=0,
                   help="estimate the gradient this many times at fixed parameters")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="env steps between greedy evaluations during training; "
                        "0 disables. Training reward reflects the exploring "
                        "policy, so this is the honest progress signal.")
    p.add_argument("--eval-rollouts", type=int, default=5,
                   help="episodes per periodic evaluation")
    p.add_argument("--run-name", type=str, default="reinforce_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_agent(args, spec, device) -> ReinforceAgent:
    return ReinforceAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        baseline=args.baseline,
        entropy_coef=args.entropy_coef,
        batch_episodes=args.batch_episodes,
        grad_clip=args.grad_clip,
        device=device,
        seed=args.seed,
    )


def collect_episode(agent: ReinforceAgent, env, seed: int | None = None):
    """Run one full trajectory, recording log-probs inside the agent."""
    state, _ = env.reset(seed=seed)
    rewards: list[float] = []
    while True:
        action = agent.act_and_record(state)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    agent.finish_episode(rewards)
    return rewards


