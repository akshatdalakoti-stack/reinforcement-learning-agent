"""Train n-step Actor-Critic on CartPole.

    python -m actor_critic.train --episodes 800 --n-steps 8

Structurally this loop differs from :mod:`reinforce.train` in one important
way: ``agent.update`` fires whenever ``n_steps`` transitions have accumulated,
*not* at episode boundaries.  That is the practical consequence of replacing
the Monte-Carlo return with a bootstrapped advantage.
"""

from __future__ import annotations

import argparse

from actor_critic.agent import ActorCriticAgent
from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advantage Actor-Critic")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-env-steps", type=int, default=200_000)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=32,
                   help="rollout length; 1 = pure TD, large = closer to REINFORCE")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--no-advantage-norm", action="store_true")
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
    p.add_argument("--run-name", type=str, default="actor_critic_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


