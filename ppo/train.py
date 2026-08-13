"""Train PPO on LunarLander (or any discrete Gymnasium environment).

    python -m ppo.train --env LunarLander-v3 --total-steps 600000
    python -m ppo.train --env CartPole-v1  --total-steps 150000

The loop shape changes here for the last time.  Q-learning updated per step,
REINFORCE per episode, A2C per n steps -- PPO collects a *fixed-size rollout*
that deliberately ignores episode boundaries, then optimises it repeatedly.
Progress is therefore measured in environment steps, not episodes: an episode
is no longer the unit of anything.

If Box2D is unavailable the script falls back to a classic-control environment
rather than failing, and says so.
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.registry import env_available, env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed
from ppo.agent import PPOAgent

FALLBACK_ENV = "CartPole-v1"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO on a discrete Gymnasium environment")
    p.add_argument("--env", type=str, default="LunarLander-v3")
    p.add_argument("--total-steps", type=int, default=600_000)
    p.add_argument("--rollout-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None,
                   help="early-stop a update once approx KL exceeds this")
    p.add_argument("--no-value-clip", action="store_true")
    p.add_argument("--no-advantage-norm", action="store_true")
    p.add_argument("--no-lr-anneal", action="store_true")
    p.add_argument("--separate-networks", action="store_true",
                   help="use two towers instead of a shared trunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="env steps between greedy evaluations during training; "
                        "0 disables. Training reward reflects the exploring "
                        "policy, so this is the honest progress signal.")
    p.add_argument("--eval-rollouts", type=int, default=5,
                   help="episodes per periodic evaluation")
    p.add_argument("--run-name", type=str, default="ppo_lunarlander")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def resolve_env_id(requested: str, quiet: bool = False) -> str:
    if env_available(requested):
        return requested
    if not quiet:
        print(f"  ! {requested} is unavailable (Box2D missing?); "
              f"falling back to {FALLBACK_ENV}")
    return FALLBACK_ENV


