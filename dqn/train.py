"""Train DQN on CartPole -- the driver for stages 2, 3 and 4.

    # stage 2: plain online DQN, no replay, no target net
    python -m dqn.train --no-replay --no-target-network --run-name dqn_online

    # stage 3: + experience replay
    python -m dqn.train --no-target-network --run-name dqn_replay

    # stage 4: the full agent
    python -m dqn.train --run-name dqn_full

The three configurations share every other hyperparameter and the same seed, so
differences in the learning curves are attributable to the ablated component.
"""

from __future__ import annotations

import argparse

from dqn.agent import DQNAgent
from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN on a Gymnasium environment")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-env-steps", type=int, default=200_000,
                   help="hard budget so ablations get equal sample allowance")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=4,
                   help="env steps between gradient steps; >1 avoids "
                        "overfitting the replay buffer")
    p.add_argument("--target-update-freq", type=int, default=250,
                   help="in GRADIENT steps, not env steps")
    p.add_argument("--tau", type=float, default=0.0,
                   help="> 0 switches to Polyak soft target updates")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    p.add_argument("--grad-clip", type=float, default=10.0)

    p.add_argument("--no-replay", action="store_true",
                   help="stage 2: train on the newest transition only")
    p.add_argument("--no-target-network", action="store_true",
                   help="bootstrap from the online network itself")
    p.add_argument("--double-dqn", action="store_true")

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
    p.add_argument("--run-name", type=str, default="dqn_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_agent(args: argparse.Namespace, spec, device) -> DQNAgent:
    return DQNAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        grad_clip=args.grad_clip,
        use_replay=not args.no_replay,
        use_target_network=not args.no_target_network,
        double_dqn=args.double_dqn,
        device=device,
        seed=args.seed,
    )


