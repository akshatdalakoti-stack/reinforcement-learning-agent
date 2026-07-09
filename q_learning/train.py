"""Train tabular Q-learning on GridWorld.

    python -m q_learning.train --episodes 2000 --seed 0

Besides the learning curve this script produces the two artefacts that make the
tabular case worth studying: a picture of the greedy policy, and a comparison
against the exact optimum from value iteration.  Being able to say "the learned
Q is within 0.01 of Q*" is a luxury we lose the moment the table becomes a
neural network, so it is worth collecting while we can.
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.gridworld import GridWorld, value_iteration
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import (
    plot_gridworld_policy,
    plot_learning_curve,
    plot_q_convergence,
)
from evaluation.utils import LOG_DIR, PLOT_DIR, resolve_dir, resolve_path, set_seed
from q_learning.agent import QLearningAgent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular Q-learning on GridWorld")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--alpha-schedule", choices=["constant", "visit_count"],
                   default="constant",
                   help="'visit_count' uses 1/n(s,a)**p, satisfying the "
                        "Robbins-Monro conditions for convergence")
    p.add_argument("--alpha-power", type=float, default=0.75)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--slip-prob", type=float, default=0.1,
                   help="0 makes the grid deterministic")
    p.add_argument("--exploring-starts", action="store_true",
                   help="begin each episode in a random cell (fixes coverage of "
                        "states the greedy policy never reaches)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=50, help="moving-average window")
    p.add_argument("--run-name", type=str, default="q_learning_gridworld")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


