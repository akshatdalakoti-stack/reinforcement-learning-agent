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


