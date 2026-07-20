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


