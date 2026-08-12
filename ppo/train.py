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

