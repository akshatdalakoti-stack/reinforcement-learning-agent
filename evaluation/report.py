"""Rebuild the summary table and progression figure from existing logs.

    python -m evaluation.report

``experiments/run_all.py`` produces the same artefacts but retrains everything.
This script only reads ``results/logs/*.csv`` and ``*.json``, so it is the right
tool when you want to regenerate the comparison after adding one run, or to
recover the table without spending twenty minutes of CPU.

Output:
* a Markdown table on stdout, ready to paste into the README
* ``results/plots/progression_cartpole.png`` -- every stage that ran on
  CartPole, plotted against environment steps so sample efficiency is
  directly comparable
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_comparison
from evaluation.utils import LOG_DIR, PLOT_DIR, resolve_dir, resolve_path

# (run_name, display name, stage) in progression order.
KNOWN_RUNS = [
    ("q_learning_gridworld", "Q-learning (tabular)", 1),
    ("dqn_online", "DQN (no replay, no target)", 2),
    ("dqn_replay", "DQN + replay", 3),
    ("dqn_cartpole", "DQN + replay + target", 4),
    ("reinforce_cartpole", "REINFORCE", 5),
    ("actor_critic_cartpole", "Actor-Critic (n-step)", 6),
    ("ppo_cartpole", "PPO (CartPole control)", 7),
    ("ppo_lunarlander", "PPO", 7),
]

