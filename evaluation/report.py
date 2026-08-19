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

CARTPOLE_CURVES = [
    ("DQN (no replay, no target)", "dqn_online"),
    ("DQN + replay", "dqn_replay"),
    ("DQN + replay + target", "dqn_cartpole"),
    ("REINFORCE", "reinforce_cartpole"),
    ("Actor-Critic", "actor_critic_cartpole"),
    ("PPO", "ppo_cartpole"),
]


def load_run(run_name: str) -> tuple[dict, dict] | None:
    """Return ``(summary_json, csv_columns)`` for a run, or None if absent."""
    log_dir = resolve_dir(*LOG_DIR)
    json_path = os.path.join(log_dir, f"{run_name}.json")
    csv_path = os.path.join(log_dir, f"{run_name}.csv")
    if not (os.path.exists(json_path) and os.path.exists(csv_path)):
        return None
    with open(json_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    return summary, MetricsRecorder.load_csv(csv_path)


def recorder_from_csv(name: str, cols: dict) -> MetricsRecorder:
    """Wrap raw CSV columns back into a recorder for plotting."""
    rec = MetricsRecorder(run_name=name)
    rec.episodes = [
        {"reward": r, "env_steps": s, "length": l}
        for r, s, l in zip(cols["reward"], cols["env_steps"], cols["length"])
    ]
    return rec


