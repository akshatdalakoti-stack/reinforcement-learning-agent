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


