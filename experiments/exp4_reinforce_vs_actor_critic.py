"""Experiment 4 -- variance: Monte-Carlo returns vs a learned baseline.

Claim
-----
REINFORCE and Actor-Critic optimise the same objective by the same theorem.
They differ in what multiplies ``grad log pi(a|s)``, and that difference is
almost entirely a variance story.

Two measurements
----------------
1. **Direct gradient variance.**  Hold the parameters fixed, estimate the
   gradient many times from independent rollouts, and measure how much the
   estimates agree (mean pairwise cosine similarity, and
   ``||mean grad|| / mean ||grad||``).  This measures the estimator itself,
   with learning dynamics factored out.

2. **Learning-curve spread across seeds.**  What that variance costs in
   practice: how differently identical configurations behave when only the
   random seed changes.

Variants
--------
* ``REINFORCE (no baseline)``   -- the textbook estimator, weights are raw ``G_t``
* ``REINFORCE (normalised)``    -- constant baseline + scaling
* ``Actor-Critic (n=32)``       -- learned state-dependent baseline ``V(s)``

    python -m experiments.exp4_reinforce_vs_actor_critic --seeds 4
"""

from __future__ import annotations

import argparse

import numpy as np

from actor_critic.train import main as ac_main
from environments.registry import make_env
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_variance_bands
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, describe, save_summary, steps_to_threshold
from reinforce.train import build_agent as build_reinforce
from reinforce.train import main as reinforce_main
from reinforce.train import measure_gradient_variance, parse_args as reinforce_args


