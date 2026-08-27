"""Experiment 5 -- what does PPO's clipping actually prevent?

Claim
-----
Clipping is what makes multi-epoch reuse of a rollout safe.  Without it, taking
many gradient steps on the same data drives the policy far from the one that
collected it, the importance ratios blow up, and performance collapses.

Design
------
The clean control is not "PPO vs A2C" -- that changes several things at once.
It is **the same PPO agent, same rollouts, same number of epochs, with the clip
range set so wide it never binds** (``--clip-epsilon 10``).  That variant is
exactly "importance-weighted policy gradient with 10 epochs of reuse", which is
the thing clipping was invented to fix.

Settings compared
-----------------
* ``clip=0.2, 10 epochs``   -- standard PPO
* ``clip=inf, 10 epochs``   -- no trust region: the failure case
* ``clip=0.2,  1 epoch``    -- clipping present but nothing to protect against
* ``clip=0.05, 10 epochs``  -- an over-tight region: stable but slow

Diagnostics
-----------
``approx_kl`` (how far the policy moved), ``clip_fraction`` (how often the
bound bound), and policy entropy (whether the policy is collapsing).

    python -m experiments.exp5_ppo_clipping --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_variance_bands
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, collapse_count, describe, save_summary
from ppo.train import main as ppo_main

# slug -> (display label, flags).  The slug keys the per-run log filenames, so
# it must be unique: two variants share the label prefix "PPO clip=0.2".
VARIANTS = {
    "clip0.2_ep10": ("PPO clip=0.2, 10 epochs", ["--clip-epsilon", "0.2", "--n-epochs", "10"]),
    "noclip_ep10": ("no clipping, 10 epochs", ["--clip-epsilon", "10.0", "--n-epochs", "10"]),
    "clip0.2_ep1": ("PPO clip=0.2, 1 epoch", ["--clip-epsilon", "0.2", "--n-epochs", "1"]),
    "clip0.05_ep10": ("PPO clip=0.05, 10 epochs", ["--clip-epsilon", "0.05", "--n-epochs", "10"]),
}


