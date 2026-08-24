"""Experiment 2 -- a 2x2 factorial over replay and the target network.

Why factorial
-------------
The obvious comparison -- online DQN vs the full agent -- changes two things at
once, so it cannot tell you which one mattered.  Running all four corners does:

                      no target network      target network
    no replay         (a) stage-2 baseline   (c) target only
    replay            (b) replay only        (d) full agent, stage 4

* (d) - (b) isolates the **target network**
* (d) - (c) isolates **replay**
* (a) is the naive deep Q-learner that stage 2 warns about

Everything else -- network, optimiser, exploration schedule, seed, environment
step budget, and the number of gradient steps taken -- is identical across all
four.  The only difference is *which transitions each gradient step sees* and
*which network produces the bootstrap target*.

A note on batch size: the no-replay variants necessarily train on a batch of
one, because a decorrelated batch of 128 is exactly the thing a replay buffer
provides.  That is a consequence of the ablation, not a confound in it.

Metric
------
Progress is measured by **periodic greedy evaluation**, not by training reward.
With ``epsilon = 0.05`` a single random action ends a CartPole episode, so a
fully-trained DQN's *training* reward oscillates between ~90 and ~250 while its
greedy policy scores a flat 500.  Judging this ablation on training reward would
mostly measure exploration noise.

    python -m experiments.exp2_replay_ablation --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from dqn.train import main as dqn_main
from evaluation.plotting import PALETTE, plot_variance_bands, save_figure
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import banner, describe, save_summary

