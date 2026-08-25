"""Experiment 3 -- the target-update frequency trade-off.

Claim
-----
The target network trades *speed* for *stability*, and the update frequency is
the dial between them.

Why a target network is needed at all
-------------------------------------
Without one, the regression target is built from the very network being
updated::

    L = ( Q_θ(s,a) - [r + gamma * max_a' Q_θ(s',a')] )^2

Since ``s`` and ``s'`` are neighbouring states, a gradient step that raises
``Q(s,a)`` also raises ``Q(s',a')`` -- which raises the target -- which raises
``Q(s,a)`` again.  The fixed point can run away.  This is the *deadly triad*:
function approximation + bootstrapping + off-policy updates.  Any two are safe;
all three together have no convergence guarantee.

Freezing the target for ``k`` gradient steps turns the problem into a sequence
of ``k``-step **supervised regressions against a stationary target**, restoring
the assumption gradient descent needs.

The trade-off
-------------
* ``k`` too small -> the target moves almost as fast as the estimate; barely
  different from having no target network.
* ``k`` too large -> the targets are stale, so value information propagates
  backwards through the state space very slowly.

Design
------
Sweep ``k`` in gradient steps, plus a no-target-network control, across seeds.
Track final performance, sample efficiency, instability, and the mean predicted
Q-value -- the last of which is where divergence shows up first.

    python -m experiments.exp3_target_network_frequency --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from dqn.train import main as dqn_main
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import PALETTE, plot_variance_bands, save_figure
from evaluation.utils import PLOT_DIR, resolve_path
from experiments._common import (
    banner,
    collapse_count,
    describe,
    save_summary,
    stability_score,
)


