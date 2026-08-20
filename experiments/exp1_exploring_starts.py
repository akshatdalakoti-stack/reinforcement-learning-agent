"""Experiment 1 -- what actually limits tabular Q-learning?

Claim
-----
With enough episodes the Bellman update is not the bottleneck; *coverage* is.
Q-learning converges to Q* only for state-action pairs it visits infinitely
often, and a fixed start state means the pairs off the optimal path are visited
almost never.

Design
------
Four configurations on the same GridWorld, several seeds each:

* fixed start,     constant alpha
* fixed start,     1/n(s,a) alpha  (satisfies the Robbins-Monro conditions)
* exploring starts, constant alpha
* exploring starts, 1/n(s,a) alpha

Because GridWorld is small enough to solve exactly with value iteration, we can
report the *true* error rather than a proxy:

* ``|Q - Q*|_max`` over decision states
* how many states the greedy policy gets optimal
* state-action coverage

    python -m experiments.exp1_exploring_starts --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.gridworld import GridWorld, value_iteration
from evaluation.plotting import PALETTE, save_figure
from evaluation.utils import PLOT_DIR, resolve_path, set_seed
from experiments._common import banner, save_summary
from q_learning.agent import QLearningAgent


