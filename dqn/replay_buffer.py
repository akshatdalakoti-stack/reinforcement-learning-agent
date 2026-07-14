"""Experience replay, written from scratch.

The problem replay solves
-------------------------
Online DQN trains on transitions in the order they happen.  That breaks two
assumptions that gradient descent quietly relies on:

1. **The samples are not i.i.d.**  Consecutive states in an episode are almost
   identical, so a minibatch of them carries far less information than its size
   suggests and the gradient is strongly biased toward whatever the agent
   happens to be doing right now.
2. **The data distribution is non-stationary.**  Improving the policy changes
   which states are visited, which changes the training distribution, which
   changes the policy.  The network chases its own tail, and can *catastrophically
   forget* how to value states it no longer visits.

A replay buffer decouples "collecting experience" from "learning from
experience".  Transitions are written to a circular buffer and sampled
uniformly at random, so each minibatch mixes many different episodes and many
different past policies.

Why this is legal at all
------------------------
Q-learning is **off-policy**: its target ``r + gamma * max_a' Q(s',a')`` never
refers to the policy that generated the action.  So a transition collected by a
policy from 10,000 steps ago is still a valid sample of the environment's
dynamics.  That is exactly why REINFORCE (stage 5) cannot use a replay buffer:
its gradient estimator is only unbiased for data from the *current* policy.

Implementation note
-------------------
Storage is a set of preallocated NumPy arrays rather than a ``deque`` of tuples.
A buffer of 100k CartPole transitions is a few megabytes this way, and sampling
is one fancy-index instead of 64 Python-level tuple unpackings.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """A single environment step: the atom of off-policy learning."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class Batch(NamedTuple):
    """A minibatch of transitions, already on the training device."""

    states: torch.Tensor        # (B, obs_dim) float32
    actions: torch.Tensor       # (B, 1)       int64
    rewards: torch.Tensor       # (B, 1)       float32
    next_states: torch.Tensor   # (B, obs_dim) float32
    dones: torch.Tensor         # (B, 1)       float32, 1.0 where terminated


