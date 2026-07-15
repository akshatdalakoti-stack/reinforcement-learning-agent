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


class ReplayBuffer:
    """Fixed-capacity circular buffer with uniform random sampling.

    Parameters
    ----------
    capacity:
        How many transitions to keep.  This is a real hyperparameter, not a
        memory detail: too small and the buffer is still highly correlated and
        forgets the past; too large and the agent keeps training on transitions
        from a policy so old they are no longer informative.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)

        self.states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self._pos = 0        # where the next write goes
        self._size = 0       # how many slots are valid
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest once full.

        ``done`` must mean "the MDP terminated", not "the episode stopped".
        A transition truncated by a time limit still has a future worth
        bootstrapping through; storing it as ``done=True`` teaches the agent
        that the world ends at step 500, which caps every value estimate.
        """
        i = self._pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample ``batch_size`` transitions *with* replacement.

        Sampling with replacement keeps this O(batch) regardless of buffer size
        and, at realistic buffer sizes, duplicates are rare enough not to matter.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self._size, size=batch_size)
        return self._to_batch(idx)

