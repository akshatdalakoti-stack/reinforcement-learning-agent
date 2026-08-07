"""Rollout storage and Generalised Advantage Estimation.

What gets stored, and why each field is needed
----------------------------------------------
=================  ==========================================================
``states``         inputs for re-evaluating the policy in later epochs
``actions``        which action to re-score
``rewards``        to build returns and advantages
``dones``          to cut the bootstrap chain at episode boundaries
``log_probs``      ``log pi_old(a|s)`` -- the ratio's denominator, frozen at
                   collection time
``values``         ``V_old(s)``, used by GAE and by value clipping
=================  ==========================================================

``log_probs`` and ``values`` are the two fields that make this a *PPO* buffer
rather than an A2C one.  PPO takes several gradient steps on the same batch, so
after the first one the network's policy is no longer the policy that collected
the data.  Storing ``log pi_old`` is what lets PPO measure -- and then bound --
how far it has drifted.

Generalised Advantage Estimation
--------------------------------
A2C picks a single ``n`` for its n-step advantage and lives with that
bias/variance trade-off.  GAE takes an exponentially-weighted average over
*all* n simultaneously::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = sum_{l>=0} (gamma * lambda)^l * delta_{t+l}

which collapses to a backward recursion::

    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

``lambda = 0`` gives the one-step TD error (low variance, high bias);
``lambda = 1`` gives the Monte-Carlo advantage (unbiased, high variance).
``lambda ~ 0.95`` is the usual compromise -- it is a second, finer-grained dial
on top of ``gamma``.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: bool,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` for one rollout.

    ``values`` holds ``V(s_t)`` for the ``T`` stored states; ``last_value`` is
    ``V(s_T)`` for the state the rollout stopped *at*, which is what lets a
    rollout end mid-episode without corrupting the estimates.

    ``returns = advantages + values`` is the regression target for the value
    head.  Using this rather than the raw discounted return keeps the critic's
    target consistent with the advantages the actor is being trained on.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - float(last_done)
            next_value = last_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns.astype(np.float32)


class RolloutBuffer:
    """Fixed-length on-policy storage.

    Unlike :class:`~dqn.replay_buffer.ReplayBuffer` this is emptied after every
    update.  It has to be: PPO's ratio is only meaningful against the policy
    that actually generated the data, so once the policy has moved past the
    trust region the data is no longer usable, however recent it looks.
    """

    def __init__(self, size: int, obs_dim: int, device: torch.device | str = "cpu") -> None:
        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)

        self.states = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)

        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)

        self._pos = 0
        self.ready = False

    def __len__(self) -> int:
        return self._pos

    @property
    def is_full(self) -> bool:
        return self._pos >= self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        if self.is_full:
            raise RuntimeError("rollout buffer is full; call finish() and reset()")
        i = self._pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self._pos += 1

    def finish(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute advantages and returns for the collected slice."""
        n = self._pos
        adv, ret = compute_gae(
            self.rewards[:n], self.values[:n], self.dones[:n],
            last_value, last_done, gamma, gae_lambda,
        )
        self.advantages[:n] = adv
        self.returns[:n] = ret
        self.ready = True

