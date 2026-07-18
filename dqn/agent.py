"""Deep Q-Network -- stages 2, 3 and 4 in a single agent.

Stage 2 (DQN)              ``use_replay=False, use_target_network=False``
Stage 3 (+ replay)         ``use_replay=True,  use_target_network=False``
Stage 4 (+ target network) ``use_replay=True,  use_target_network=True``

Keeping them in one class is deliberate: the ablation scripts then compare the
*same* optimiser, network, exploration schedule and seed with exactly one idea
switched off, which is the only way the comparison means anything.

The loss
--------
For a minibatch of transitions ``(s, a, r, s', d)``::

    y      = r + gamma * (1 - d) * max_a' Q_target(s', a')      # no gradient
    L(θ)   = SmoothL1( Q_online(s, a) - y )

The ``detach``/``no_grad`` on ``y`` is not an optimisation -- it is the
algorithm.  Q-learning is a fixed-point iteration ``Q <- T Q``, not the
minimisation of a loss; if gradients flowed into the target the network could
trivially reduce the loss by dragging ``y`` toward its prediction instead of
the other way round.

Why Huber rather than MSE
-------------------------
Early in training the TD error can be enormous, and MSE turns a large error
into a proportionally large gradient, which blows the network up.  Huber
(``SmoothL1``) is quadratic near zero and linear beyond ``delta=1``, so it
bounds the gradient magnitude while keeping the same minimiser.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn.network import QNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """Value-based agent for discrete actions.

    Parameters
    ----------
    use_replay:
        ``True`` samples uniformly from the buffer.  ``False`` trains on the
        single most recent transition -- "online DQN", the stage-2 baseline.
    use_target_network:
        ``True`` bootstraps from a frozen copy of the network that is
        synchronised every ``target_update_freq`` gradient steps.  ``False``
        bootstraps from the online network itself, so the regression target
        moves at every update.
    target_update_freq:
        Hard-sync period, in *gradient steps*.  Small values approach the
        no-target-network case; large values make the target stale but stable.
    tau:
        If > 0, use Polyak soft updates ``θ_target <- τθ + (1-τ)θ_target``
        every step instead of periodic hard copies.
    double_dqn:
        Decouple action *selection* from action *evaluation* in the target:
        ``y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))``.  The plain
        ``max`` is systematically optimistic, because the max of noisy estimates
        is biased above the true max; Double DQN removes most of that bias.
    epsilon_decay_steps:
        Exploration is annealed linearly in *environment steps*, not episodes.
        Episodes change length dramatically during training on CartPole, so an
        episode-based schedule would decay at an unpredictable rate.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        learning_starts: int = 1_000,
        train_freq: int = 1,
        target_update_freq: int = 500,
        tau: float = 0.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        grad_clip: float = 10.0,
        use_replay: bool = True,
        use_target_network: bool = True,
        double_dqn: bool = False,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.target_update_freq = int(target_update_freq)
        self.tau = float(tau)
        self.grad_clip = float(grad_clip)
        self.use_replay = bool(use_replay)
        self.use_target_network = bool(use_target_network)
        self.double_dqn = bool(double_dqn)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.epsilon = float(epsilon_start)

        self.online_net = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        # The target network is always constructed, even when unused, so that
        # checkpoints have a stable shape and `_td_target` stays branch-free.
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(lr))

        # Even the no-replay variant writes to a buffer; it just reads back only
        # the newest entry. One code path, one difference.
        self.buffer = ReplayBuffer(
            capacity=buffer_size if self.use_replay else max(2, batch_size),
            obs_dim=obs_dim,
            seed=seed,
            device=self.device,
        )

        self.rng = np.random.default_rng(seed)
        self.env_steps = 0
        self.gradient_steps = 0
        self.target_syncs = 0
        self._recent_losses: list[float] = []
        self._recent_q: list[float] = []

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy over the online network's Q-values."""
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.online_net(obs.unsqueeze(0))
        return int(q_values.argmax(dim=1).item())

    def update_epsilon(self) -> float:
        """Linear anneal from ``epsilon_start`` to ``epsilon_end``."""
        frac = min(1.0, self.env_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        return self.epsilon

    # ---------------------------------------------------------------- learning
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Record one transition and advance the step/exploration counters."""
        self.buffer.push(state, action, reward, next_state, terminated)
        self.env_steps += 1
        self.update_epsilon()

    def train_step(self) -> float | None:
        """One gradient step; ``None`` when it is too early to learn."""
        if self.env_steps < self.learning_starts:
            return None
        if self.env_steps % self.train_freq != 0:
            return None
        if self.use_replay and len(self.buffer) < self.batch_size:
            return None

        batch = (
            self.buffer.sample(self.batch_size)
            if self.use_replay
            else self.buffer.latest(1)   # stage 2: the transition just observed
        )

        # Q(s, a) for the actions actually taken.
        q_pred = self.online_net(batch.states).gather(1, batch.actions)

        with torch.no_grad():
            target_q = self._bootstrap_values(batch.next_states)
            # (1 - done) zeroes the future for genuinely terminal transitions.
            y = batch.rewards + self.gamma * (1.0 - batch.dones) * target_q

        loss = F.smooth_l1_loss(q_pred, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.gradient_steps += 1
        self._sync_target()

        loss_value = float(loss.item())
        self._recent_losses.append(loss_value)
        self._recent_q.append(float(q_pred.mean().item()))
        return loss_value

    def _bootstrap_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """max_a' Q(s', a'), from whichever network is acting as the target."""
        net = self.target_net if self.use_target_network else self.online_net
        if self.double_dqn:
            best = self.online_net(next_states).argmax(dim=1, keepdim=True)
            return net(next_states).gather(1, best)
        return net(next_states).max(dim=1, keepdim=True).values

    def _sync_target(self) -> None:
        if not self.use_target_network:
            return
        if self.tau > 0.0:
            with torch.no_grad():
                for tgt, src in zip(self.target_net.parameters(),
                                    self.online_net.parameters()):
                    tgt.mul_(1.0 - self.tau).add_(self.tau * src)
            return
        if self.gradient_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_syncs += 1

    # ------------------------------------------------------------ diagnostics
    def pop_stats(self) -> dict[str, float]:
        """Mean loss and mean predicted Q since the last call, then reset.

        Mean Q is worth watching on its own: runaway growth in the predicted
        value -- far above the largest return the environment can pay -- is the
        signature of the deadly triad diverging, and it shows up in this number
        long before the reward curve collapses.
        """
        stats = {
            "loss": float(np.mean(self._recent_losses)) if self._recent_losses else float("nan"),
            "mean_q": float(np.mean(self._recent_q)) if self._recent_q else float("nan"),
        }
        self._recent_losses.clear()
        self._recent_q.clear()
        return stats

    # -------------------------------------------------------------- checkpoint
    def config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "use_replay": self.use_replay,
            "use_target_network": self.use_target_network,
            "double_dqn": self.double_dqn,
            "target_update_freq": self.target_update_freq,
            "tau": self.tau,
            "hidden_sizes": [
                m.out_features for m in self.online_net.net if isinstance(m, nn.Linear)
            ][:-1],
        }

