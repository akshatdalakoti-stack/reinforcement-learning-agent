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

