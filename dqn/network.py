"""The function approximator that replaces the Q-table.

A table stores one independent number per ``(s, a)`` pair.  CartPole's state is
four real numbers, so the table would need infinitely many rows -- and even if
we discretised it, nothing learned about the state ``(0.10, 0.02, 0.01, 0.30)``
would transfer to ``(0.101, 0.02, 0.01, 0.30)``, because table cells do not
talk to each other.

A neural network replaces "one parameter per state" with "one shared parameter
vector for all states".  That buys *generalisation*, and the whole of stages
3 and 4 is about paying for it: the moment estimates share parameters, updating
one state's value silently moves every other state's value too.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP mapping a state vector to one Q-value per discrete action.

    The head outputs all ``n_actions`` values in a single forward pass, which is
    what makes ``max_a' Q(s', a')`` one cheap operation instead of ``n_actions``
    separate evaluations.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)

        layers: list[nn.Module] = []
        last = self.obs_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        layers.append(nn.Linear(last, self.n_actions))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Orthogonal init with a small output gain: the last layer starts near
        # zero, so early Q-values are small and the initial greedy policy is
        # effectively arbitrary rather than confidently wrong.
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``(batch, obs_dim)`` -> ``(batch, n_actions)``."""
        return self.net(obs)

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> int:
        """argmax_a Q(s,a) for a single un-batched observation."""
        return int(self.forward(obs.unsqueeze(0)).argmax(dim=1).item())
