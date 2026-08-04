"""Shared-trunk actor-critic network for PPO.

Stage 6 used two separate towers.  PPO conventionally shares a trunk and splits
into a policy head and a value head:

* the value function's dense, well-shaped learning signal helps carve out
  features the policy can also use;
* one forward pass produces both ``pi(.|s)`` and ``V(s)``, and PPO needs both
  at every one of its many optimisation steps, so the saving is real.

The catch is that the two losses now share parameters and have very different
magnitudes -- which is precisely why PPO carries an explicit ``value_coef``.
Set it too high and the value loss swamps the trunk and the policy stops
improving; too low and the value head underfits, the advantages get noisy, and
the policy loses its signal.

Set ``shared_trunk=False`` to fall back to independent towers, which makes the
architectural choice itself testable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _trunk(obs_dim: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last = obs_dim
    for size in hidden_sizes:
        layers += [_layer(last, size, gain=2 ** 0.5), nn.Tanh()]
        last = size
    return nn.Sequential(*layers), last


