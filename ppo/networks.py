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


class ActorCriticNetwork(nn.Module):
    """Policy logits and a state value from one module.

    Orthogonal initialisation with ``gain=sqrt(2)`` on Tanh layers, ``0.01`` on
    the policy head and ``1.0`` on the value head is the standard PPO recipe.
    The tiny policy-head gain is the one that matters most: it makes the initial
    policy near-uniform, so the agent starts with maximum entropy and the
    clipping ratio starts at exactly 1.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        shared_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        self.shared_trunk = bool(shared_trunk)
        self.hidden_sizes = list(hidden_sizes)

        if self.shared_trunk:
            self.trunk, feat = _trunk(obs_dim, hidden_sizes)
            self.value_trunk = None
        else:
            self.trunk, feat = _trunk(obs_dim, hidden_sizes)
            self.value_trunk, _ = _trunk(obs_dim, hidden_sizes)

        self.policy_head = _layer(feat, n_actions, gain=0.01)
        self.value_head = _layer(feat, 1, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``obs -> (logits, values)``."""
        policy_features = self.trunk(obs)
        value_features = (
            policy_features if self.shared_trunk else self.value_trunk(obs)
        )
        return self.policy_head(policy_features), self.value_head(value_features).squeeze(-1)

