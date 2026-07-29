"""Two networks: one that acts, one that judges.

The brief asks for *separate* Actor and Critic networks, and that is what this
module provides.  It is worth being explicit about the trade-off, because PPO
in stage 7 makes the opposite choice:

**Separate towers** (here).  The actor's gradient cannot corrupt the critic's
features and vice versa.  The two objectives have very different scales -- a
policy-gradient loss is O(1) while a value loss on CartPole returns is O(100) --
and with separate networks they cannot fight over a shared representation.  The
cost is roughly double the parameters and no feature sharing.

**Shared trunk** (stage 7).  Cheaper, and the value function's rich learning
signal helps shape features the policy can use.  The risk is that the value
loss, being numerically larger, dominates the shared layers -- which is why PPO
needs an explicit ``value_coef`` to rebalance them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(obs_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = obs_dim
    for size in hidden_sizes:
        linear = nn.Linear(last, size)
        nn.init.orthogonal_(linear.weight, gain=2 ** 0.5)
        nn.init.constant_(linear.bias, 0.0)
        layers += [linear, nn.Tanh()]
        last = size
    head = nn.Linear(last, out_dim)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.constant_(head.bias, 0.0)
    layers.append(head)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """pi(a|s) as logits over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        # gain 0.01 -> near-uniform initial policy -> maximum initial entropy.
        self.net = _mlp(obs_dim, hidden_sizes, n_actions, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    """V(s): the expected return from ``s`` under the *current* policy.

    Note what this is not.  DQN's network learns ``Q(s,a)``, the value of each
    action, because it derives its policy from those values by ``argmax``.  The
    critic here learns only ``V(s)``, a single number per state, because it is
    not choosing anything -- the actor does that.  Estimating one number instead
    of ``n_actions`` numbers is a strictly easier regression problem, and it is
    all the actor needs to know whether an action beat expectations.
    """

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        # gain 1.0 on a value head: unlike the policy head there is no reason to
        # start near zero output beyond keeping the initial TD errors modest.
        self.net = _mlp(obs_dim, hidden_sizes, 1, out_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``(batch, obs_dim)`` -> ``(batch,)``."""
        return self.net(obs).squeeze(-1)
