"""A network that outputs a *policy* instead of values.

DQN's network answers "how good is each action here?" and the policy is derived
from it by ``argmax``.  That derivation is where value-based methods get awkward:

* the policy is a discontinuous function of the parameters -- an arbitrarily
  small weight change can flip the argmax and change behaviour completely;
* the policy is deterministic, so exploration has to be bolted on from outside
  (epsilon-greedy), and cannot itself be learned or annealed by the objective;
* extending to continuous actions requires solving ``argmax_a Q(s,a)`` at every
  step, which is its own optimisation problem.

A policy network answers the question directly: it maps a state to a
*distribution* over actions, ``pi_theta(a|s)``.  Now behaviour changes smoothly
with the parameters, exploration is part of the model (a high-entropy
distribution explores; a peaked one exploits), and the objective we care about
-- expected return -- can be differentiated with respect to theta.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """MLP producing logits over discrete actions.

    The network emits *logits*, not probabilities.  ``Categorical`` applies
    ``log_softmax`` internally, which is numerically stable in a way that
    ``log(softmax(x))`` is not: computing the probability first can underflow to
    zero and send ``log pi`` to ``-inf``, which poisons the gradient.
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
            layers.append(nn.Tanh())
            last = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, self.n_actions)

        for module in self.body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.constant_(module.bias, 0.0)
        # Small gain on the head keeps the initial policy close to uniform,
        # i.e. maximum entropy.  Starting from a confident policy is a real
        # failure mode: the gradient can only reinforce what is already sampled,
        # so an unlucky confident start is slow to escape.
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs))

    def distribution(self, obs: torch.Tensor) -> Categorical:
        """The action distribution pi(.|s)."""
        return Categorical(logits=self.forward(obs))

    def act(self, obs: torch.Tensor, greedy: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action, returning ``(action, log_prob, entropy)``.

        ``greedy=True`` takes the mode instead -- used only at evaluation time,
        where we want the policy's best guess rather than a sample from it.
        """
        dist = self.distribution(obs)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()
