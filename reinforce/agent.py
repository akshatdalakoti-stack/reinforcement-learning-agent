"""REINFORCE (Williams, 1992) -- Monte-Carlo policy gradient.

The derivation in three lines
-----------------------------
We want to maximise ``J(theta) = E_{tau ~ pi_theta}[R(tau)]``.  The trajectory
distribution depends on theta, so we cannot push the gradient inside the
expectation directly.  The likelihood-ratio ("log-derivative") trick fixes that::

    grad p = p * grad log p

    grad J = E_tau [ R(tau) * grad log p_theta(tau) ]
           = E_tau [ sum_t grad log pi_theta(a_t|s_t) * G_t ]

The environment's transition probabilities drop out of ``grad log p_theta(tau)``
because they do not depend on theta.  **This is the whole reason policy
gradients work without a model.**

Causality lets us replace ``R(tau)`` with the *reward-to-go* ``G_t``: an action
at time ``t`` cannot influence rewards collected before ``t``, so including them
only adds variance.

The loss we actually write
--------------------------
PyTorch minimises, and there is no "loss" here in the supervised sense -- only a
surrogate whose *gradient* equals the (negated) policy gradient::

    L(theta) = - (1/N) * sum_t log pi_theta(a_t|s_t) * A_t  -  beta * H[pi]

``A_t`` is treated as a constant (``detach``).  It is a *weight* on the
log-probability, not something to differentiate through.

The problem with this algorithm
-------------------------------
``G_t`` is a Monte-Carlo estimate: unbiased, but its variance grows with the
episode length and with the reward scale.  A CartPole return varies between 10
and 500 across episodes of the *same* policy, so the gradient direction is
dominated by which episodes happened to be sampled.  Two mitigations live here:

* **Baseline subtraction.**  For any function ``b(s)`` that does not depend on
  the action, ``E[b(s) * grad log pi(a|s)] = 0``, because
  ``sum_a pi(a|s) grad log pi(a|s) = grad sum_a pi(a|s) = grad 1 = 0``.
  Subtracting a baseline therefore changes the variance but *not* the
  expectation.  ``--baseline mean`` uses the batch mean return.
* **Return normalisation.**  Dividing by the batch standard deviation is a
  biased but very effective variance reduction, and doubles as adaptive
  learning-rate scaling.

Stage 6 replaces the batch-mean baseline with a *learned, state-dependent* one,
``V(s)``, which is exactly what an actor-critic is.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from reinforce.policy import PolicyNetwork


class ReinforceAgent:
    """Monte-Carlo policy gradient with an optional constant baseline.

    Parameters
    ----------
    baseline:
        ``"none"``   -- raw discounted returns (highest variance; the textbook form)
        ``"mean"``   -- subtract the batch mean return
        ``"norm"``   -- subtract the mean and divide by the std
    entropy_coef:
        Weight on the entropy bonus ``-beta * H[pi]``.  Policy gradient is prone
        to *premature convergence*: whatever action is sampled most gets
        reinforced, which makes it more likely, which gets it sampled more.  The
        entropy term pushes back against that collapse.
    batch_episodes:
        How many complete trajectories to average before each update.  REINFORCE
        needs whole episodes because ``G_t`` requires knowing all future rewards.
        Averaging several trajectories is the cheapest variance reduction there is
        (variance falls as 1/N).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        baseline: str = "norm",
        entropy_coef: float = 0.01,
        batch_episodes: int = 1,
        grad_clip: float = 0.5,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if baseline not in ("none", "mean", "norm"):
            raise ValueError("baseline must be one of 'none', 'mean', 'norm'")
        self.device = torch.device(device)
        self.obs_dim, self.n_actions = int(obs_dim), int(n_actions)
        self.gamma = float(gamma)
        self.baseline = baseline
        self.entropy_coef = float(entropy_coef)
        self.batch_episodes = int(batch_episodes)
        self.grad_clip = float(grad_clip)

        if seed is not None:
            torch.manual_seed(seed)
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(lr))

        # Buffers for the episodes collected since the last update.
        self._log_probs: list[torch.Tensor] = []
        self._entropies: list[torch.Tensor] = []
        self._returns: list[torch.Tensor] = []
        self._episodes_buffered = 0

        self.env_steps = 0
        self.updates = 0
        self.last_grad_norm = float("nan")

    # ------------------------------------------------------------------ acting
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Sample an action.  Kept gradient-free for the evaluation harness."""
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.policy.act(obs, greedy=greedy)
        return action

    def act_and_record(self, state: np.ndarray) -> int:
        """Sample an action *and keep the log-probability in the graph*.

        REINFORCE differentiates ``log pi(a_t|s_t)`` for the actions that were
        actually taken, so those tensors must stay attached to the computation
        graph until the update.  This is precisely why REINFORCE cannot reuse
        old data: the graph belongs to the parameters as they were at sampling
        time, and once they change the stored log-probs are stale.
        """
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        dist = self.policy.distribution(obs.unsqueeze(0))
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action).squeeze(0))
        self._entropies.append(dist.entropy().squeeze(0))
        self.env_steps += 1
        return int(action.item())

    # ---------------------------------------------------------------- learning
    def discounted_returns(self, rewards: Sequence[float]) -> torch.Tensor:
        """G_t = r_t + gamma*r_{t+1} + ... computed backwards in O(T).

            G_T     = r_T
            G_t     = r_t + gamma * G_{t+1}
        """
        returns = np.empty(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return torch.as_tensor(returns, device=self.device)

    def finish_episode(self, rewards: Sequence[float]) -> None:
        """Convert one finished trajectory's rewards into returns."""
        self._returns.append(self.discounted_returns(rewards))
        self._episodes_buffered += 1

    @property
    def ready_to_update(self) -> bool:
        return self._episodes_buffered >= self.batch_episodes

    def update(self) -> dict[str, float]:
        """Apply the policy gradient over every buffered episode."""
        if not self._returns:
            return {"loss": float("nan"), "entropy": float("nan")}

        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)
        returns = torch.cat(self._returns)

        advantages = self._apply_baseline(returns)

        # The minus sign turns gradient *ascent* on J into descent on L.
        policy_loss = -(log_probs * advantages).mean()
        entropy_bonus = entropies.mean()
        loss = policy_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates += 1
        self.last_grad_norm = float(grad_norm)
        stats = {
            "loss": float(policy_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "grad_norm": float(grad_norm),
            "return_std": float(returns.std().item()) if returns.numel() > 1 else 0.0,
        }
        self._clear()
        return stats

