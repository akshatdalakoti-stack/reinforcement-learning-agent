"""Advantage Actor-Critic (A2C), n-step.

What REINFORCE leaves on the table
----------------------------------
REINFORCE weights ``grad log pi(a_t|s_t)`` by the Monte-Carlo return ``G_t``.
That estimator is unbiased but its variance is enormous, for two reasons:

1. ``G_t`` sums every reward and every random transition from ``t`` to the end
   of the episode, so its variance compounds over the horizon.
2. Nothing tells the agent what return was *expected*.  On CartPole, where all
   rewards are +1, a return of 200 is excellent early in training and mediocre
   later -- but REINFORCE reinforces every action in a 200-step episode equally,
   including the one that eventually knocked the pole over.

The fix, in one substitution
----------------------------
Replace the return with the **advantage**::

    A(s,a) = Q(s,a) - V(s)

"How much better was this action than what this state was worth anyway?"  Since
``V(s)`` does not depend on the action, subtracting it leaves the gradient
unbiased (the baseline lemma from :mod:`reinforce.agent`), but recentres the
weights around zero: genuinely better-than-expected actions get positive weight,
worse-than-expected ones get negative weight, and a merely average action in a
good state gets ~0 instead of a large positive push.

We never learn ``Q`` directly.  We estimate the advantage from sampled rewards
and the critic's ``V``.  With ``n = 1`` that is the plain TD error::

    A_t = r_t + gamma * V(s_{t+1}) - V(s_t)        # delta_t

and with ``n > 1``::

    A_t = sum_{k=0}^{n-1} gamma^k r_{t+k}  +  gamma^n V(s_{t+n})  -  V(s_t)

``n`` is the bias/variance dial.  ``n = 1`` bootstraps immediately: minimum
variance, maximum bias (it inherits every error in ``V``).  ``n = inf`` is
REINFORCE: unbiased, maximum variance.  Everything useful is in between.

What this buys, structurally
----------------------------
Because the advantage only needs ``n`` steps rather than a full episode, the
agent can update **mid-episode**.  REINFORCE cannot: ``G_t`` is undefined until
the episode ends.  On a task with 1000-step episodes that is the difference
between one update per 1000 steps and one per 5.

The new failure mode
--------------------
The actor now trusts a critic that is itself being learned.  Early in training
``V`` is nonsense, so the advantages are nonsense, so the policy moves in a bad
direction, which changes the state distribution the critic is trying to fit.
Two learners chasing each other can oscillate or lock into a bad equilibrium --
which is exactly the instability PPO is designed to bound.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actor_critic.networks import Actor, Critic


class ActorCriticAgent:
    """n-step advantage actor-critic with separate actor and critic networks.

    Parameters
    ----------
    n_steps:
        Rollout length between updates.  Also the bias/variance dial described
        above.
    value_coef, entropy_coef:
        Loss weights.  ``value_coef`` matters less here than in PPO because the
        two networks have separate optimisers and cannot compete for parameters.
    normalize_advantages:
        Standardise advantages within the rollout.  Cheap, slightly biased,
        and it keeps the effective policy step size stable as returns grow.
    """

