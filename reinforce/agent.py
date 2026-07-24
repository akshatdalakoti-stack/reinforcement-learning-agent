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

