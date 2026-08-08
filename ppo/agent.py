"""Proximal Policy Optimization (Schulman et al., 2017).

The problem PPO solves
----------------------
Every policy-gradient method so far has thrown its data away after a single
gradient step.  That is not squeamishness -- it is forced.  The estimator
``E_{a ~ pi_theta}[grad log pi_theta(a|s) A(s,a)]`` is an expectation over the
*current* policy, so the moment theta moves, the collected data is off-policy
and the estimator is biased.  With a rollout of 2048 steps and one gradient
step per rollout, sample efficiency is terrible.

Importance sampling makes reuse *possible*::

    E_{a ~ pi_old} [ (pi_theta(a|s) / pi_old(a|s)) * A(s,a) ]

and defining ``r(theta) = pi_theta(a|s) / pi_old(a|s)`` gives the surrogate
objective ``L = E[r(theta) * A]``, which is valid for data from ``pi_old``.

But importance sampling is only trustworthy while the two policies are close.
As ``r`` departs from 1 the variance explodes, and worse, the surrogate stops
predicting the true objective: maximising ``r * A`` with no constraint drives
``r`` toward infinity wherever ``A > 0``.  One large step off a noisy advantage
estimate produces a policy that is both bad *and* badly estimated -- and unlike
supervised learning there is no fixed dataset to recover from, because the
broken policy now collects the next batch of data. This is *performance
collapse*, and it is the characteristic failure of naive policy gradient.

TRPO solved it with a hard KL constraint and second-order optimisation.  PPO
gets most of the benefit with a first-order trick:

    L_CLIP(theta) = E[ min( r*A , clip(r, 1-eps, 1+eps)*A ) ]

Read the ``min`` carefully -- it is not symmetric, and the asymmetry is the
whole idea:

* ``A > 0`` (the action was better than expected).  Increasing ``r`` raises the
  objective, but ``clip`` caps the second term at ``(1+eps)*A``.  Past that
  point the ``min`` selects the clipped branch, which is *constant* in theta,
  so its gradient is zero.  The incentive to keep pushing disappears.
* ``A < 0`` (the action was worse than expected).  Decreasing ``r`` raises the
  objective; the clip floors the second term at ``(1-eps)*A``.  Since ``A`` is
  negative, ``min`` again selects the clipped branch once ``r < 1-eps``, and
  the gradient vanishes.

So the objective is a *pessimistic lower bound*: it takes the less optimistic
of the clipped and unclipped estimates.  Improvements inside the trust region
are rewarded; steps outside it are simply not rewarded further.  There is no
barrier and no constraint -- the gradient just goes flat, which is why plain
Adam can optimise it.

Why the ``min`` and not just ``clip``
-------------------------------------
Using ``clip(r,...) * A`` alone would zero the gradient whenever ``r`` left the
region *in either direction* -- including when the policy has drifted too far
and needs to come **back**.  The ``min`` keeps the unclipped term active in
exactly that case, so a policy that overshot can still be pulled toward the
data.  Dropping the ``min`` is one of the classic ways to get a PPO
implementation that silently fails to recover from a bad update.

Remaining failure modes
-----------------------
* **Entropy collapse.**  Clipping bounds each step, not the cumulative drift.
  Over many updates the policy can still become deterministic, stop exploring,
  and freeze at a local optimum.  The entropy bonus opposes this; watching
  ``entropy`` fall toward 0 is the early warning.
* **Too many epochs.**  Each epoch pushes the policy further from ``pi_old``.
  Once most samples are clipped, the extra epochs are computing zero-gradient
  batches -- wasted work at best, and at worst overfitting the value head to a
  small rollout.  ``clip_fraction`` and ``approx_kl`` are the metrics to watch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from ppo.networks import ActorCriticNetwork
from ppo.rollout import RolloutBuffer


class PPOAgent:
    """Clipped-surrogate PPO with GAE.

    Parameters
    ----------
    clip_epsilon:
        Half-width of the trust region on the probability ratio.  0.1-0.3 in
        practice; smaller is more conservative and slower.
    n_epochs, minibatch_size:
        How hard to squeeze each rollout.  This is the sample-efficiency knob
        that separates PPO from A2C -- and the main way to destabilise it.
    value_coef:
        Weight on the value loss.  Matters because the trunk is shared.
    target_kl:
        Optional early-stopping threshold.  If the approximate KL between
        ``pi_old`` and ``pi_theta`` exceeds this mid-update, the remaining
        epochs are skipped.  A safety net for when clipping alone is not
        holding the policy in place.
    clip_value_loss:
        Apply the same clipping idea to the value head, bounding how far
        ``V(s)`` may move from ``V_old(s)`` in one update.
    """

