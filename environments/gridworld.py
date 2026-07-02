"""A GridWorld MDP written from scratch -- no Gymnasium, no wrappers.

The point of implementing this by hand is that every element of the formal MDP
definition ``(S, A, P, R, gamma)`` appears explicitly in the code:

* ``S`` -- ``rows * cols`` integer states, ``index = r * cols + c``
* ``A`` -- four movement intents (left, right, up, down)
* ``P`` -- ``_transition_distribution``: with probability ``1 - slip_prob`` the
  agent moves as intended, otherwise it slips to one of the two perpendicular
  directions.  Walls and grid edges bounce the agent back to where it was.
* ``R`` -- ``step_reward`` on every move, plus a terminal payoff at the goal
  (``+1``) or the pit (``-1``)
* ``gamma`` -- lives with the agent, not with the environment

The class exposes the Gymnasium five-tuple API (``reset`` / ``step`` returning
``terminated`` and ``truncated`` separately) so the same training-loop shape
works here and on CartPole, but nothing is inherited from Gymnasium.
"""

from __future__ import annotations

import numpy as np

# (dr, dc) in row/column space.  The row index grows downwards, matching how the
# grid is printed and how matplotlib's imshow lays out the policy plot.
ACTION_DELTAS: tuple[tuple[int, int], ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))
ACTION_NAMES: tuple[str, ...] = ("LEFT", "RIGHT", "UP", "DOWN")

DEFAULT_LAYOUT = [
    ".....",
    ".##..",
    "...#.",
    ".#..P",
    "S.#.G",
]


class GridWorld:
    """Tabular navigation MDP.

    Layout characters: ``S`` start, ``G`` goal (+1), ``P`` pit (-1), ``#`` wall,
    ``.`` free cell.

    Parameters
    ----------
    slip_prob:
        Probability that the intended action is replaced by one of the two
        perpendicular actions (each with half that probability).  Above 0 the
        environment is stochastic, which is what forces Q-learning to *average*
        returns with a learning rate instead of copying the last outcome seen.
    step_reward:
        Small negative reward per step, so shorter paths score higher.
    max_steps:
        Truncation limit.  Truncation is reported separately from termination
        because bootstrapping must continue through a time-limit cutoff.
    """

    metadata = {"render_modes": ["ansi", "human"]}

