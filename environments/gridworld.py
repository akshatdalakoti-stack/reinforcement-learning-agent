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

    def __init__(
        self,
        layout: list[str] | None = None,
        slip_prob: float = 0.1,
        step_reward: float = -0.04,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
        max_steps: int = 100,
        render_mode: str | None = None,
    ) -> None:
        self.layout = list(layout or DEFAULT_LAYOUT)
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        if any(len(row) != self.cols for row in self.layout):
            raise ValueError("all layout rows must have the same width")

        self.slip_prob = float(slip_prob)
        self.step_reward = float(step_reward)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.walls: set[tuple[int, int]] = set()
        self.terminals: dict[tuple[int, int], float] = {}
        self.start: tuple[int, int] | None = None
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                cell = (r, c)
                if ch == "#":
                    self.walls.add(cell)
                elif ch == "G":
                    self.terminals[cell] = float(goal_reward)
                elif ch == "P":
                    self.terminals[cell] = float(pit_reward)
                elif ch == "S":
                    self.start = cell
        if self.start is None:
            raise ValueError("layout must contain a start cell 'S'")

        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTION_DELTAS)
        self.action_names = ACTION_NAMES
        self.action_deltas = ACTION_DELTAS

        self._agent: tuple[int, int] = self.start
        self._steps = 0
        self._rng = np.random.default_rng()

    # --------------------------------------------------------------- geometry
    def to_index(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.cols + cell[1]

    def to_cell(self, index: int) -> tuple[int, int]:
        return divmod(int(index), self.cols)

    def is_terminal_state(self, index: int) -> bool:
        return self.to_cell(index) in self.terminals

