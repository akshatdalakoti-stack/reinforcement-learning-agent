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

    def walkable_mask(self) -> np.ndarray:
        """Boolean ``(rows, cols)`` array, False on walls (used for plotting)."""
        mask = np.ones((self.rows, self.cols), dtype=bool)
        for r, c in self.walls:
            mask[r, c] = False
        return mask

    def _move(self, cell: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols):
            return cell          # walked into the outer edge -> stay put
        if nxt in self.walls:
            return cell          # walked into a wall -> stay put
        return nxt

    # ------------------------------------------------------------------ model
    def _transition_distribution(
        self, cell: tuple[int, int], action: int
    ) -> list[tuple[float, tuple[int, int]]]:
        """The distribution over next cells as ``(probability, next_cell)`` pairs."""
        if self.slip_prob <= 0.0:
            return [(1.0, self._move(cell, action))]
        # LEFT/RIGHT (0, 1) slip to UP/DOWN (2, 3) and vice versa.
        perpendicular = (2, 3) if action in (0, 1) else (0, 1)
        outcomes = [(1.0 - self.slip_prob, self._move(cell, action))]
        for alt in perpendicular:
            outcomes.append((self.slip_prob / 2.0, self._move(cell, alt)))
        return outcomes

    def transition_model(self) -> tuple[np.ndarray, np.ndarray]:
        """Full ``(P, R)`` tensors for dynamic programming.

        ``P[s, a, s2]`` and ``R[s, a]``.  A planner may use these; the learning
        agents in this repository never do -- that is exactly the model-free /
        model-based line the project is drawn along.
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            cell = self.to_cell(s)
            if cell in self.walls or cell in self.terminals:
                P[s, :, s] = 1.0     # absorbing; no reward accrues after the end
                continue
            for a in range(self.n_actions):
                for prob, nxt in self._transition_distribution(cell, a):
                    P[s, a, self.to_index(nxt)] += prob
                    R[s, a] += prob * (self.step_reward + self.terminals.get(nxt, 0.0))
        return P, R

    # -------------------------------------------------------------- gym-style
    def decision_states(self) -> list[int]:
        """State indices where the agent actually chooses an action.

        Walls cannot be occupied and terminals end the episode, so these are the
        only states a policy is ever defined on.
        """
        return [
            s
            for s in range(self.n_states)
            if self.to_cell(s) not in self.walls and self.to_cell(s) not in self.terminals
        ]

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset to the start cell, or to a uniformly random one.

        ``options={"exploring_start": True}`` implements *exploring starts*: the
        episode begins in a random legal cell instead of ``S``.  With a single
        fixed start, states far from the optimal path are visited so rarely that
        their Q-values never converge -- exploring starts is the classic fix,
        and the contrast between the two is the point of the
        ``experiments/exp1_exploring_starts.py`` ablation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and options.get("exploring_start"):
            candidates = self.decision_states()
            self._agent = self.to_cell(int(self._rng.choice(candidates)))
        else:
            self._agent = self.start
        self._steps = 0
        return self.to_index(self._agent), {"cell": self._agent}

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside 0..{self.n_actions - 1}")

        intended = self._move(self._agent, action)
        outcomes = self._transition_distribution(self._agent, action)
        probs = np.array([p for p, _ in outcomes])
        chosen = outcomes[int(self._rng.choice(len(outcomes), p=probs / probs.sum()))][1]

        self._agent = chosen
        self._steps += 1

        reward = self.step_reward + self.terminals.get(chosen, 0.0)
        terminated = chosen in self.terminals
        truncated = (not terminated) and self._steps >= self.max_steps
        info = {"cell": chosen, "slipped": chosen != intended}
        return self.to_index(chosen), reward, terminated, truncated, info

    def render(self) -> str:
        rows = []
        for r in range(self.rows):
            line = []
            for c in range(self.cols):
                line.append("A" if (r, c) == self._agent else self.layout[r][c])
            rows.append(" ".join(line))
        text = "\n".join(rows)
        if self.render_mode == "human":
            print(text)
        return text

