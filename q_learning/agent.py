"""Tabular Q-learning.

The whole algorithm is the single update in :meth:`QLearningAgent.update`:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
                              \\_________________________________/
                                        TD error, delta

Three ideas are worth naming explicitly, because everything later in this
repository is a reaction to one of them:

1. **Bootstrapping.**  The target ``r + gamma * max_a' Q(s',a')`` is built from
   the agent's *own current estimate* of the next state, not from a Monte-Carlo
   return.  That is what lets learning happen every step instead of every
   episode -- and it is also what makes deep versions unstable, because the
   target moves whenever the estimator moves.
2. **Off-policy learning.**  The ``max`` is over the *greedy* action, while the
   action actually taken came from an epsilon-greedy behaviour policy.  So the
   agent explores with one policy and learns the value of another.  This is
   what makes replay buffers legal in DQN, and what makes them illegal in
   REINFORCE.
3. **No model.**  ``P`` and ``R`` are never read.  The agent only ever sees
   sampled transitions, so it learns from experience rather than from planning.
"""

from __future__ import annotations

import json
import os

import numpy as np


class QLearningAgent:
    """A Q-table plus epsilon-greedy exploration.

    Parameters
    ----------
    n_states, n_actions:
        Table dimensions.  The table *is* the agent -- one independent
        parameter per state-action pair, which is why this approach cannot
        survive a continuous state space (see :mod:`dqn`).
    alpha:
        Learning rate.  In a stochastic environment the TD target is a noisy
        sample of the true expectation, so ``alpha < 1`` makes the update an
        exponentially-weighted running average rather than an overwrite.
    gamma:
        Discount factor.  Sets the effective planning horizon at ~1/(1-gamma)
        steps and keeps returns finite.
    epsilon_start / epsilon_end / epsilon_decay:
        Multiplicative decay schedule for exploration, applied once per
        episode.  Early on the agent needs coverage; later it needs to exploit,
        or the recorded episode reward never reflects what it actually learned.
    optimistic_init:
        Value used to fill the table.  Initialising above the achievable return
        is a second exploration mechanism: unvisited actions look attractive
        until they are tried and corrected downward.
    alpha_schedule:
        ``"constant"`` keeps ``alpha`` fixed; ``"visit_count"`` uses
        ``alpha_t(s,a) = 1 / n(s,a) ** alpha_power``.

        This matters more than it looks.  Tabular Q-learning is only guaranteed
        to converge to ``Q*`` when the step sizes satisfy the Robbins-Monro
        conditions ``sum_t alpha_t = inf`` and ``sum_t alpha_t^2 < inf``.  A
        constant ``alpha`` violates the second one, so in a *stochastic*
        environment the estimates never settle -- they keep bouncing around
        ``Q*`` with a variance floor proportional to ``alpha``.  The decaying
        schedule satisfies both and does converge; the price is that it adapts
        slowly if the environment ever changes.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        alpha_schedule: str = "constant",
        alpha_power: float = 0.75,
        seed: int | None = None,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        if alpha_schedule not in ("constant", "visit_count"):
            raise ValueError("alpha_schedule must be 'constant' or 'visit_count'")
        self.alpha_schedule = alpha_schedule
        self.alpha_power = float(alpha_power)

        self.q_table = np.full((self.n_states, self.n_actions), float(optimistic_init))
        self.rng = np.random.default_rng(seed)

        # Diagnostics: how much the table moved, and how often each cell was hit.
        self.visit_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        self.last_episode_max_delta = 0.0

    # ------------------------------------------------------------------ acting
    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection -- the exploration/exploitation knob.

        Ties among equally-valued actions are broken uniformly at random rather
        than by ``argmax``'s first-index rule.  With a zero-initialised table
        every action ties at the start, and a deterministic ``argmax`` would
        make the agent march in one direction forever.
        """
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state: int) -> int:
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    # ---------------------------------------------------------------- learning
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> float:
        """Apply one Bellman backup and return the TD error.

        ``terminated`` (the MDP genuinely ended) must be distinguished from
        truncation (a time limit fired).  On termination there is no future, so
        the target is just ``r``.  On truncation the future still exists, we
        simply stopped watching, so we keep bootstrapping -- treating the two
        the same teaches the agent that time limits destroy value.
        """
        # Count the visit first, so the decaying schedule uses n >= 1.
        self.visit_counts[state, action] += 1
        alpha = self._step_size(state, action)

        best_next = 0.0 if terminated else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]

        self.q_table[state, action] += alpha * td_error

        self.last_episode_max_delta = max(self.last_episode_max_delta, abs(alpha * td_error))
        return float(td_error)

    def _step_size(self, state: int, action: int) -> float:
        if self.alpha_schedule == "constant":
            return self.alpha
        n = max(1, int(self.visit_counts[state, action]))
        return 1.0 / (n ** self.alpha_power)

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def start_episode(self) -> None:
        self.last_episode_max_delta = 0.0

    # ------------------------------------------------------------ introspection
    def state_values(self) -> np.ndarray:
        """V(s) = max_a Q(s,a) -- the greedy value implied by the table."""
        return self.q_table.max(axis=1)

    def greedy_policy(self) -> np.ndarray:
        return self.q_table.argmax(axis=1)

    def coverage(self, valid_states=None) -> float:
        """Fraction of (state, action) cells ever updated.

        ``valid_states`` restricts the denominator to states the agent can
        actually stand in.  Without it, walls and terminal cells -- which are
        never decision points -- would permanently drag the number down and
        make full coverage look like a bug.
        """
        counts = self.visit_counts if valid_states is None else self.visit_counts[valid_states]
        return float((counts > 0).mean()) if counts.size else 0.0

    # -------------------------------------------------------------- checkpoint
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            q_table=self.q_table,
            visit_counts=self.visit_counts,
            meta=json.dumps(
                {
                    "n_states": self.n_states,
                    "n_actions": self.n_actions,
                    "alpha": self.alpha,
                    "alpha_schedule": self.alpha_schedule,
                    "alpha_power": self.alpha_power,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                }
            ),
        )
        return path if path.endswith(".npz") else path + ".npz"

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "QLearningAgent":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        agent = cls(
            n_states=meta["n_states"],
            n_actions=meta["n_actions"],
            alpha=meta["alpha"],
            alpha_schedule=meta.get("alpha_schedule", "constant"),
            alpha_power=meta.get("alpha_power", 0.75),
            gamma=meta["gamma"],
            epsilon_start=meta["epsilon"],
            seed=seed,
        )
        agent.q_table = data["q_table"]
        agent.visit_counts = data["visit_counts"]
        return agent
