# rl-agent — building up to PPO, one idea at a time

An implementation-from-scratch study of the modern deep-RL progression:

```
Q-Learning → DQN → +Replay → +Target Network → REINFORCE → Actor-Critic → PPO
```

Gymnasium supplies the simulators. PyTorch supplies autograd. **Everything
between them — every update rule, buffer, advantage estimator and loss — is
written here.** No Stable-Baselines3, no RLlib, no CleanRL.

The goal is not to collect seven working agents. It is to be able to answer, for
each step: *what broke, why the previous method could not fix it, what
mathematical idea was introduced, and what experiment demonstrates that the idea
worked.* Each stage below is written to that structure, and each claim is backed
by a controlled experiment in `experiments/`.

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [Quick start](#quick-start)
- [Part I — The mathematical foundation](#part-i--the-mathematical-foundation)
  - [Markov Decision Processes](#markov-decision-processes)
  - [Returns and the discount factor](#returns-and-the-discount-factor)
  - [Policies, value functions, Q-functions](#policies-value-functions-q-functions)
  - [The Bellman equations](#the-bellman-equations)
  - [Temporal-difference learning](#temporal-difference-learning)
  - [Exploration vs exploitation](#exploration-vs-exploitation)
- [Part II — The progression](#part-ii--the-progression)
  - [Stage 1 — Tabular Q-learning](#stage-1--tabular-q-learning)
  - [Stage 2 — Deep Q-Network](#stage-2--deep-q-network)
  - [Stage 3 — Experience replay](#stage-3--experience-replay)
  - [Stage 4 — Target networks](#stage-4--target-networks)
  - [Stage 5 — REINFORCE](#stage-5--reinforce)
  - [Stage 6 — Actor-Critic](#stage-6--actor-critic)
  - [Stage 7 — PPO](#stage-7--ppo)
- [Repository layout](#repository-layout)
- [Metrics, logs and reproducibility](#metrics-logs-and-reproducibility)
- [Implementation details that matter more than they look](#implementation-details-that-matter-more-than-they-look)
- [References](#references)

---

## Results at a glance

All numbers below are from this repository, on a 12-core CPU (no GPU), seed 0.
"Final greedy evaluation" runs the deterministic policy with exploration off, on
seeds the agent never trained on — see [Metrics](#metrics-logs-and-reproducibility)
for why that is not the same as training reward.

| # | stage | environment | env steps | episodes | train time | final greedy evaluation |
|---|---|---|---:|---:|---:|---|
| 1 | Q-learning (tabular) | GridWorld | 94,674 | 6,000 | 6 s | **18/18 optimal actions**, $\Vert Q - Q^*\Vert_{\max} = 0.12$ |
| 2 | DQN, no replay, no target | CartPole-v1 | 25,378 | 2,000 | 21 s | 9.7 ± 0.9 — **diverged** |
| 3 | DQN + replay | CartPole-v1 | 25,502 | 2,000 | 28 s | 9.2 ± 1.0 — **diverged** |
| 4 | DQN + replay + target net | CartPole-v1 | 200,036 | 1,909 | 246 s | **500.0 ± 0.0** — solved |
| 5 | REINFORCE | CartPole-v1 | 200,130 | 508 | 336 s | 459.4 ± 56.1 |
| 6 | Actor-Critic (n-step) | CartPole-v1 | 200,203 | 1,029 | 193 s | **500.0 ± 0.0** — solved |
| 7 | PPO | CartPole-v1 | 200,578 | 754 | 336 s | **500.0 ± 0.0** — solved |
| 7 | **PPO** | **LunarLander-v3** | 599,800 | 1,959 | 820 s | **250.3 ± 46.8** — solved (threshold 200) |

Three findings worth pulling out of that table, each developed in the stage it
belongs to:

**The target network is load-bearing; on this task, replay alone is not.**
Rows 2 and 3 do not merely learn slowly — they *diverge*. The mean predicted
Q-value is the tell:

| variant | final mean predicted Q | largest possible value |
|---|---|---:|
| no replay, no target net | **3,142,030** | 99.3 |
| replay, no target net | **7,453,550** | 99.3 |
| replay + target net | **91.3** | 99.3 |

With γ = 0.99 and a 500-step cap, no CartPole state is worth more than
∑₀⁴⁹⁹ 0.99ᵗ ≈ 99.3. The two variants without a target network predict values
four to five orders of magnitude larger. That is the deadly triad diverging, not
a tuning problem — and replay makes it *worse*, because more useful gradient
steps on a self-referential target close the feedback loop faster. The full 2×2
factorial that separates the two effects is [Experiment 2](#stage-3--experience-replay).

**REINFORCE's gradient estimator is essentially noise.** Forty independent
single-episode gradient estimates *at identical parameters* have a mean pairwise
cosine similarity of **+0.003** — orthogonal. It still learns, but it is the only
algorithm here that fails to reach a stable 500, and the ± 56.1 in its row is
that variance surfacing as behaviour.

**CartPole cannot rank stages 4–7, and it is worth saying so.** Given the same
200,000-step budget, all four working methods reach a greedy 500. Sampling the
greedy policy every 5,000 steps:

| stage | first greedy ≥475 | evals ≥475 | holds ≥475 to the end from | last 5 evaluations |
|---|---:|---:|---:|---|
| DQN + replay + target | 95,125 | 35% | 180,475 | 500 500 500 500 500 |
| REINFORCE | 10,441 | 80% | **never** | 500 427 344 346 **426** |
| Actor-Critic | **5,033** | 72% | 75,324 | 500 500 500 500 500 |
| PPO | 6,323 | 69% | 176,477 | 500 486 500 500 500 |

There is no clean monotone ordering here, and claiming one would be
over-reading a single seed on a task where every method saturates. What *is*
robust is the last column: **REINFORCE is the only method that reaches 500 and
then loses it**, drifting back to ~400 over the final 10% of training. Its
gradient noise never switches off, so there is no such thing as "converged".
Everything else ends pinned at the ceiling.

The progression's real payoff is not visible on CartPole at all — it is that PPO
is the only method here that also solves **LunarLander**, an 8-dimensional
problem with shaped rewards and a genuine failure mode (crashing), inside
600,000 steps.

![Progression on CartPole](results/plots/progression_cartpole.png)

---

## Quick start

```bash
pip install -r requirements.txt
```

LunarLander (stage 7) needs Box2D, which builds from source on Windows:

```bash
pip install swig && pip install box2d-py --no-build-isolation
```

If it will not build, `ppo/train.py` falls back to CartPole automatically and
says so — nothing else in the project depends on it.

Run the whole progression and regenerate every figure and table (~30 min on CPU):

```bash
python -m experiments.run_all
```

To check everything is wired up without waiting, use the short-budget version
(~3 min). It writes to `_quick`-suffixed run names, so a smoke test can never
overwrite real results:

```bash
python -m experiments.run_all --quick --skip-lunar
```

Add `--with-ablations` to also run all five controlled experiments.

Or one stage at a time:

```bash
python -m q_learning.train --exploring-starts --alpha-schedule visit_count
python -m dqn.train --no-replay --no-target-network --run-name dqn_online
python -m dqn.train --no-target-network --run-name dqn_replay
python -m dqn.train --run-name dqn_cartpole
python -m reinforce.train --gradient-variance 40
python -m actor_critic.train
python -m ppo.train --env LunarLander-v3 --total-steps 600000
```

Evaluate any saved checkpoint with exploration switched off:

```bash
python -m evaluation.evaluate --checkpoint checkpoints/ppo_lunarlander.pt --env LunarLander-v3
```

Everything runs on CPU. `--device cuda` is used automatically when available;
these networks are small enough that CPU is often faster.

---

## Part I — The mathematical foundation

### Markov Decision Processes

Every algorithm here assumes the environment is a **Markov Decision Process**,
the tuple

$$(\mathcal{S},\ \mathcal{A},\ P,\ R,\ \gamma)$$

| symbol | name | meaning |
|---|---|---|
| $s \in \mathcal{S}$ | **state** | everything the agent observes about the world |
| $a \in \mathcal{A}$ | **action** | what the agent may do |
| $P(s' \mid s,a)$ | **transition** | probability of landing in $s'$ after $a$ in $s$ |
| $R(s,a)$ | **reward** | the scalar the agent is trying to maximise |
| $\gamma \in [0,1)$ | **discount** | how much future reward is worth now |

The load-bearing assumption is the **Markov property**:

$$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0)$$

The current state is a sufficient statistic for the future. This is what makes
value functions well-defined — if the past mattered beyond $s_t$, then "the
value of state $s$" would not be a function of $s$ at all.

`environments/gridworld.py` implements every element of this tuple explicitly:
`_transition_distribution` **is** $P$, and `transition_model()` returns the full
$P$ and $R$ tensors. That is worth having, because it lets us solve the MDP
exactly by value iteration and check what the learning agents converge to.

Note what changes at stage 2: on CartPole, $\mathcal{S} = \mathbb{R}^4$ is
continuous, so $P$ and $R$ can no longer be written down as tables — and the
agent never sees them anyway. Every method here is **model-free**: it learns
from sampled transitions $(s, a, r, s')$, never from $P$.

### Returns and the discount factor

The agent maximises the **discounted return**:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

$\gamma$ does two jobs. Mathematically it keeps $G_t$ finite on
non-terminating tasks. Practically it sets the **effective horizon** at roughly
$1/(1-\gamma)$ steps — $\gamma = 0.99$ means the agent plans about 100 steps
ahead. It is a modelling choice, not a constant of nature, which is why it
belongs to the agent in this codebase, not to the environment.

### Policies, value functions, Q-functions

A **policy** $\pi(a \mid s)$ maps states to a distribution over actions. This is
the object we are ultimately searching for. Value-based methods (stages 1–4)
represent it implicitly as $\arg\max_a Q(s,a)$; policy-gradient methods (stages
5–7) represent it directly as a network. That distinction organises the whole
second half of this document.

The **state-value function** is the expected return from $s$ under $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s\right]$$

The **action-value function** (Q-function) additionally fixes the first action:

$$Q^\pi(s,a) = \mathbb{E}_\pi\!\left[G_t \mid s_t = s,\ a_t = a\right]$$

Why prefer $Q$ over $V$? Because $Q$ is directly actionable. Given $Q$, the
greedy policy is $\arg\max_a Q(s,a)$ — no model needed. Given only $V$, choosing
an action requires knowing $P$ to look one step ahead. **That single fact is why
value-based model-free control learns $Q$ and not $V$** — and, symmetrically,
why the critic in stage 6 can get away with learning $V$: it is not choosing
anything.

The two connect through the **advantage**:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

"How much better was this action than this state's average?" This is the
quantity stages 6 and 7 are built around.

### The Bellman equations

Values are recursive. The return from $s$ is one reward plus the discounted
return from wherever you land:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]$$

For the *optimal* policy, the expectation over actions becomes a maximum — the
**Bellman optimality equation**:

$$Q^*(s,a) = \mathbb{E}_{s'}\!\left[R(s,a) + \gamma \max_{a'} Q^*(s',a')\right]$$

This is the equation the entire repository is trying to solve. Two facts about
it drive everything that follows:

1. **It is a fixed point, not a minimum.** $Q^*$ is the unique solution of
   $Q = \mathcal{T}Q$, where $\mathcal{T}$ is the Bellman optimality operator.
   $\mathcal{T}$ is a $\gamma$-contraction in the sup-norm, so iterating it
   converges — *in the tabular case*. Once $Q$ is a neural network, the
   contraction argument no longer applies, and stages 2–4 are about the
   consequences.
2. **It is defined by an expectation we cannot compute.** $\mathbb{E}_{s'}$
   requires $P$. Model-free methods replace it with a sample — which is
   temporal-difference learning.

### Temporal-difference learning

Given one sampled transition $(s,a,r,s')$, the sampled version of the right-hand
side is the **TD target**, and its disagreement with the current estimate is the
**TD error**:

$$\delta_t = \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target}} - Q(s,a)$$

and we move the estimate a little way toward the target:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta_t$$

TD learning sits between two extremes:

| | Monte-Carlo | TD(0) |
|---|---|---|
| target | actual return $G_t$ | $r + \gamma Q(s',a')$ |
| needs | the full episode | one step |
| bias | none | inherits errors in $Q$ |
| variance | high (sums all future randomness) | low |

The word for using your own estimate inside your own target is
**bootstrapping**, and it is the single most consequential idea in this
repository. It buys per-step learning and low variance. It costs bias, and — as
stages 2–4 show — it is the ingredient that makes deep value learning unstable.
This same tension reappears at stage 6 as the $n$-step dial, and at stage 7 as
GAE's $\lambda$.

### Exploration vs exploitation

The agent only learns about actions it takes, and it only takes actions it
currently believes are good. Left alone, that loop is self-confirming: an action
undervalued by early noise is never tried again, so the error is never
corrected.

This project uses four different answers to that problem, and it is worth
noticing that they are genuinely different in kind:

| mechanism | where | how it works |
|---|---|---|
| $\varepsilon$-greedy | stages 1–4 | act randomly with probability $\varepsilon$ |
| exploring starts | stage 1 | begin episodes in random states |
| stochastic policy | stages 5–7 | exploration *is* the policy; it is learned |
| entropy bonus | stages 5–7 | pay the policy to stay uncertain |

The shift at stage 5 is a real conceptual change, not a tuning detail. Under
$\varepsilon$-greedy, exploration is bolted on from outside the model and the
objective knows nothing about it. Under a stochastic policy, exploration is part
of the thing being optimised — which means it can be shaped by the objective,
and also that it can *collapse*, which is a failure mode $\varepsilon$-greedy
does not have.

---

## Part II — The progression

