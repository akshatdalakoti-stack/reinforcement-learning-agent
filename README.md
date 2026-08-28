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

### Stage 1 — Tabular Q-learning

> `q_learning/` · GridWorld · `python -m q_learning.train --exploring-starts --alpha-schedule visit_count`

**1. The problem it solves.** Find an optimal policy in an unknown MDP, using
only sampled experience — no access to $P$ or $R$.

**2. Why nothing simpler suffices.** Value iteration solves the same MDP in a
few lines, but it *reads the transition model*. An agent dropped into a world it
has not been handed a specification for cannot do that. Q-learning replaces the
expectation $\mathbb{E}_{s'}$ in the Bellman equation with a sample, and the
sample is something the agent can actually obtain by acting.

**3. The mathematical idea.** The Bellman optimality equation solved by
stochastic approximation:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target}} - Q(s,a)\big]$$

Two properties deserve emphasis, because both are inherited by every later
stage:

- **Off-policy.** The $\max$ evaluates the *greedy* policy, while the action
  taken came from an $\varepsilon$-greedy behaviour policy. The agent explores
  with one policy and learns the value of another. This is what will make replay
  buffers legal in stage 3 — and illegal in stage 5.
- **Bootstrapping.** The target uses the agent's own current estimate of $s'$.

**4. How training changes.** Learning happens once per environment step, not
once per episode. There is no dataset, no epochs, no loss in the supervised
sense — just repeated application of a contraction mapping to a table.

**5. Failure modes.**
- **Coverage.** Convergence to $Q^*$ requires visiting every $(s,a)$ pair
  infinitely often. With a fixed start state, pairs off the optimal path are
  visited almost never, and their values stay wrong.
- **Constant $\alpha$ never converges.** Tabular Q-learning converges when the
  step sizes satisfy the Robbins–Monro conditions
  $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. A constant
  $\alpha$ violates the second, so in a *stochastic* environment the estimates
  bounce around $Q^*$ forever with a variance floor proportional to $\alpha$.
- **It does not scale.** One independent parameter per $(s,a)$ pair. Nothing
  learned about one state transfers to a similar one. This is what stage 2 fixes.

**6. The experiment.** `experiments/exp1_exploring_starts.py`. GridWorld is small
enough to solve exactly by value iteration, so we can report the *true* error
rather than a proxy. Four variants crossing fixed/exploring starts with
constant/decaying $\alpha$, 5 seeds, 6000 episodes:

| variant | $\max_{s,a} \Vert Q - Q^* \Vert$ | optimal actions | min visits per $(s,a)$ |
|---|---|---|---|
| fixed start, constant $\alpha$ | 0.262 | 16.4 / 18 | 33 |
| fixed start, $\alpha=1/n^{0.75}$ | 0.262 | 16.2 / 18 | 27 |
| exploring starts, constant $\alpha$ | 0.188 | 17.0 / 18 | 138 |
| exploring starts, $\alpha=1/n^{0.75}$ | **0.158** | **17.6 / 18** | 129 |

The update rule is byte-for-byte identical in all four rows. What differs is
which transitions the agent sees and how it averages them — and that alone
moves the error by 40%. Note that the decaying step size only helps *once
coverage is fixed*: rows 1 and 2 are identical, because a better averaging rule
cannot help a state you never visit.

![Convergence to Q*](results/plots/exp1_exploring_starts.png)

The figure separates the two effects by their *shape*, not just their level. The
two constant-$\alpha$ curves (blue, green) oscillate forever — that is the
Robbins–Monro violation, a variance floor proportional to $\alpha$ that no
amount of extra training removes. The two $\alpha = 1/n^{0.75}$ curves are
smooth, because the step sizes are square-summable and the estimates actually
settle. But settling is not the same as being *right*: the red curve settles
early and stays at 0.26, having converged confidently to a wrong answer in the
states its fixed start rarely reaches. Only the purple curve — coverage *and* a
valid step-size schedule — keeps descending.

![Q-learning learning curve](results/plots/q_learning_gridworld_curve.png)

The learned policy, and the exact optimum it is converging to:

![Learned policy](results/plots/q_learning_gridworld_policy.png)

The pale top-right corner is the coverage problem made visible: those states lie
off every good path from the start, so their values are still near their
initialisation while the rest of the grid has converged.

---

### Stage 2 — Deep Q-Network

> `dqn/` · CartPole-v1 · `python -m dqn.train --no-replay --no-target-network`

**1. The problem it solves.** Q-learning where the state space is continuous, so
a table is not merely large but *impossible*.

**2. Why the previous stage is insufficient.** CartPole's state is
$(x, \dot{x}, \theta, \dot{\theta}) \in \mathbb{R}^4$. There is no row to index.
Discretising trades one problem for two: the bin count explodes with dimension,
and bins do not share information — everything learned about
$(0.10, 0.02, 0.01, 0.30)$ is invisible to the neighbouring state
$(0.101, 0.02, 0.01, 0.30)$. The tabular representation has *no notion of
similarity between states*, and that is what has to change.

**3. The mathematical idea.** Replace the table with a parameterised function
$Q_\theta(s,a)$ and fit it by gradient descent on the squared TD error:

$$\mathcal{L}(\theta) = \mathbb{E}\Big[\big(\underbrace{r + \gamma \max_{a'} Q_\theta(s',a')}_{\text{treated as a constant}} - Q_\theta(s,a)\big)^2\Big]$$

Generalisation now comes for free: one shared parameter vector describes all
states, and similar inputs produce similar outputs by construction.

**4. How training changes.** A forward pass, a loss, a backward pass, an
optimiser step. The network emits all $|\mathcal{A}|$ values at once, so
$\max_{a'}$ is one cheap operation. Critically, the target is `detach`ed — this
is not an optimisation, it is the algorithm. Q-learning is a fixed-point
iteration, not a loss minimisation; if gradients flowed into the target, the
network could reduce the loss by dragging the target toward its prediction
instead of the reverse.

**5. Failure modes.** This is where it gets interesting — the naive version is
*worse* than the tabular one it generalises:

- **Correlated samples.** Consecutive states within an episode are nearly
  identical. Gradient descent assumes i.i.d. samples; a stream of near-duplicates
  gives a gradient that reflects one moment, not the task.
- **Non-stationary data distribution.** Improving the policy changes which states
  are visited, which changes the training distribution. The network chases its
  own tail and **catastrophically forgets** states it no longer visits.
- **The deadly triad.** Function approximation + bootstrapping + off-policy
  updates. Any two are safe. All three together have no convergence guarantee,
  and the tabular contraction argument no longer applies.

**6. The experiment.** The `online` row of
`experiments/exp2_replay_ablation.py` — see stage 3, where the comparison lives.

---

### Stage 3 — Experience replay

> `dqn/replay_buffer.py` · `python -m dqn.train --no-target-network`

**1. The problem it solves.** The correlated, non-stationary data stream of
stage 2.

**2. Why stage 2 is insufficient.** Online DQN trains on transitions in the order
they occur. Each gradient step overfits the current moment, and the next step
partially undoes it. Data is also used exactly once and thrown away, which is
wasteful when environment interaction is the expensive resource.

**3. The mathematical idea.** Store transitions $(s,a,r,s',d)$ in a circular
buffer and sample minibatches **uniformly at random**. A minibatch then mixes
many different episodes and many different past policies, restoring something
much closer to the i.i.d. assumption gradient descent needs.

The legality argument is the important part, and it is one line: **Q-learning is
off-policy.** Its target $r + \gamma\max_{a'}Q(s',a')$ never refers to the policy
that chose $a$. A transition collected 10,000 steps ago is still a valid sample
of the environment's dynamics. This is precisely why stage 5 cannot do the same.

**4. How training changes.** Acting and learning decouple. The agent may take one
environment step and one gradient step, or one step and four gradient steps, or
collect for a while and then train — these are now independent knobs
(`--train-freq`).

**5. Failure modes.**
- **Buffer size is a real hyperparameter.** Too small and the samples are still
  correlated; too large and the agent trains on transitions from policies so old
  they are no longer informative.
- **Uniform sampling is not obviously right.** Rare, high-TD-error transitions
  are drawn as often as common boring ones (the motivation for prioritised
  replay, not implemented here).
- **It does not fix the moving target — and that turns out to dominate.** Replay
  decorrelates the *inputs*; the regression *target* is still built from the
  network being updated. On CartPole the experiment below shows this is not a
  minor residual problem: with no target network, adding replay changes the
  final reward by **−0.0**, and makes the underlying divergence *worse*. Replay
  only pays off once stage 4 is in place. If you are reading this progression
  expecting each idea to add its own increment, this is the point where that
  expectation breaks.

**6. The experiment.** `experiments/exp2_replay_ablation.py` — three variants,
identical in every other respect, 3 seeds each. The middle variant is what
isolates replay; comparing `online` against the full agent would confound two
ideas.

Three seeds each, 120,000 environment steps, identical hyperparameters, judged
on the best periodic **greedy** evaluation:

| | no target network | target network |
|---|---|---|
| **no replay** | (a) 9.7 ± 0.1 · peak Q **8,121,814** | (c) 12.4 ± 0.9 · peak Q 39.0 |
| **replay** | (b) 9.7 ± 0.1 · peak Q **18,707,648** | (d) **456.7 ± 61.3** · peak Q 69.3 |

Main effects on best greedy reward:

| change | effect |
|---|---:|
| add a target network, *without* replay | +2.7 |
| add replay, *without* a target network | −0.0 |
| add a target network, *with* replay | **+446.9** |
| add replay, *with* a target network | **+444.3** |

**This is almost pure interaction.** Neither idea does anything on its own; the
two together are worth ~450 reward. That is a more interesting result than the
usual "both help a little", and the mechanism is legible in the peak-Q column:

- **Replay without a target network makes things actively worse** (peak Q
  18.7M vs 8.1M). Replay is doing its job — supplying many decorrelated,
  informative gradient steps — and every one of those steps feeds a
  self-referential target. Better data closes the divergence loop *faster*.
- **A target network without replay stops the divergence** (peak Q 39.0, a
  sane number) **but cannot learn.** Each gradient step sees a single
  transition, correlated with the last, so the network fits the current moment
  and forgets it immediately. Stable, and stuck.
- **Together**, the target supplies a stationary objective and replay supplies
  decorrelated data to fit it with — the two conditions ordinary supervised
  learning needs. Only then does the agent solve the task, reaching the
  threshold in **8,363 ± 2,363** environment steps.

So the honest version of the stage 3 → 4 story is not "replay helps, then the
target network helps more". It is: **replay and the target network fix different
halves of the same broken assumption, and deep Q-learning needs both.** Replay
makes the inputs look i.i.d.; the target network makes the labels stationary.
Supervised learning quietly assumes both, and stage 2 has neither.

![Factorial bars](results/plots/exp2_factorial_bars.png)

![Replay ablation](results/plots/exp2_replay_ablation.png)

---

### Stage 4 — Target networks

> `dqn/agent.py` · `python -m dqn.train`

**1. The problem it solves.** The regression target moving in lockstep with the
estimator.

**2. Why stage 3 is insufficient.** Replay fixed the input distribution but not
the target. Consider what one gradient step does:

$$\mathcal{L} = \big(Q_\theta(s,a) - [r + \gamma \max_{a'} Q_\theta(s',a')]\big)^2$$

$s$ and $s'$ are neighbouring states, so the network cannot raise $Q_\theta(s,a)$
without also raising $Q_\theta(s',a')$ — which raises the target — which raises
$Q_\theta(s,a)$ again. The estimate and the thing it is chasing are the same
function. There is no stationary target to converge to, and the value estimates
can run away.

**3. The mathematical idea.** Keep a frozen copy $\theta^-$, synchronised every
$k$ gradient steps, and build the target from it:

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$$

Between syncs, $y$ is a *constant*. The problem becomes a sequence of ordinary
supervised regressions against a stationary target — exactly the setting
gradient descent has guarantees for. The bootstrapping is still there; it has
just been made *piecewise stationary*.

**4. How training changes.** One extra network, and one extra hyperparameter: the
sync period $k$. `--tau > 0` switches to Polyak averaging
$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ every step, which is the same
idea applied smoothly instead of in jumps.

**5. Failure modes.**
- **$k$ too small** approaches having no target network at all.
- **$k$ too large** makes the targets stale, so value information propagates
  backwards through the state space very slowly.
- **Overestimation persists.** $\mathbb{E}[\max_a \hat{Q}] \ge \max_a \mathbb{E}[\hat{Q}]$
  — the max of noisy estimates is biased *upward*, and the target network does
  nothing about that. Watch `mean_q` climb above any achievable return. The fix
  is Double DQN, available here as `--double-dqn`:
  $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s',a'))$, which
  decouples selection from evaluation.

**6. The experiment.** `experiments/exp3_target_network_frequency.py` sweeps $k$
against a no-target-network control.

Three seeds per setting, 100,000 environment steps, replay on throughout:

| sync period $k$ | best greedy reward | steps to greedy ≥195 | peak mean predicted Q |
|---|---|---:|---:|
| *no target network* | 9.7 ± 0.1 | never | **18,707,648** |
| 1 | 9.7 ± 0.1 | never | **18,707,648** |
| 50 | 411.3 ± 125.5 | 23,440 | 165.2 |
| 250 | 403.5 ± 71.4 | **8,363** | 62.7 |
| 1,000 | **437.1 ± 89.0** | 36,750 | 23.5 |
| 5,000 | 9.7 ± 0.1 | never | **2.4** |

**First, a sanity check that the implementation is honest.** $k = 1$ reproduces
the no-target-network control *to the digit* — same reward, same 18,707,648 peak
Q, same standard deviation. It should: syncing after every gradient step means
the target network is always an exact copy of the online network, which is
definitionally the same algorithm. Any implementation where those two rows
differ has a bug in it.

**Both extremes fail, and the peak-Q column says they fail for opposite
reasons.** Read it top to bottom: 18.7M → 165 → 62.7 → 23.5 → 2.4. The sync
period is, quite directly, a speed limit on how fast value estimates may grow.

- At $k = 1$ there is no speed limit, and the estimates run away — the deadly
  triad, unimpeded.
- At $k = 5{,}000$ the limit is so tight that after 100,000 steps the network
  still values states at 2.4, against a true maximum of ~99. It has not
  diverged; it simply has not been allowed to learn. The targets are so stale
  that reward information cannot propagate backwards through the state space.
- In between — anywhere from 50 to 1,000 — it works, and the differences among
  those three rows are inside the seed noise at $n=3$.

That is the trade-off stated concretely: **too fast is divergence, too slow is
paralysis, and the useful range spans more than an order of magnitude.** The
project's default of 250 sits in that range and is the fastest to threshold
here, but 50 and 1,000 would have been defensible choices too.

![Target frequency trade-off](results/plots/exp3_target_frequency_tradeoff.png)

![Target frequency trade-off](results/plots/exp3_target_frequency_tradeoff.png)

At this point the value-based line is complete, and it solves CartPole:

| | env steps | training time | greedy evaluation | final mean Q |
|---|---:|---:|---|---:|
| DQN + replay + target | 200,036 | 193 s | **500.0 ± 0.0** (solved) | 91.3 |

The mean predicted Q settling at 91.3, just under the theoretical maximum of
99.3, is the sign that the value function is calibrated rather than merely
useful. Compare the millions reached without a target network.

![DQN learning curve](results/plots/dqn_cartpole_curve.png)

---

### Stage 5 — REINFORCE

> `reinforce/` · `python -m reinforce.train --gradient-variance 40`

**1. The problem it solves.** Optimising the policy *directly*, rather than
inferring it from values.

**2. Why stages 1–4 are insufficient.** Even a working DQN has structural
limits, all traceable to deriving $\pi$ from $Q$ by $\arg\max$:

- The policy is a **discontinuous** function of $\theta$. An arbitrarily small
  weight change can flip the argmax and change behaviour completely.
- The policy is **deterministic**, so exploration must be bolted on from outside
  ($\varepsilon$-greedy) and cannot be shaped by the objective.
- **Continuous actions are intractable**: $\arg\max_a Q(s,a)$ becomes its own
  optimisation problem at every step.
- When the optimal policy is genuinely **stochastic**, no $\arg\max$ can express it.

**3. The mathematical idea.** Parameterise $\pi_\theta(a|s)$ and ascend
$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ directly. The trajectory
distribution depends on $\theta$, so the gradient cannot pass inside the
expectation — until the **likelihood-ratio trick**,
$\nabla p = p \nabla \log p$:

$$\nabla_\theta J = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]$$

The single most important consequence: **the transition probabilities cancel**.
$\log p_\theta(\tau)$ contains $\sum_t \log P(s_{t+1}|s_t,a_t)$, which does not
depend on $\theta$ and differentiates to zero. That is why policy gradients work
without a model.

Causality lets us use the reward-to-go $G_t$ rather than the full-trajectory
return: an action at time $t$ cannot influence earlier rewards, so including them
only adds variance.

**4. How training changes.** Completely. There is no value function and no
bootstrapping. The agent collects a whole trajectory, computes returns backwards,
and takes **one** gradient step on

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_t \log \pi_\theta(a_t|s_t)\, A_t$$

which is not a loss but a *surrogate* whose gradient equals the negated policy
gradient. $A_t$ is detached — it is a weight on the log-probability, not
something to differentiate through. The data is then discarded, because it is
on-policy: the estimator is only valid for the policy that generated it.

**5. Failure modes.**
- **Variance.** $G_t$ sums every reward and every random transition to the end of
  the episode. Its variance compounds with the horizon.
- **No sense of expectation.** On CartPole every reward is $+1$, so a 200-step
  episode reinforces *every* action in it equally — including the one that
  eventually dropped the pole.
- **Sample inefficiency.** One gradient step per episode, then the data is gone.
- **Premature convergence.** Whatever gets sampled gets reinforced, which makes
  it more likely to be sampled. The entropy bonus opposes this.

Two variance reductions are implemented. The **baseline lemma** is the important
one: for any $b(s)$ that does not depend on the action, subtracting it from the
weight leaves the gradient's expectation unchanged. Writing the expectation over
actions out in full:

$$\mathbb{E}_{a \sim \pi_\theta}\big[b(s)\,\nabla_\theta \log \pi_\theta(a|s)\big]
= b(s) \sum_a \pi_\theta(a|s)\, \nabla_\theta \log \pi_\theta(a|s)
= b(s) \sum_a \nabla_\theta \pi_\theta(a|s)$$

$$= b(s)\, \nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\, \nabla_\theta 1 = 0$$

The second equality is the log-derivative trick run backwards
($\pi \nabla \log \pi = \nabla \pi$), and the last step is just the fact that a
probability distribution sums to 1 regardless of $\theta$. So subtracting a
baseline changes the variance but **not the expectation** — we get to choose
$b(s)$ freely to minimise noise, and stage 6 chooses the best one available.
`--baseline mean` uses the batch mean; `--baseline norm` also divides by the
batch std (biased, but very effective).

**6. The experiment.** `--gradient-variance N` freezes the parameters and
estimates the gradient $N$ times from $N$ independent rollouts. Every estimate
targets the *same* quantity $\nabla_\theta J$, so any disagreement between them
is pure estimator variance, with learning dynamics factored out. Measured at the
**end** of the canonical run (40 single-episode estimates at the trained
policy):

| measure | value | what 'no variance' would look like |
|---|---|---|
| mean pairwise cosine similarity | **+0.003** | 1.000 |
| $\Vert\text{mean grad}\Vert / \text{mean}\Vert\text{grad}\Vert$ | **0.118** | 1.000 |

Two rollouts from the *same* policy produce gradient estimates that are
essentially **orthogonal**. REINFORCE works at all only because those errors
average out over many updates — which is exactly why it needs so many. (Stage 6
repeats this probe at an *untrained* policy, where the number is +0.006, and
shows what the baseline does to it.)

The practical cost, on the canonical run:

| | env steps | episodes | training time | greedy evaluation |
|---|---:|---:|---:|---|
| REINFORCE (normalised baseline) | 200,130 | 508 | 271 s | 459.4 ± **56.1** |

Note the standard deviation. DQN and the next two stages all reach $500.0 \pm 0.0$;
REINFORCE lands near the top but *inconsistently*, which is the variance above
showing up in behaviour.

![REINFORCE learning curve](results/plots/reinforce_cartpole_curve.png)

---

### Stage 6 — Actor-Critic

> `actor_critic/` · `python -m actor_critic.train`

**1. The problem it solves.** REINFORCE's variance, and its inability to learn
before an episode ends.

**2. Why stage 5 is insufficient.** Quantified above: cosine similarity +0.003.
The estimator is unbiased but so noisy that most of each update is cancellation.
And $G_t$ is undefined until the episode terminates, so on a task with
1000-step episodes REINFORCE gets one update per 1000 steps.

**3. The mathematical idea.** Replace the Monte-Carlo return with the
**advantage**, estimated using a learned value function:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

"How much better was this action than this state was worth anyway?" By the
baseline lemma, subtracting $V(s)$ leaves the gradient unbiased while recentring
the weights around zero: better-than-expected actions get positive weight,
worse-than-expected ones negative, and a merely average action in a great state
gets $\approx 0$ instead of a large positive push.

We never learn $Q$ directly. The advantage comes from sampled rewards plus the
critic:

$$A_t = \underbrace{r_t + \gamma V(s_{t+1}) - V(s_t)}_{n=1,\ \text{the TD error } \delta_t} \qquad A_t = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$$

$n$ is the bias/variance dial: $n=1$ is minimum variance and maximum bias (it
inherits every error in $V$); $n=\infty$ *is* REINFORCE. Everything useful is in
between. This is the same trade-off from Part I, now exposed as a hyperparameter.

**4. How training changes.** Two networks — an **actor** emitting $\pi(a|s)$ and
a **critic** emitting $V(s)$ — with separate optimisers. The actor is updated
using the advantage as a weight; the critic regresses onto the $n$-step returns.
Because the advantage needs only $n$ steps, updates happen **mid-episode**. The
critic learns $V$, not $Q$: it is not choosing anything, so one number per state
suffices, which is a strictly easier regression than DQN's.

**5. Failure modes.**
- **Two coupled learners.** The actor trusts a critic that is itself learning.
  Early on $V$ is nonsense, so advantages are nonsense, so the policy moves
  badly — which changes the state distribution the critic is fitting. They can
  oscillate or lock into a bad equilibrium.
- **Bias.** Bootstrapping through an imperfect $V$ biases the gradient. Unlike
  REINFORCE, the estimator is no longer unbiased.
- **Still on-policy, still one step per batch.** Nothing yet allows reusing data.
- **Entropy collapse**, as in stage 5.

**6. The experiment.** `experiments/exp4_reinforce_vs_actor_critic.py` runs both
parts: the direct gradient-variance probe, and learning curves across seeds
(where variance shows up as the width of the band).

**Part 1 — the estimator, measured directly.** 60 independent single-episode
gradient estimates at identical (untrained) parameters:

| weighting | mean pairwise cosine similarity | $\Vert$mean grad$\Vert$ / mean$\Vert$grad$\Vert$ |
|---|---:|---:|
| raw return $G_t$ (no baseline) | +0.006 | 0.101 |
| centred and scaled (`--baseline norm`) | **+0.107** | **0.495** |

The baseline does not change what the gradient estimates *on average* — that is
the lemma above, and it is exact. What it changes is how much they disagree, and
the effect is large: **18× the cosine agreement and ~5× the signal-to-noise**,
from arithmetic that costs one mean and one standard deviation per batch.

**Part 2 — what that variance costs.** Four seeds, 120,000 steps each:

| variant | greedy reward (mean ± across-seed std) | per-seed results |
|---|---|---|
| REINFORCE, no baseline | 351.9 ± **156.2** | 500, **134**, 500, **274** |
| REINFORCE, normalised | **500.0 ± 0.0** | 500, 500, 500, 500 |
| Actor-Critic (n=32) | **500.0 ± 0.0** | 500, 500, 500, 500 |

The unbaselined estimator is not merely slower — it is *unreliable*. Two of four
seeds never got there, with identical hyperparameters and only the RNG differing.
Adding a constant baseline takes the across-seed spread from 156.2 to exactly
zero.

**An honest caveat about this stage.** On CartPole the *constant* baseline
already captures nearly all the available variance reduction, so Actor-Critic's
learned, state-dependent $V(s)$ does not beat it on final score — both sit at
500.0 ± 0.0. What the critic buys here is visible elsewhere in the log:

| | updates from 120k steps | wall clock |
|---|---:|---:|
| REINFORCE (normalised) | 340–430 | 156–173 s |
| Actor-Critic (n=32) | **3,060–4,043** | **54–67 s** |

Roughly **ten times as many policy updates** from the same samples, at a third
of the wall clock, because the advantage is available every 32 steps instead of
once per episode. On a task with a longer horizon, or one where a good constant
baseline does not exist because state values genuinely differ, that gap becomes
a quality gap too. CartPole is not that task, and the table above should not be
read as though it were.

![REINFORCE vs Actor-Critic](results/plots/exp4_reinforce_vs_actor_critic.png)

| | env steps | episodes | updates | training time | greedy evaluation |
|---|---:|---:|---:|---:|---|
| Actor-Critic (n = 32) | 200,203 | 1,029 | 6,721 | 145 s | **500.0 ± 0.0** (solved) |

The update count is the structural change made concrete: REINFORCE managed 508
updates from the same 200,000 environment steps, one per episode. Actor-Critic
performed 6,721 — one per 30 steps — because a bootstrapped advantage does not
have to wait for the episode to end.

---

### Stage 7 — PPO

> `ppo/` · LunarLander-v3 · `python -m ppo.train --env LunarLander-v3 --total-steps 600000`

**1. The problem it solves.** Reusing on-policy data safely — many gradient
steps per batch instead of one.

**2. Why stage 6 is insufficient.** Every policy-gradient method so far throws
its data away after a single step. That is not squeamishness, it is forced: the
estimator is an expectation over the *current* policy, so the instant $\theta$
moves, the collected data is off-policy and the estimator is biased. With a
2048-step rollout and one gradient step, sample efficiency is poor.

**3. The mathematical idea, in two parts.**

*Importance sampling* makes reuse possible:

$$\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A(s,a)\right], \qquad r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$$

But it is only trustworthy while the policies are close. As $r$ departs from 1
the variance explodes, and worse, maximising $r A$ with no constraint drives
$r \to \infty$ wherever $A > 0$. One large step off a noisy advantage gives a
policy that is both bad *and* badly estimated — and unlike supervised learning
there is no fixed dataset to recover from, because the broken policy collects the
next batch. That is **performance collapse**.

*Clipping* bounds it. TRPO used a hard KL constraint and second-order
optimisation; PPO gets most of the benefit first-order:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\min\big(r(\theta)A,\ \ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\big)\Big]$$

The asymmetry is the whole idea:

- $A > 0$: raising $r$ raises the objective, but the clipped term caps at
  $(1+\epsilon)A$. Past that the `min` selects the clipped branch, which is
  *constant in $\theta$*, so **its gradient is zero**. The incentive to keep
  pushing disappears.
- $A < 0$: lowering $r$ raises the objective; the clipped term floors at
  $(1-\epsilon)A$, and the `min` again selects it once $r < 1-\epsilon$.

So the objective is a **pessimistic lower bound**: it takes the less optimistic
of the clipped and unclipped estimates. There is no barrier and no constraint —
the gradient simply goes flat outside the trust region, which is why plain Adam
can optimise it.

*Why `min` and not just `clip`.* Using `clip(r)A` alone would zero the gradient
whenever $r$ left the region **in either direction**, including when the policy
has already drifted too far and needs to come *back*. The `min` keeps the
unclipped term active in exactly that case. Dropping it is a classic way to get
a PPO that silently cannot recover from a bad update.

*GAE.* A2C picks one $n$; GAE exponentially averages over all $n$ at once:

$$A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l} \quad\Longleftrightarrow\quad A_t = \delta_t + \gamma\lambda(1-d_t)A_{t+1}$$

$\lambda = 0$ is the one-step TD error, $\lambda = 1$ is Monte-Carlo; $\lambda
\approx 0.95$ is a finer dial on the same bias/variance axis.

**4. How training changes.** The loop stops being episode-shaped. PPO collects a
fixed-size rollout that deliberately **crosses episode boundaries**, then runs
several epochs of shuffled minibatch SGD on it:

$$\mathcal{L}_{\text{total}} = \underbrace{-\,\mathcal{L}^{\text{CLIP}}}_{\text{actor}} + \underbrace{c_v \mathcal{L}^{\text{VF}}}_{\text{critic}} - \underbrace{c_e\, \mathcal{H}[\pi_\theta]}_{\text{entropy}}$$

Mind the first sign: $\mathcal{L}^{\text{CLIP}}$ above is an objective to be
*maximised*, so the quantity actually handed to the optimiser negates it. The
entropy term is negated for the opposite reason — we want entropy high, so
minimising $-\mathcal{H}$ pays the policy to stay uncertain.

Storing $\log\pi_{\text{old}}$ and $V_{\text{old}}$ at collection time is what
makes this a PPO buffer rather than an A2C one — after the first epoch the
network is no longer the policy that gathered the data, and $\log\pi_{\text{old}}$
is what lets PPO *measure* the drift it is bounding.

**5. Failure modes.**
- **Entropy collapse.** Clipping bounds each step, not cumulative drift. Over
  many updates the policy can still become deterministic and stop exploring.
  Watch `entropy` fall toward 0.
- **Too many epochs.** Each epoch pushes further from $\pi_{\text{old}}$; once
  most samples are clipped the extra epochs compute zero-gradient batches.
  `clip_fraction` and `approx_kl` are the metrics to watch, and `--target-kl`
  is the safety net.
- **Value/policy interference.** With a shared trunk the value loss is
  numerically much larger and can swamp the policy gradient — hence `value_coef`,
  and `--separate-networks` to test the choice.
- **Clipping is a heuristic, not a guarantee.** It bounds the ratio per sample,
  which is not the same as bounding the KL divergence.

**6. The experiment.** `experiments/exp5_ppo_clipping.py`. The clean control is
not "PPO vs A2C" — that changes several things at once. It is *the same PPO
agent, same rollouts, same epochs, with the clip range set so wide it never
binds* (`--clip-epsilon 10`). That variant is precisely "importance-weighted
policy gradient with 10 epochs of reuse", which is the thing clipping was
invented to fix.

Three seeds, 80,000 environment steps on CartPole-v1:

| variant | greedy reward | mean KL/update | max KL | clip fraction | collapses |
|---|---|---:|---:|---:|---:|
| **PPO, clip 0.2, 10 epochs** | **488.6 ± 13.6** | 0.0041 | 0.0076 | 0.041 | **0.0** |
| no clipping, 10 epochs | 207.0 ± 20.8 | **0.3500** | **1.8598** | 0.001 | **10.3** |
| PPO, clip 0.2, 1 epoch | 377.1 ± 86.8 | 0.0002 | 0.0011 | 0.000 | 4.7 |
| PPO, clip 0.05, 10 epochs | 481.2 ± 22.2 | 0.0005 | 0.0007 | 0.105 | 0.0 |

Row 2 is the whole argument. Same agent, same rollouts, same ten epochs, same
optimiser — only the trust region is removed. The policy then moves **243×
further per update** (max KL 1.86 vs 0.0076), suffers **10.3 collapse events per
run** instead of zero, and ends at less than half the reward. This is
performance collapse happening exactly where the theory says it should.

Rows 3 and 4 rule out two cheaper explanations:

- **Row 3** keeps clipping but takes only one epoch per rollout, so there is
  almost no off-policy drift for the clip to bound. Its KL is tiny and it never
  clips — yet it reaches only 377 ± 87, because it is throwing away nine tenths
  of the optimisation it paid for in samples. So clipping is not merely a
  safety device: it is what *buys* the extra epochs. **Clipping and multi-epoch
  reuse are the same idea, and neither is worth much alone.**
- **Row 4** tightens the region to 0.05. It clips 2.5× as often (0.105 vs
  0.041), stays perfectly stable, and gives up a little final performance.
  That is the conservative end of the dial behaving as advertised, and it
  confirms the effect in row 1 comes from *bounding* the step rather than from
  clipping being incidentally helpful.

![PPO clipping ablation](results/plots/exp5_ppo_clipping.png)

The unclipped run (red) is the shape to remember: it climbs to ~100, collapses
to near zero, climbs again, collapses again — five or six times over. That is
not noise, it is the trust region being absent. Note also that the curves end at
different episode counts despite an identical 80,000-step budget: the good
policies survive ~167 steps per episode and so fit fewer episodes into the
budget, while the collapsing run averages ~47. The x-axis is episodes, so a
*shorter* line here means a *better* agent.

And the headline run, on the environment the whole progression was building
toward:

| | env steps | episodes | updates | training time | greedy evaluation |
|---|---:|---:|---:|---:|---|
| PPO on LunarLander-v3 | 599,800 | 1,959 | 293 (2,930 epochs) | 820 s | **250.3 ± 46.8** — solved (threshold 200) |
| PPO on CartPole-v1 | 200,578 | 754 | 98 (980 epochs) | 336 s | **500.0 ± 0.0** — solved |

Two details in the LunarLander curve are worth reading off the figure. Policy
entropy starts at **1.382 nats** against a theoretical maximum of
$\ln 4 = 1.386$ — the uniform distribution over four actions, confirming that
the `gain=0.01` policy-head initialisation does what it claims — and falls to
~0.57 as the policy commits.

And mean episode length **rises to ~683 steps around episode 388, then falls
back to ~283**. That non-monotonicity is the agent's curriculum, discovered on
its own: it first learns to *hover*, because not crashing is worth far more than
crashing, and only then learns to actually *land*, which is both higher-scoring
and much faster. A reward curve alone would have shown steady improvement and
hidden the fact that two different behaviours were being learned in sequence.

![PPO on LunarLander](results/plots/ppo_lunarlander_curve.png)

---

### The progression on one task

Stages 2–7 all run on CartPole-v1 in `run_all.py`, plotted against environment
steps so that sample efficiency is directly comparable:

![Progression on CartPole](results/plots/progression_cartpole.png)

The curve plotted is the **greedy** policy, sampled every 5,000 environment
steps. Plotting training reward instead would mostly compare exploration
schemes: DQN acts with ε = 0.05 throughout, and on CartPole one random action
ends the episode, so its training reward oscillates between roughly 90 and 250
while the policy it would actually deploy scores a flat 500. That version of the
figure is saved alongside as `progression_cartpole_training.png` — the contrast
between the two is itself instructive.

---

## Repository layout

```
rl-agent/
├── environments/
│   ├── gridworld.py        hand-written MDP + exact value iteration
│   └── registry.py         Gymnasium factories, seeding, space introspection
├── q_learning/             stage 1: Q-table, epsilon-greedy, Bellman update
├── dqn/
│   ├── network.py          the MLP that replaces the table
│   ├── replay_buffer.py    stage 3: circular buffer, uniform sampling
│   ├── agent.py            stages 2-4 in one class, replay/target as flags
│   └── train.py
├── reinforce/              stage 5: policy network, returns, log-prob loss
├── actor_critic/           stage 6: separate Actor and Critic, TD advantage
├── ppo/
│   ├── networks.py         shared-trunk actor-critic
│   ├── rollout.py          rollout storage + GAE
│   ├── agent.py            clipped surrogate objective
│   └── train.py
├── evaluation/
│   ├── metrics.py          per-episode recording, moving averages, CSV/JSON
│   ├── plotting.py         learning curves, ablation bands, policy maps
│   ├── evaluate.py         greedy rollouts, PeriodicEvaluator, checkpoint CLI
│   ├── report.py           rebuild the summary table + figures from logs
│   └── utils.py            seeding, device resolution, paths
├── experiments/            exp1-exp5 + run_all.py
├── checkpoints/            saved models
├── results/{plots,logs}/   figures and per-episode CSVs
├── tests_smoke.py          correctness checks for the hand-written maths
├── requirements.txt
└── README.md
```

Two utilities worth knowing about:

```bash
python tests_smoke.py       # 26 correctness checks, ~30 s
python -m evaluation.report # rebuild tables/figures from logs, no retraining
```

`tests_smoke.py` targets the places where a bug would be invisible in a learning
curve — it checks that the GridWorld sampler actually matches its own
`transition_model()`, that value iteration's output satisfies the Bellman
equation, that GAE reduces to the TD error at λ=0 and the Monte-Carlo advantage
at λ=1, that the DQN target network receives no gradient and syncs exactly on
schedule, that PPO's ratio starts at exactly 1 and its clipped objective has
zero gradient outside the trust region *but a live one when the policy needs to
come back*, and that every checkpoint round-trips to identical behaviour.

Stages 2–4 deliberately share one `DQNAgent`. Replay and the target network are
constructor flags, not subclasses, so the ablations compare identical code paths
with exactly one idea removed. If they were separate classes, any difference in
the curves could be an accident of implementation.

---

## Metrics, logs and reproducibility

Every training run records, per episode:

| metric | where |
|---|---|
| episode reward | all stages |
| moving-average reward | computed by `metrics.moving_average` |
| episode length | all stages |
| training loss | DQN (Huber), REINFORCE/A2C (policy loss), PPO (total) |
| policy entropy | REINFORCE, Actor-Critic, PPO |
| environment steps | all stages — the sample-efficiency axis |
| training time | all stages, via `MetricsRecorder.elapsed` |
| periodic greedy evaluation | every 5,000 env steps (`--eval-every`), all deep stages |
| final performance | greedy evaluation, exploration off, unseen seeds |

Plus per-algorithm diagnostics: $\varepsilon$ and mean predicted $Q$ for DQN;
gradient norm and return std for REINFORCE; value loss and mean $V$ for A2C;
`clip_fraction` and `approx_kl` for PPO.

Runs write `results/logs/<run_name>.csv` (one row per episode) and
`<run_name>.json` (run summary), and figures to `results/plots/`.

**Training reward is not final performance.** During training the agent is still
exploring, so the logged reward measures the behaviour policy. Every reported
"eval reward" in this README comes from `evaluation/evaluate.py`, running the
greedy/deterministic policy on seeds the agent never trained on.

Seeding covers Python's `random`, NumPy, PyTorch (CPU and CUDA), the Gymnasium
environment, and each agent's own sampling RNG. Exact bit-level reproducibility
across machines is still not guaranteed — BLAS reductions differ — but same
machine, same seed reproduces. That was checked rather than assumed: re-running
each stage from scratch reproduced its evaluation to the digit
(REINFORCE 459.35 ± 56.09 and a +0.003 gradient cosine both times; online DQN
9.65 ± 0.91 both times), and re-evaluating the saved checkpoints reproduced the
same numbers a third time — which also confirms `save`/`load` is lossless.

---

## Implementation details that matter more than they look

Things that are easy to get wrong and quietly cost you a working agent:

**`terminated` vs `truncated`.** Gymnasium reports these separately and the
distinction is not cosmetic. Termination means the MDP ended, so
$V(s_{\text{next}}) = 0$. Truncation means a time limit fired while the future
was still real, so you must keep bootstrapping. Storing a truncation as
`done=True` teaches the agent that surviving to the time limit is worth exactly
as much as failing — which caps every value estimate in the problem. Every
training loop here passes `terminated` alone into the buffer and the TD target.

**Detaching the target.** Q-learning is a fixed-point iteration, not a loss
minimisation. If gradients flow into the TD target, the network can reduce the
loss by moving the target toward its prediction instead of the reverse. Same for
the advantage in stages 6–7: it is a *weight* on the log-probability, not
something to differentiate through.

**Huber over MSE.** Early TD errors can be huge; MSE turns them into
proportionally huge gradients. `SmoothL1` is quadratic near zero and linear
beyond, bounding the gradient without changing the minimiser.

**Logits, not probabilities.** `Categorical(logits=...)` applies
`log_softmax` internally. Computing probabilities first and taking their log can
underflow to $-\infty$ and poison the gradient.

**Random tie-breaking in `argmax`.** With a zero-initialised Q-table every action
ties, and NumPy's `argmax` always returns index 0 — so the agent marches in one
direction forever. `q_learning/agent.py` breaks ties uniformly.

**Gradient steps per environment step.** Training every step with a large batch
was what kept DQN pinned at ~90 reward here for several attempts. Dropping to
one gradient step per 4 environment steps (`--train-freq 4`) was the change that
made it solve the task — too much optimisation per sample overfits the buffer.

**Adam's `eps` in PPO.** PPO uses `eps=1e-5`, not torch's `1e-8` default. Once
most of a batch is clipped the gradients are near zero, and the smaller
denominator stops Adam from taking enormous steps on numerical noise.

**The state you bootstrap from at a rollout boundary.** PPO's rollouts cross
episode boundaries, so the training loop calls `env.reset()` in the middle of
collection. If you then hand `update()` the *current* `state`, and the rollout
happened to fill up exactly on a terminal step, you are bootstrapping
$V(s)$ from a freshly-reset observation belonging to a **different episode** —
splicing two unrelated trajectories together at the seam. `ppo/train.py` tracks
`boundary_state` / `boundary_done` separately from the loop's `state` for this
reason. It fires rarely (roughly once per 300 rollouts at LunarLander episode
lengths), which is exactly what makes it the kind of bug that survives for a
long time. `tests_smoke.py` pins it.

**Advantage normalisation is per minibatch in PPO**, per rollout in A2C. It
keeps the effective policy step size stable as the reward scale changes during
training. It is technically biased, and worth it.

---

