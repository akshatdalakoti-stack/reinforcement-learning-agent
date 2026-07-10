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

