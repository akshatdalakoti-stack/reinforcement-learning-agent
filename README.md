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

