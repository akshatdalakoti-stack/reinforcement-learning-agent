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

