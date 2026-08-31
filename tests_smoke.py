"""Fast correctness checks for the hand-written components.

    python tests_smoke.py

Not a substitute for the training runs. These target the places where a silent
bug would be invisible in a learning curve -- a wrong discount, a target network
that is secretly being trained, a GAE recursion that ignores episode
boundaries. Every check below tests something this project implements by hand
rather than something PyTorch or Gymnasium already guarantees.
"""

from __future__ import annotations

import numpy as np
import torch

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_gridworld_transitions():
    """P(s'|s,a) must be a distribution, and step() must actually sample from it."""
    from environments.gridworld import GridWorld

    env = GridWorld(slip_prob=0.2)
    P, _ = env.transition_model()
    check("gridworld: P sums to 1 over next states", np.allclose(P.sum(axis=2), 1.0))
    check("gridworld: P is non-negative", bool((P >= 0).all()))

    s, a = env.to_index((2, 0)), 1  # RIGHT from a free cell
    counts = np.zeros(env.n_states)
    sampler = GridWorld(slip_prob=0.2)
    for i in range(20_000):
        sampler.reset(seed=i)
        sampler._agent = env.to_cell(s)
        nxt, _, _, _, _ = sampler.step(a)
        counts[nxt] += 1
    empirical = counts / counts.sum()
    check("gridworld: sampled dynamics match transition_model",
          np.allclose(empirical, P[s, a], atol=0.02),
          f"max diff {np.abs(empirical - P[s, a]).max():.4f}")


def test_value_iteration_fixed_point():
    """V* must satisfy the Bellman optimality equation it was derived from."""
    from environments.gridworld import GridWorld, value_iteration

    env = GridWorld(slip_prob=0.1)
    gamma = 0.95
    _, V = value_iteration(env, gamma=gamma)
    P, R = env.transition_model()
    residual = max(
        abs(float((R[s] + gamma * P[s] @ V).max()) - float(V[s]))
        for s in env.decision_states()
    )
    check("value iteration: Bellman residual ~ 0", residual < 1e-6, f"residual {residual:.2e}")


def test_q_learning_deterministic_convergence():
    """On a deterministic grid with full coverage, Q must reach Q* almost exactly."""
    from environments.gridworld import GridWorld, value_iteration
    from q_learning.agent import QLearningAgent

    env = GridWorld(slip_prob=0.0, max_steps=60)
    q_star, _ = value_iteration(env, gamma=0.95)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.95,
                           epsilon_start=1.0, epsilon_end=0.2,
                           epsilon_decay=0.999, seed=0)
    for ep in range(4000):
        state, _ = env.reset(seed=ep, options={"exploring_start": True})
        while True:
            a = agent.act(state)
            nxt, r, term, trunc, _ = env.step(a)
            agent.update(state, a, r, nxt, term)
            state = nxt
            if term or trunc:
                break
        agent.decay_epsilon()
    ds = env.decision_states()
    gap = float(np.abs(agent.q_table[ds] - q_star[ds]).max())
    check("q-learning: converges to Q* on a deterministic grid", gap < 0.02, f"gap {gap:.4f}")


def test_discounted_returns():
    """G_t must match the closed form on a known reward sequence."""
    from reinforce.agent import ReinforceAgent

    agent = ReinforceAgent(obs_dim=2, n_actions=2, gamma=0.9, seed=0)
    got = agent.discounted_returns([1.0, 2.0, 3.0]).numpy()
    want = np.array([1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3.0], dtype=np.float32)
    check("reinforce: discounted returns", np.allclose(got, want, atol=1e-5), f"{got} vs {want}")


def test_gae_limits():
    """lambda=0 must give the TD error; lambda=1 the Monte-Carlo advantage."""
    from ppo.rollout import compute_gae

    rng = np.random.default_rng(0)
    T = 12
    rewards = rng.normal(size=T).astype(np.float32)
    values = rng.normal(size=T).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    gamma, last_value = 0.99, 0.5

    adv0, _ = compute_gae(rewards, values, dones, last_value, False, gamma, 0.0)
    next_values = np.append(values[1:], last_value)
    check("gae: lambda=0 equals the one-step TD error",
          np.allclose(adv0, rewards + gamma * next_values - values, atol=1e-5))

    adv1, ret1 = compute_gae(rewards, values, dones, last_value, False, gamma, 1.0)
    mc = np.zeros(T, dtype=np.float64)
    running = last_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        mc[t] = running
    check("gae: lambda=1 equals the Monte-Carlo advantage",
          np.allclose(adv1, mc - values, atol=1e-4))
    check("gae: returns = advantages + values", np.allclose(ret1, adv1 + values, atol=1e-5))


def test_gae_respects_done():
    """A terminal flag must cut the bootstrap chain."""
    from ppo.rollout import compute_gae

    adv, _ = compute_gae(
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([5.0, 5.0, 5.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        last_value=5.0, last_done=False, gamma=0.99, gae_lambda=0.95,
    )
    # t=1 terminated, so its advantage is exactly r - V(s) with no bootstrap.
    check("gae: done truncates bootstrapping", abs(float(adv[1]) - (1.0 - 5.0)) < 1e-5,
          f"got {adv[1]:.4f}, want {1.0 - 5.0:.4f}")


def test_gae_terminal_at_rollout_boundary():
    """A rollout that ends exactly on a termination must not bootstrap.

    Regression test. PPO's rollouts deliberately cross episode boundaries, so
    the training loop resets the environment mid-rollout. If it then hands
    ``update()`` the freshly-reset observation with ``last_done=False``, GAE
    splices two unrelated episodes together at the seam. The loop tracks the
    true successor separately to prevent exactly this.
    """
    from ppo.rollout import compute_gae

    rewards = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    values = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # terminated on the last step

    correct, _ = compute_gae(rewards, values, dones, last_value=42.0, last_done=True,
                             gamma=0.99, gae_lambda=0.95)
    wrong, _ = compute_gae(rewards, values, dones, last_value=42.0, last_done=False,
                           gamma=0.99, gae_lambda=0.95)
    check("gae: terminal at the rollout boundary zeroes the bootstrap",
          abs(float(correct[-1]) - (2.0 - 5.0)) < 1e-5,
          f"got {correct[-1]:.4f}, want {2.0 - 5.0:.4f}")
    check("gae: mislabelling that boundary would corrupt the advantage",
          abs(float(wrong[-1]) - float(correct[-1])) > 1.0,
          "the two cases must differ, or the test proves nothing")


def test_replay_buffer():
    """Circular overwrite, correct shapes, and a correct latest()."""
    from dqn.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=5, obs_dim=3, seed=0)
    for i in range(8):
        buf.push(np.full(3, i, dtype=np.float32), i % 2, float(i),
                 np.full(3, i + 1, dtype=np.float32), i == 7)
    check("replay: length caps at capacity", len(buf) == 5)
    check("replay: oldest entries were overwritten", 0.0 not in set(buf.rewards.tolist()))

    batch = buf.sample(4)
    check("replay: batch shapes",
          batch.states.shape == (4, 3) and batch.actions.shape == (4, 1)
          and batch.rewards.shape == (4, 1) and batch.dones.shape == (4, 1))
    latest = buf.latest(1)
    check("replay: latest() returns the newest transition",
          float(latest.rewards[0, 0]) == 7.0 and float(latest.dones[0, 0]) == 1.0)


def test_dqn_target_is_detached():
    """No gradient may flow into the bootstrap target."""
    from dqn.agent import DQNAgent

    agent = DQNAgent(obs_dim=4, n_actions=2, buffer_size=200, batch_size=8,
                     learning_starts=0, train_freq=1, seed=0)
    for i in range(50):
        agent.observe(np.random.randn(4).astype(np.float32), i % 2, 1.0,
                      np.random.randn(4).astype(np.float32), False)
    before = [p.detach().clone() for p in agent.target_net.parameters()]
    agent.train_step()
    check("dqn: target network unchanged by a gradient step",
          all(torch.equal(b, a) for b, a in zip(before, agent.target_net.parameters())))
    check("dqn: target network carries no grads",
          all(p.grad is None for p in agent.target_net.parameters()))
    check("dqn: online network did receive gradients",
          any(p.grad is not None and p.grad.abs().sum() > 0
              for p in agent.online_net.parameters()))


def test_dqn_target_sync():
    """The target must be copied exactly on schedule, and not before."""
    from dqn.agent import DQNAgent

    agent = DQNAgent(obs_dim=4, n_actions=2, buffer_size=200, batch_size=8,
                     learning_starts=0, train_freq=1, target_update_freq=5, seed=0)
    for i in range(100):
        agent.observe(np.random.randn(4).astype(np.float32), i % 2, 1.0,
                      np.random.randn(4).astype(np.float32), False)
    for _ in range(4):
        agent.train_step()
    check("dqn: target lags the online net between syncs",
          any(not torch.equal(t, o) for t, o in
              zip(agent.target_net.parameters(), agent.online_net.parameters())))
    agent.train_step()  # the 5th gradient step triggers the sync
    check("dqn: target matches the online net right after a sync",
          all(torch.equal(t, o) for t, o in
              zip(agent.target_net.parameters(), agent.online_net.parameters())))
    check("dqn: sync counter incremented", agent.target_syncs == 1)


def test_ppo_clipping_gradients():
    """The clip must kill the gradient outside the region -- but the min() must
    keep it alive when the policy needs to come back."""

    def surrogate_grad(ratio_value: float, advantage: float, eps: float = 0.2) -> float:
        adv = torch.tensor([advantage])
        log_ratio = torch.tensor([np.log(ratio_value)], requires_grad=True)
        ratio = log_ratio.exp()
        loss = -torch.min(ratio * adv,
                          torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
        loss.backward()
        return float(log_ratio.grad.abs().item())

    check("ppo: gradient is live inside the trust region",
          surrogate_grad(1.0, 1.0) > 1e-6)
    check("ppo: gradient vanishes past the clip (A>0, r>1+eps)",
          surrogate_grad(1.5, 1.0) < 1e-9, f"grad {surrogate_grad(1.5, 1.0):.3e}")
    check("ppo: gradient vanishes past the clip (A<0, r<1-eps)",
          surrogate_grad(0.5, -1.0) < 1e-9, f"grad {surrogate_grad(0.5, -1.0):.3e}")
    # The asymmetric cases the min() exists to protect: the policy has drifted
    # the *wrong* way and must be pulled back, so the gradient must survive.
    check("ppo: min() keeps the gradient live for recovery (A>0, r<1-eps)",
          surrogate_grad(0.5, 1.0) > 1e-6)
    check("ppo: min() keeps the gradient live for recovery (A<0, r>1+eps)",
          surrogate_grad(1.5, -1.0) > 1e-6)


def test_ppo_ratio_starts_at_one():
    """Before any update pi_new == pi_old, so every ratio must be exactly 1."""
    from ppo.agent import PPOAgent

    agent = PPOAgent(obs_dim=4, n_actions=2, rollout_size=32, seed=0)
    states = np.random.randn(32, 4).astype(np.float32)
    actions, log_probs = [], []
    for s in states:
        a, lp, _ = agent.act_with_value(s)
        actions.append(a)
        log_probs.append(lp)
    new_lp, _, _ = agent.policy_value_net.evaluate_actions(
        torch.as_tensor(states), torch.as_tensor(actions)
    )
    ratio = (new_lp - torch.as_tensor(log_probs)).exp()
    check("ppo: ratio is exactly 1 before any optimisation step",
          torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5),
          f"max deviation {(ratio - 1).abs().max():.2e}")


def test_policy_entropy_starts_maximal():
    """The gain=0.01 policy head should make the initial policy near-uniform."""
    from ppo.networks import ActorCriticNetwork

    net = ActorCriticNetwork(obs_dim=8, n_actions=4)
    entropy = float(net.distribution(torch.randn(256, 8)).entropy().mean().item())
    check("init: initial policy entropy ~ ln(n_actions)",
          abs(entropy - np.log(4)) < 0.02, f"{entropy:.4f} vs {np.log(4):.4f}")


def test_checkpoint_roundtrip():
    """Every agent must reload to identical greedy behaviour."""
    import os
    import tempfile

    from actor_critic.agent import ActorCriticAgent
    from dqn.agent import DQNAgent
    from ppo.agent import PPOAgent
    from reinforce.agent import ReinforceAgent

    obs = np.random.randn(4).astype(np.float32)
    tmp = tempfile.mkdtemp()
    for name, agent in [
        ("dqn", DQNAgent(obs_dim=4, n_actions=2, seed=0)),
        ("reinforce", ReinforceAgent(obs_dim=4, n_actions=2, seed=0)),
        ("actor_critic", ActorCriticAgent(obs_dim=4, n_actions=2, seed=0)),
        ("ppo", PPOAgent(obs_dim=4, n_actions=2, seed=0)),
    ]:
        path = os.path.join(tmp, f"{name}.pt")
        agent.save(path)
        restored = type(agent).load(path, device="cpu")
        check(f"checkpoint: {name} round-trips to the same greedy action",
              agent.act(obs, greedy=True) == restored.act(obs, greedy=True))


def main() -> int:
    print("Smoke tests for the hand-written components\n")
    for test in [
        test_gridworld_transitions,
        test_value_iteration_fixed_point,
        test_q_learning_deterministic_convergence,
        test_discounted_returns,
        test_gae_limits,
        test_gae_respects_done,
        test_gae_terminal_at_rollout_boundary,
        test_replay_buffer,
        test_dqn_target_is_detached,
        test_dqn_target_sync,
        test_ppo_clipping_gradients,
        test_ppo_ratio_starts_at_one,
        test_policy_entropy_starts_maximal,
        test_checkpoint_roundtrip,
    ]:
        print(f"\n{test.__name__}")
        test()

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
