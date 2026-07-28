"""Train REINFORCE on CartPole, and measure its gradient variance.

    python -m reinforce.train --episodes 1500 --baseline norm

The variance analysis is the point of this stage, not an extra.  REINFORCE
usually *does* learn CartPole; what it does badly is learn it *reliably*.  This
script quantifies that in two ways:

* ``--gradient-variance N`` estimates the policy gradient N times from N
  independent single-episode rollouts **at the same parameters**, then reports
  the cosine similarity between those estimates.  If single-trajectory gradients
  pointed the same way, similarity would be near 1.0.  It is not.
* the returned recorder keeps per-episode reward, so
  ``experiments/exp4_reinforce_vs_actor_critic.py`` can plot the across-seed
  spread that this variance produces.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed
from reinforce.agent import ReinforceAgent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="REINFORCE on a Gymnasium environment")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--max-env-steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--baseline", choices=["none", "mean", "norm"], default="norm")
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--batch-episodes", type=int, default=1,
                   help="trajectories averaged per update; >1 cuts variance as 1/N")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--gradient-variance", type=int, default=0,
                   help="estimate the gradient this many times at fixed parameters")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="env steps between greedy evaluations during training; "
                        "0 disables. Training reward reflects the exploring "
                        "policy, so this is the honest progress signal.")
    p.add_argument("--eval-rollouts", type=int, default=5,
                   help="episodes per periodic evaluation")
    p.add_argument("--run-name", type=str, default="reinforce_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_agent(args, spec, device) -> ReinforceAgent:
    return ReinforceAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        baseline=args.baseline,
        entropy_coef=args.entropy_coef,
        batch_episodes=args.batch_episodes,
        grad_clip=args.grad_clip,
        device=device,
        seed=args.seed,
    )


def collect_episode(agent: ReinforceAgent, env, seed: int | None = None):
    """Run one full trajectory, recording log-probs inside the agent."""
    state, _ = env.reset(seed=seed)
    rewards: list[float] = []
    while True:
        action = agent.act_and_record(state)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    agent.finish_episode(rewards)
    return rewards


def measure_gradient_variance(agent: ReinforceAgent, env, n_samples: int, seed: int) -> dict:
    """How much do independent single-trajectory gradients disagree?

    The parameters are held fixed: every estimate is of the *same* quantity
    ``grad J(theta)``, differing only in the sampled trajectory.  We report the
    mean pairwise cosine similarity, plus the ratio of the mean estimate's norm
    to the mean of the individual norms -- a signal-to-noise measure that would
    be 1.0 if every estimate agreed and near 0 if they were pure noise.
    """
    snapshot = copy.deepcopy(agent.policy.state_dict())
    gradients = []

    for i in range(n_samples):
        agent._clear()
        collect_episode(agent, env, seed=seed + 50_000 + i)
        log_probs = torch.stack(agent._log_probs)
        returns = torch.cat(agent._returns)
        advantages = agent._apply_baseline(returns)
        loss = -(log_probs * advantages).mean()

        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(agent.gradient_snapshot())
        # No optimiser step is taken, so the parameters are already unchanged;
        # restoring explicitly documents the invariant this probe depends on.
        agent.policy.load_state_dict(snapshot)

    agent.optimizer.zero_grad(set_to_none=True)
    agent._clear()

    G = np.stack(gradients)
    norms = np.linalg.norm(G, axis=1)
    unit = G / np.maximum(norms[:, None], 1e-12)
    sims = unit @ unit.T
    off_diagonal = sims[~np.eye(len(G), dtype=bool)]

    mean_grad_norm = float(np.linalg.norm(G.mean(axis=0)))
    return {
        "samples": n_samples,
        "mean_cosine_similarity": float(off_diagonal.mean()),
        "std_cosine_similarity": float(off_diagonal.std()),
        "mean_individual_norm": float(norms.mean()),
        "norm_of_mean": mean_grad_norm,
        "snr": float(mean_grad_norm / max(norms.mean(), 1e-12)),
    }


def train(args) -> tuple[ReinforceAgent, MetricsRecorder, object, object]:
    set_seed(args.seed)
    device = get_device(args.device)

    env = make_env(args.env, seed=args.seed)
    spec = env_spec(env)
    agent = build_agent(args, spec, device)
    recorder = MetricsRecorder(run_name=args.run_name, config=vars(args))
    evaluator = PeriodicEvaluator(
        make_env(args.env, seed=args.seed + 424_242),
        every=args.eval_every, episodes=args.eval_rollouts,
    )

    if not args.quiet:
        print(f"[reinforce] {args.env}  baseline={args.baseline}  "
              f"batch_episodes={args.batch_episodes}  device={device}  seed={args.seed}")

    pending: list[tuple[int, float, int]] = []
    for episode in range(args.episodes):
        rewards = collect_episode(agent, env, seed=args.seed * 1000 + episode)
        pending.append((episode, float(sum(rewards)), len(rewards)))

        if not agent.ready_to_update:
            continue

        stats = agent.update()
        eval_reward = evaluator.maybe_evaluate(agent, agent.env_steps)
        shared = {} if eval_reward is None else {"eval_reward": eval_reward}
        # One update can cover several episodes; log each with the shared stats
        # so the x-axis stays "episodes" and lines up with the other algorithms.
        for ep, ep_reward, ep_len in pending:
            recorder.record_episode(
                episode=ep,
                reward=ep_reward,
                length=ep_len,
                env_steps=agent.env_steps,
                loss=stats["loss"],
                entropy=stats["entropy"],
                grad_norm=stats["grad_norm"],
                return_std=stats["return_std"],
                **shared,
            )
        pending.clear()

        if not args.quiet and (episode + 1) % max(1, args.episodes // 12) == 0:
            avg = recorder.rewards[-args.window:].mean()
            print(
                f"  ep {episode + 1:>5}/{args.episodes}  steps={agent.env_steps:>7}  "
                f"avg{args.window}={avg:7.1f}  entropy={stats['entropy']:.3f}  "
                f"|grad|={stats['grad_norm']:.3f}  return_std={stats['return_std']:7.1f}"
            )

        if agent.env_steps >= args.max_env_steps:
            if not args.quiet:
                print(f"  reached the {args.max_env_steps} env-step budget; stopping")
            break

    return agent, recorder, env, evaluator


def main(argv=None):
    args = parse_args(argv)
    agent, recorder, env, evaluator = train(args)

    eval_env = make_env(args.env, seed=args.seed + 99_999)
    result = evaluate_agent(agent, eval_env, episodes=args.eval_episodes, seed=100_000)
    threshold = eval_env.spec.reward_threshold if eval_env.spec else None

    summary = recorder.summary(args.window)
    print("\n--- results -------------------------------------------------")
    print(f"  baseline                 {args.baseline}")
    print(f"  episodes                 {summary['episodes']}")
    print(f"  environment steps        {summary['env_steps']}")
    print(f"  policy updates           {agent.updates}")
    print(f"  training time            {summary['training_time_sec']} s "
          f"({summary['steps_per_second']} env-steps/s)")
    print(f"  best episode reward      {summary['best_episode_reward']:.1f}")
    print(f"  final performance (train){summary['final_performance_train']:8.1f} "
          f"(last {args.window} episodes)")
    print("  greedy evaluation:")
    print(format_report(args.run_name, result, threshold))

    variance = None
    if args.gradient_variance > 0:
        print(f"\n  gradient-variance probe ({args.gradient_variance} independent "
              "single-episode estimates at fixed parameters):")
        variance = measure_gradient_variance(agent, env, args.gradient_variance, args.seed)
        print(f"    mean pairwise cosine similarity  {variance['mean_cosine_similarity']:+.3f} "
              f"(1.0 = perfect agreement, 0.0 = orthogonal)")
        print(f"    ||mean grad|| / mean||grad||     {variance['snr']:.3f}  "
              "(signal-to-noise; 1.0 would mean no variance at all)")
        recorder.config["gradient_variance"] = variance

    if not args.no_save:
        path = agent.save(resolve_path("checkpoints", f"{args.run_name}.pt"))
        print(f"  checkpoint               {path}")
    # Persist the greedy evaluation alongside the per-episode log, so that
    # evaluation/report.py can rebuild the summary table without retraining.
    recorder.config["final_eval"] = {
        "mean_reward": result["mean_reward"],
        "std_reward": result["std_reward"],
        "episodes": result["episodes"],
        "threshold": threshold,
        "solved": (threshold is not None and result["mean_reward"] >= threshold),
    }
    recorder.save(resolve_dir(*LOG_DIR))

    if not args.no_plots:
        plot_learning_curve(
            recorder,
            resolve_path(*PLOT_DIR, f"{args.run_name}_curve.png"),
            window=args.window,
            title=f"REINFORCE on {args.env} (baseline={args.baseline})",
        )

    env.close()
    eval_env.close()
    evaluator.env.close()
    return agent, recorder, result


