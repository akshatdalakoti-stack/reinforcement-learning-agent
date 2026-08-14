"""Train PPO on LunarLander (or any discrete Gymnasium environment).

    python -m ppo.train --env LunarLander-v3 --total-steps 600000
    python -m ppo.train --env CartPole-v1  --total-steps 150000

The loop shape changes here for the last time.  Q-learning updated per step,
REINFORCE per episode, A2C per n steps -- PPO collects a *fixed-size rollout*
that deliberately ignores episode boundaries, then optimises it repeatedly.
Progress is therefore measured in environment steps, not episodes: an episode
is no longer the unit of anything.

If Box2D is unavailable the script falls back to a classic-control environment
rather than failing, and says so.
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.registry import env_available, env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed
from ppo.agent import PPOAgent

FALLBACK_ENV = "CartPole-v1"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO on a discrete Gymnasium environment")
    p.add_argument("--env", type=str, default="LunarLander-v3")
    p.add_argument("--total-steps", type=int, default=600_000)
    p.add_argument("--rollout-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None,
                   help="early-stop a update once approx KL exceeds this")
    p.add_argument("--no-value-clip", action="store_true")
    p.add_argument("--no-advantage-norm", action="store_true")
    p.add_argument("--no-lr-anneal", action="store_true")
    p.add_argument("--separate-networks", action="store_true",
                   help="use two towers instead of a shared trunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5_000,
                   help="env steps between greedy evaluations during training; "
                        "0 disables. Training reward reflects the exploring "
                        "policy, so this is the honest progress signal.")
    p.add_argument("--eval-rollouts", type=int, default=5,
                   help="episodes per periodic evaluation")
    p.add_argument("--run-name", type=str, default="ppo_lunarlander")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def resolve_env_id(requested: str, quiet: bool = False) -> str:
    if env_available(requested):
        return requested
    if not quiet:
        print(f"  ! {requested} is unavailable (Box2D missing?); "
              f"falling back to {FALLBACK_ENV}")
    return FALLBACK_ENV


def build_agent(args, spec, device) -> PPOAgent:
    return PPOAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        rollout_size=args.rollout_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        normalize_advantages=not args.no_advantage_norm,
        clip_value_loss=not args.no_value_clip,
        target_kl=args.target_kl,
        anneal_lr=not args.no_lr_anneal,
        shared_trunk=not args.separate_networks,
        device=device,
        seed=args.seed,
    )


def train(args) -> tuple[PPOAgent, MetricsRecorder, object, str, object]:
    set_seed(args.seed)
    device = get_device(args.device)
    env_id = resolve_env_id(args.env, args.quiet)

    env = make_env(env_id, seed=args.seed)
    spec = env_spec(env)
    agent = build_agent(args, spec, device)
    recorder = MetricsRecorder(run_name=args.run_name, config={**vars(args), "env_id": env_id})
    evaluator = PeriodicEvaluator(
        make_env(env_id, seed=args.seed + 424_242),
        every=args.eval_every, episodes=args.eval_rollouts,
    )
    last_eval: float | None = None

    if not args.quiet:
        print(f"[ppo] {env_id}  rollout={args.rollout_size}  epochs={args.n_epochs}  "
              f"clip={args.clip_epsilon}  device={device}  seed={args.seed}")

    state, _ = env.reset(seed=args.seed)
    ep_reward, ep_len, episode = 0.0, 0, 0
    last_stats: dict[str, float] = {}
    # The episode is no longer the update boundary, so an episode that ends
    # mid-rollout is logged as it finishes, using the most recent update stats.

    # The true successor of the last stored transition, and whether it was
    # terminal.  These must be tracked separately from `state`: when a rollout
    # boundary happens to land exactly on an episode end, `state` has already
    # been reset to a *new* episode's first observation, and bootstrapping
    # V(s) from that would splice two unrelated episodes together.
    boundary_state, boundary_done = state, False

    while agent.env_steps < args.total_steps:
        # ---- collect a fixed-size rollout, crossing episode boundaries ------
        while not agent.rollout_full:
            action, log_prob, value = agent.act_with_value(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # `terminated` only: truncation must not cut the bootstrap chain,
            # or the agent learns that surviving to the time limit is worthless.
            agent.store(state, action, float(reward), terminated, log_prob, value)
            boundary_state, boundary_done = next_state, terminated

            state = next_state
            ep_reward += float(reward)
            ep_len += 1

            if terminated or truncated:
                recorder.record_episode(
                    episode=episode,
                    reward=ep_reward,
                    length=ep_len,
                    env_steps=agent.env_steps,
                    loss=last_stats.get("loss"),
                    entropy=last_stats.get("entropy"),
                    value_loss=last_stats.get("value_loss", float("nan")),
                    clip_fraction=last_stats.get("clip_fraction", float("nan")),
                    approx_kl=last_stats.get("approx_kl", float("nan")),
                    **({} if last_eval is None else {"eval_reward": last_eval}),
                )
                last_eval = None
                episode += 1
                ep_reward, ep_len = 0.0, 0
                state, _ = env.reset()

        # ---- optimise it several times --------------------------------------
        progress = agent.env_steps / max(1, args.total_steps)
        lr = agent.set_lr(progress)
        # Usually the rollout was cut by the size limit rather than by the
        # environment, so `boundary_done` is False and GAE bootstraps through
        # the cut with V(boundary_state) -- which is the whole point of letting
        # rollouts cross episode boundaries. When the cut *does* coincide with a
        # real termination, `boundary_done` is True and the bootstrap is zeroed.
        last_stats = agent.update(last_state=boundary_state, last_done=boundary_done)
        # Evaluate between rollouts: the greedy policy is what we actually ship,
        # and it is not what the training reward above measures.
        last_eval = evaluator.maybe_evaluate(agent, agent.env_steps)

        if not args.quiet and agent.updates % 10 == 0:
            avg = recorder.rewards[-args.window:].mean() if recorder.episodes else float("nan")
            print(
                f"  upd {agent.updates:>4}  steps={agent.env_steps:>7}  "
                f"avg{args.window}={avg:8.1f}  ent={last_stats['entropy']:.3f}  "
                f"clip_frac={last_stats['clip_fraction']:.3f}  "
                f"kl={last_stats['approx_kl']:.4f}  v_loss={last_stats['value_loss']:8.2f}  "
                f"lr={lr:.2e}"
            )

    return agent, recorder, env, env_id, evaluator


def main(argv=None):
    args = parse_args(argv)
    agent, recorder, env, env_id, evaluator = train(args)

    eval_env = make_env(env_id, seed=args.seed + 99_999)
    result = evaluate_agent(agent, eval_env, episodes=args.eval_episodes, seed=100_000)
    threshold = eval_env.spec.reward_threshold if eval_env.spec else None

    summary = recorder.summary(args.window)
    clip_fracs = recorder.series("clip_fraction")
    print("\n--- results -------------------------------------------------")
    print(f"  environment              {env_id}")
    print(f"  episodes                 {summary['episodes']}")
    print(f"  environment steps        {summary['env_steps']}")
    print(f"  PPO updates              {agent.updates} "
          f"({agent.updates * args.n_epochs} epochs of minibatch SGD)")
    print(f"  training time            {summary['training_time_sec']} s "
          f"({summary['steps_per_second']} env-steps/s)")
    print(f"  best episode reward      {summary['best_episode_reward']:.1f}")
    print(f"  final performance (train){summary['final_performance_train']:8.1f} "
          f"(last {args.window} episodes)")
    if np.isfinite(clip_fracs).any():
        print(f"  mean clip fraction       {np.nanmean(clip_fracs):.3f} "
              "(share of samples hitting the trust-region boundary)")
    print("  greedy evaluation:")
    print(format_report(args.run_name, result, threshold))

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
            title=f"PPO on {env_id} (clip={args.clip_epsilon}, epochs={args.n_epochs})",
        )

    env.close()
    eval_env.close()
    evaluator.env.close()
    return agent, recorder, result


if __name__ == "__main__":
    main()
