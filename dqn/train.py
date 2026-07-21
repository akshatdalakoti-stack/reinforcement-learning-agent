"""Train DQN on CartPole -- the driver for stages 2, 3 and 4.

    # stage 2: plain online DQN, no replay, no target net
    python -m dqn.train --no-replay --no-target-network --run-name dqn_online

    # stage 3: + experience replay
    python -m dqn.train --no-target-network --run-name dqn_replay

    # stage 4: the full agent
    python -m dqn.train --run-name dqn_full

The three configurations share every other hyperparameter and the same seed, so
differences in the learning curves are attributable to the ablated component.
"""

from __future__ import annotations

import argparse

from dqn.agent import DQNAgent
from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN on a Gymnasium environment")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-env-steps", type=int, default=200_000,
                   help="hard budget so ablations get equal sample allowance")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=4,
                   help="env steps between gradient steps; >1 avoids "
                        "overfitting the replay buffer")
    p.add_argument("--target-update-freq", type=int, default=250,
                   help="in GRADIENT steps, not env steps")
    p.add_argument("--tau", type=float, default=0.0,
                   help="> 0 switches to Polyak soft target updates")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    p.add_argument("--grad-clip", type=float, default=10.0)

    p.add_argument("--no-replay", action="store_true",
                   help="stage 2: train on the newest transition only")
    p.add_argument("--no-target-network", action="store_true",
                   help="bootstrap from the online network itself")
    p.add_argument("--double-dqn", action="store_true")

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
    p.add_argument("--run-name", type=str, default="dqn_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_agent(args: argparse.Namespace, spec, device) -> DQNAgent:
    return DQNAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_freq=args.target_update_freq,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        grad_clip=args.grad_clip,
        use_replay=not args.no_replay,
        use_target_network=not args.no_target_network,
        double_dqn=args.double_dqn,
        device=device,
        seed=args.seed,
    )


def train(args: argparse.Namespace) -> tuple[DQNAgent, MetricsRecorder, object, object]:
    set_seed(args.seed)
    device = get_device(args.device)

    env = make_env(args.env, seed=args.seed)
    spec = env_spec(env)
    agent = build_agent(args, spec, device)
    recorder = MetricsRecorder(run_name=args.run_name, config=vars(args))
    evaluator = PeriodicEvaluator(
        make_env(args.env, seed=args.seed + 424_242),
        every=args.eval_every,
        episodes=args.eval_rollouts,
    )

    variant = (
        f"replay={'on' if agent.use_replay else 'OFF'} "
        f"target_net={'on' if agent.use_target_network else 'OFF'}"
    )
    if not args.quiet:
        print(f"[dqn] {args.env}  {variant}  device={device}  seed={args.seed}")

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed * 1000 + episode)
        ep_reward, ep_len = 0.0, 0

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # `terminated` only: CartPole's 500-step cutoff is a truncation, and
            # storing it as done=True would teach the agent that balancing
            # successfully for 500 steps is worth as little as falling over.
            agent.observe(state, action, float(reward), next_state, terminated)
            agent.train_step()

            state = next_state
            ep_reward += float(reward)
            ep_len += 1
            if terminated or truncated:
                break

        stats = agent.pop_stats()
        eval_reward = evaluator.maybe_evaluate(agent, agent.env_steps)
        extra = {} if eval_reward is None else {"eval_reward": eval_reward}
        recorder.record_episode(
            episode=episode,
            reward=ep_reward,
            length=ep_len,
            env_steps=agent.env_steps,
            loss=stats["loss"],
            epsilon=agent.epsilon,
            mean_q=stats["mean_q"],
            gradient_steps=agent.gradient_steps,
            **extra,
        )

        if not args.quiet and (episode + 1) % max(1, args.episodes // 12) == 0:
            avg = recorder.rewards[-args.window:].mean()
            print(
                f"  ep {episode + 1:>4}/{args.episodes}  steps={agent.env_steps:>7}  "
                f"avg{args.window}={avg:7.1f}  eps={agent.epsilon:.3f}  "
                f"loss={stats['loss']:.4f}  meanQ={stats['mean_q']:7.2f}  "
                f"greedy={evaluator.final:7.1f}"
            )

        if agent.env_steps >= args.max_env_steps:
            if not args.quiet:
                print(f"  reached the {args.max_env_steps} env-step budget; stopping")
            break

    return agent, recorder, env, evaluator


def main(argv=None) -> tuple[DQNAgent, MetricsRecorder, dict, object]:
    args = parse_args(argv)
    agent, recorder, env, evaluator = train(args)

    eval_env = make_env(args.env, seed=args.seed + 99_999)
    result = evaluate_agent(agent, eval_env, episodes=args.eval_episodes, seed=100_000)
    threshold = eval_env.spec.reward_threshold if eval_env.spec else None

    summary = recorder.summary(args.window)
    print("\n--- results -------------------------------------------------")
    print(f"  variant                  replay={agent.use_replay} "
          f"target_net={agent.use_target_network} double={agent.double_dqn}")
    print(f"  episodes                 {summary['episodes']}")
    print(f"  environment steps        {summary['env_steps']}")
    print(f"  gradient steps           {agent.gradient_steps}")
    print(f"  target syncs             {agent.target_syncs}")
    print(f"  training time            {summary['training_time_sec']} s "
          f"({summary['steps_per_second']} env-steps/s)")
    print(f"  best episode reward      {summary['best_episode_reward']:.1f}")
    print(f"  final performance (train){summary['final_performance_train']:8.1f} "
          f"(last {args.window} episodes, still exploring)")
    if evaluator.history:
        best_at = next(s for s, r in evaluator.history if r == evaluator.best)
        print(f"  best greedy in training  {evaluator.best:8.1f} "
              f"(at {best_at:,} env steps)")
        print(f"  last greedy in training  {evaluator.final:8.1f}")
    print("  greedy evaluation (final):")
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
            title=f"DQN on {args.env} "
                  f"(replay={'on' if agent.use_replay else 'off'}, "
                  f"target net={'on' if agent.use_target_network else 'off'})",
        )

    env.close()
    eval_env.close()
    evaluator.env.close()
    return agent, recorder, result, evaluator


if __name__ == "__main__":
    main()
