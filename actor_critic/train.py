"""Train n-step Actor-Critic on CartPole.

    python -m actor_critic.train --episodes 800 --n-steps 8

Structurally this loop differs from :mod:`reinforce.train` in one important
way: ``agent.update`` fires whenever ``n_steps`` transitions have accumulated,
*not* at episode boundaries.  That is the practical consequence of replacing
the Monte-Carlo return with a bootstrapped advantage.
"""

from __future__ import annotations

import argparse

from actor_critic.agent import ActorCriticAgent
from environments.registry import env_spec, make_env
from evaluation.evaluate import PeriodicEvaluator, evaluate_agent, format_report
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_learning_curve
from evaluation.utils import LOG_DIR, PLOT_DIR, get_device, resolve_dir, resolve_path, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advantage Actor-Critic")
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-env-steps", type=int, default=200_000)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=32,
                   help="rollout length; 1 = pure TD, large = closer to REINFORCE")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.005)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--no-advantage-norm", action="store_true")
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
    p.add_argument("--run-name", type=str, default="actor_critic_cartpole")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_agent(args, spec, device) -> ActorCriticAgent:
    return ActorCriticAgent(
        obs_dim=spec.obs_dim,
        n_actions=spec.n_actions,
        hidden_sizes=args.hidden,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        n_steps=args.n_steps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        normalize_advantages=not args.no_advantage_norm,
        device=device,
        seed=args.seed,
    )


def train(args) -> tuple[ActorCriticAgent, MetricsRecorder, object, object]:
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
        print(f"[actor-critic] {args.env}  n_steps={args.n_steps}  "
              f"device={device}  seed={args.seed}")

    last_stats = {"loss": float("nan"), "entropy": float("nan"),
                  "value_loss": float("nan"), "mean_value": float("nan")}

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed * 1000 + episode)
        ep_reward, ep_len = 0.0, 0

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store(state, action, float(reward), terminated)

            state = next_state
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

            # Update on a full rollout *or* at the end of an episode, so the
            # final partial rollout is not discarded.
            if agent.ready_to_update or done:
                last_stats = agent.update(last_state=state, last_done=terminated)
            if done:
                break

        eval_reward = evaluator.maybe_evaluate(agent, agent.env_steps)
        extra = {} if eval_reward is None else {"eval_reward": eval_reward}
        recorder.record_episode(
            episode=episode,
            reward=ep_reward,
            length=ep_len,
            env_steps=agent.env_steps,
            loss=last_stats["loss"],
            entropy=last_stats["entropy"],
            value_loss=last_stats["value_loss"],
            mean_value=last_stats["mean_value"],
            updates=agent.updates,
            **extra,
        )

        if not args.quiet and (episode + 1) % max(1, args.episodes // 12) == 0:
            avg = recorder.rewards[-args.window:].mean()
            print(
                f"  ep {episode + 1:>4}/{args.episodes}  steps={agent.env_steps:>7}  "
                f"avg{args.window}={avg:7.1f}  entropy={last_stats['entropy']:.3f}  "
                f"v_loss={last_stats['value_loss']:8.3f}  "
                f"meanV={last_stats['mean_value']:7.2f}  updates={agent.updates}"
            )

        if agent.env_steps >= args.max_env_steps:
            if not args.quiet:
                print(f"  reached the {args.max_env_steps} env-step budget; stopping")
            break

    return agent, recorder, env, evaluator


