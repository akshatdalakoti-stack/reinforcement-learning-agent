"""Train tabular Q-learning on GridWorld.

    python -m q_learning.train --episodes 2000 --seed 0

Besides the learning curve this script produces the two artefacts that make the
tabular case worth studying: a picture of the greedy policy, and a comparison
against the exact optimum from value iteration.  Being able to say "the learned
Q is within 0.01 of Q*" is a luxury we lose the moment the table becomes a
neural network, so it is worth collecting while we can.
"""

from __future__ import annotations

import argparse

import numpy as np

from environments.gridworld import GridWorld, value_iteration
from evaluation.metrics import MetricsRecorder
from evaluation.plotting import (
    plot_gridworld_policy,
    plot_learning_curve,
    plot_q_convergence,
)
from evaluation.utils import LOG_DIR, PLOT_DIR, resolve_dir, resolve_path, set_seed
from q_learning.agent import QLearningAgent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular Q-learning on GridWorld")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--alpha-schedule", choices=["constant", "visit_count"],
                   default="constant",
                   help="'visit_count' uses 1/n(s,a)**p, satisfying the "
                        "Robbins-Monro conditions for convergence")
    p.add_argument("--alpha-power", type=float, default=0.75)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--slip-prob", type=float, default=0.1,
                   help="0 makes the grid deterministic")
    p.add_argument("--exploring-starts", action="store_true",
                   help="begin each episode in a random cell (fixes coverage of "
                        "states the greedy policy never reaches)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=50, help="moving-average window")
    p.add_argument("--run-name", type=str, default="q_learning_gridworld")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def train(args: argparse.Namespace):
    set_seed(args.seed)

    env = GridWorld(slip_prob=args.slip_prob, max_steps=args.max_steps)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        alpha_schedule=args.alpha_schedule,
        alpha_power=args.alpha_power,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    recorder = MetricsRecorder(run_name=args.run_name, config=vars(args))
    q_star, _ = value_iteration(env, gamma=args.gamma)
    decision_states = env.decision_states()
    reset_options = {"exploring_start": args.exploring_starts}
    deltas: list[float] = []
    q_star_gaps: list[float] = []

    env_steps = 0
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode, options=reset_options)
        agent.start_episode()
        ep_reward, ep_len, td_abs = 0.0, 0, []

        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Note `terminated`, not `terminated or truncated`: a time-limit
            # cutoff must not be learned as "the world ends here".
            td_error = agent.update(state, action, reward, next_state, terminated)

            td_abs.append(abs(td_error))
            ep_reward += reward
            ep_len += 1
            env_steps += 1
            state = next_state
            if terminated or truncated:
                break

        agent.decay_epsilon()
        deltas.append(max(agent.last_episode_max_delta, 1e-12))

        # Distance to the exact solution over the states a policy is defined on.
        gap = float(np.abs(agent.q_table[decision_states] - q_star[decision_states]).max())
        q_star_gaps.append(gap)

        recorder.record_episode(
            episode=episode,
            reward=ep_reward,
            length=ep_len,
            env_steps=env_steps,
            loss=float(np.mean(np.square(td_abs))),   # mean squared TD error
            epsilon=agent.epsilon,
            q_star_gap=gap,
            coverage=agent.coverage(decision_states),
        )

        if not args.quiet and (episode + 1) % max(1, args.episodes // 10) == 0:
            print(
                f"  ep {episode + 1:>5}/{args.episodes}  "
                f"avg100={recorder.rewards[-100:].mean():+.3f}  "
                f"eps={agent.epsilon:.3f}  "
                f"|Q-Q*|max={gap:.4f}  coverage={agent.coverage(decision_states):.0%}"
            )

    return env, agent, recorder, q_star, deltas, q_star_gaps


def main(argv=None) -> MetricsRecorder:
    args = parse_args(argv)
    print(f"[q-learning] GridWorld  slip={args.slip_prob}  seed={args.seed}")
    env, agent, recorder, q_star, deltas, gaps = train(args)

    ckpt = agent.save(resolve_path("checkpoints", f"{args.run_name}.npz"))

    decision_states = env.decision_states()
    greedy = agent.greedy_policy()
    # Compare against the *set* of optimal actions: with a stochastic grid two
    # actions can be tied at the optimum, and calling one of them wrong would
    # understate the policy quality.
    greedy_matches = sum(
        1 for s in decision_states
        if q_star[s, greedy[s]] >= q_star[s].max() - 1e-9
    )
    n_decision_states = len(decision_states)

    summary = recorder.summary(args.window)
    print("\n--- results -------------------------------------------------")
    print(f"  episodes                 {summary['episodes']}")
    print(f"  environment steps        {summary['env_steps']}")
    print(f"  training time            {summary['training_time_sec']} s")
    print(f"  final performance        {summary['final_performance_train']:+.3f} "
          f"(mean reward, last {args.window} episodes)")
    print(f"  final |Q - Q*|_max       {gaps[-1]:.4f}")
    print(f"  greedy policy matches Q* {greedy_matches}/{n_decision_states} states")
    print(f"  state-action coverage    {agent.coverage(decision_states):.0%}")
    print(f"  checkpoint               {ckpt}")

    recorder.config["final_eval"] = {
        "final_q_star_gap": gaps[-1],
        "optimal_actions": greedy_matches,
        "decision_states": n_decision_states,
        "coverage": agent.coverage(decision_states),
    }
    recorder.save(resolve_dir(*LOG_DIR))

    if not args.no_plots:
        plot_learning_curve(
            recorder,
            resolve_path(*PLOT_DIR, f"{args.run_name}_curve.png"),
            window=args.window,
            title="Q-learning on GridWorld",
        )
        plot_q_convergence(
            deltas,
            resolve_path(*PLOT_DIR, f"{args.run_name}_convergence.png"),
            title="Q-learning convergence: largest table update per episode",
        )
        plot_gridworld_policy(
            env,
            agent.q_table,
            resolve_path(*PLOT_DIR, f"{args.run_name}_policy.png"),
            title="Learned greedy policy (arrows) and V(s) (shading)",
        )
        plot_gridworld_policy(
            env,
            q_star,
            resolve_path(*PLOT_DIR, f"{args.run_name}_policy_optimal.png"),
            title="Optimal policy from value iteration (ground truth)",
        )

    print("\n  greedy rollout from the learned table:")
    state, _ = env.reset(seed=args.seed)
    for _ in range(args.max_steps):
        print("    " + env.render().replace("\n", "\n    "))
        state, _, terminated, truncated, _ = env.step(agent.greedy_action(state))
        if terminated or truncated:
            print("    " + env.render().replace("\n", "\n    "))
            break

    return recorder


