"""Run the whole progression end to end and write a comparison table.

    python -m experiments.run_all              # canonical runs (~15 min on CPU)
    python -m experiments.run_all --quick      # smaller budgets, for a smoke test
    python -m experiments.run_all --with-ablations   # + all five experiments

Order matters: each stage is run with the *same* seed and, where the task is
shared, the same environment-step budget, so the final table compares
algorithms rather than budgets.
"""

from __future__ import annotations

import argparse
import json
import time

from evaluation.utils import LOG_DIR, PLOT_DIR, resolve_path
from experiments._common import banner


def _fmt(value, spec=".1f"):
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="short budgets for a smoke test")
    p.add_argument("--with-ablations", action="store_true", help="also run exp1-exp5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-lunar", action="store_true",
                   help="train PPO on CartPole instead of LunarLander")
    p.add_argument("--tag", type=str, default=None,
                   help="suffix appended to every run name. Defaults to '_quick' "
                        "under --quick so a smoke test cannot overwrite the "
                        "canonical logs, checkpoints and figures.")
    args = p.parse_args(argv)

    scale = 0.2 if args.quick else 1.0
    tag = args.tag if args.tag is not None else ("_quick" if args.quick else "")
    def tagged(base: str) -> str:
        """Append the run tag, so --quick never overwrites canonical results."""
        return f"{base}{tag}"
    seed = str(args.seed)
    rows: list[dict] = []
    started = time.perf_counter()

    def record(stage, algo, env, recorder, result, extra=""):
        summary = recorder.summary()
        rows.append({
            "stage": stage,
            "algorithm": algo,
            "env": env,
            "env_steps": summary["env_steps"],
            "episodes": summary["episodes"],
            "train_time_s": summary["training_time_sec"],
            "final_train_reward": summary["final_performance_train"],
            "eval_reward": result["mean_reward"] if result else None,
            "eval_std": result["std_reward"] if result else None,
            "note": extra,
        })

    # ---------------------------------------------------------------- stage 1
    banner("Stage 1/7: Tabular Q-learning", "GridWorld, hand-written MDP")
    from q_learning.train import main as q_main

    q_rec = q_main([
        "--episodes", str(int(6000 * scale)),
        "--epsilon-decay", "0.9995", "--epsilon-end", "0.1",
        "--exploring-starts", "--alpha-schedule", "visit_count",
        "--seed", seed, "--run-name", tagged("q_learning_gridworld"), "--quiet",
    ])
    rows.append({
        "stage": 1, "algorithm": "Q-learning (tabular)", "env": "GridWorld",
        "env_steps": q_rec.summary()["env_steps"],
        "episodes": q_rec.summary()["episodes"],
        "train_time_s": q_rec.summary()["training_time_sec"],
        "final_train_reward": q_rec.summary()["final_performance_train"],
        "eval_reward": None, "eval_std": None,
        "note": "converges to the exact optimum from value iteration",
    })

    # ------------------------------------------------------------ stages 2-4
    from dqn.train import main as dqn_main

    dqn_episodes = str(int(2000 * scale))
    dqn_budget = str(int(200_000 * scale))
    dqn_stages = [
        (2, "DQN (online, no replay)", ["--no-replay", "--no-target-network"],
         "dqn_online", "trains on the newest transition only"),
        (3, "DQN + replay", ["--no-target-network"],
         "dqn_replay", "uniform minibatches from a 50k buffer"),
        (4, "DQN + replay + target net", [],
         "dqn_cartpole", "frozen bootstrap target"),
    ]
    for stage, name, flags, run_name, note in dqn_stages:
        banner(f"Stage {stage}/7: {name}", "CartPole-v1")
        agent, rec, result, _ = dqn_main([
            *flags, "--episodes", dqn_episodes, "--max-env-steps", dqn_budget,
            "--seed", seed, "--run-name", tagged(run_name), "--quiet",
        ])
        record(stage, name, "CartPole-v1", rec, result, note)

    # ---------------------------------------------------------------- stage 5
    banner("Stage 5/7: REINFORCE", "CartPole-v1")
    from reinforce.train import main as reinforce_main

    _, rec, result = reinforce_main([
        "--episodes", str(int(1500 * scale)),
        "--max-env-steps", str(int(200_000 * scale)),
        "--gradient-variance", "0" if args.quick else "40",
        "--seed", seed, "--run-name", tagged("reinforce_cartpole"), "--quiet",
    ])
    record(5, "REINFORCE", "CartPole-v1", rec, result, "Monte-Carlo policy gradient")

    # ---------------------------------------------------------------- stage 6
    banner("Stage 6/7: Actor-Critic", "CartPole-v1")
    from actor_critic.train import main as ac_main

    _, rec, result = ac_main([
        "--episodes", str(int(2000 * scale)),
        "--max-env-steps", str(int(200_000 * scale)),
        "--seed", seed, "--run-name", tagged("actor_critic_cartpole"), "--quiet",
    ])
    record(6, "Actor-Critic (n-step)", "CartPole-v1", rec, result,
           "learned state-dependent baseline")

    # ---------------------------------------------------------------- stage 7
    from ppo.train import main as ppo_main

    ppo_env = "CartPole-v1" if args.skip_lunar else "LunarLander-v3"
    ppo_steps = int((150_000 if args.skip_lunar else 600_000) * scale)
    banner("Stage 7/7: PPO", ppo_env)
    _, rec, result = ppo_main([
        "--env", ppo_env, "--total-steps", str(ppo_steps),
        "--seed", seed,
        "--run-name", tagged("ppo_cartpole" if args.skip_lunar else "ppo_lunarlander"),
        "--quiet",
    ])
    record(7, "PPO", rec.config.get("env_id", ppo_env), rec, result,
           "clipped surrogate, 10 epochs per rollout")

    # PPO on CartPole too, so stages 2-7 share one task for a fair comparison.
    if not args.skip_lunar:
        banner("Stage 7/7 (control): PPO on CartPole", "same task as stages 2-6")
        _, rec_cp, result_cp = ppo_main([
            "--env", "CartPole-v1", "--total-steps", str(int(80_000 * scale)),
            "--seed", seed, "--run-name", tagged("ppo_cartpole"), "--quiet",
        ])
        record(7, "PPO (control)", "CartPole-v1", rec_cp, result_cp,
               "same task as stages 2-6")

    # ------------------------------------------------------------------ table
    banner("Progression summary", f"seed={args.seed}, total wall clock "
                                  f"{time.perf_counter() - started:.0f} s")
    header = (f"  {'#':<2} {'algorithm':<28}{'environment':<16}"
              f"{'env steps':>11}{'time (s)':>10}{'eval reward':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        eval_txt = (
            "n/a" if row["eval_reward"] is None
            else f"{row['eval_reward']:.1f} +/- {row['eval_std']:.1f}"
        )
        print(f"  {row['stage']:<2} {row['algorithm']:<28}{row['env']:<16}"
              f"{row['env_steps']:>11,}{row['train_time_s']:>10.1f}{eval_txt:>14}")

    for row in rows:
        if row["note"]:
            print(f"     {row['stage']}: {row['note']}")

    path = resolve_path(*LOG_DIR, f"progression_summary{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n  table -> {path}")

    # ---- one figure with every stage on it -------------------------------
    from evaluation.metrics import MetricsRecorder
    from evaluation.plotting import plot_comparison

    cartpole_runs = {}
    for name, run_name in [
        ("DQN online", "dqn_online"),
        ("DQN + replay", "dqn_replay"),
        ("DQN + replay + target", "dqn_cartpole"),
        ("REINFORCE", "reinforce_cartpole"),
        ("Actor-Critic", "actor_critic_cartpole"),
        ("PPO", "ppo_cartpole"),
    ]:
        csv_path = resolve_path(*LOG_DIR, f"{tagged(run_name)}.csv")
        try:
            cols = MetricsRecorder.load_csv(csv_path)
        except FileNotFoundError:
            continue
        rec = MetricsRecorder(run_name=name)
        rec.episodes = [
            {"reward": r, "env_steps": s, "length": l}
            for r, s, l in zip(cols["reward"], cols["env_steps"], cols["length"])
        ]
        cartpole_runs[name] = rec

    if len(cartpole_runs) > 1:
        plot_comparison(
            cartpole_runs,
            resolve_path(*PLOT_DIR, f"progression_cartpole{tag}.png"),
            window=50,
            title="The progression on one task: sample efficiency on CartPole-v1",
            x_axis="env_steps",
        )

    if args.with_ablations:
        from experiments import (
            exp1_exploring_starts,
            exp2_replay_ablation,
            exp3_target_network_frequency,
            exp4_reinforce_vs_actor_critic,
            exp5_ppo_clipping,
        )

        seeds = ["--seeds", "2" if args.quick else "3"]
        exp1_exploring_starts.main(seeds)
        exp2_replay_ablation.main(seeds)
        exp3_target_network_frequency.main(seeds)
        exp4_reinforce_vs_actor_critic.main(seeds)
        exp5_ppo_clipping.main(seeds)

    return rows


if __name__ == "__main__":
    main()
