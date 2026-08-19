"""Rebuild the summary table and progression figure from existing logs.

    python -m evaluation.report

``experiments/run_all.py`` produces the same artefacts but retrains everything.
This script only reads ``results/logs/*.csv`` and ``*.json``, so it is the right
tool when you want to regenerate the comparison after adding one run, or to
recover the table without spending twenty minutes of CPU.

Output:
* a Markdown table on stdout, ready to paste into the README
* ``results/plots/progression_cartpole.png`` -- every stage that ran on
  CartPole, plotted against environment steps so sample efficiency is
  directly comparable
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluation.metrics import MetricsRecorder
from evaluation.plotting import plot_comparison
from evaluation.utils import LOG_DIR, PLOT_DIR, resolve_dir, resolve_path

# (run_name, display name, stage) in progression order.
KNOWN_RUNS = [
    ("q_learning_gridworld", "Q-learning (tabular)", 1),
    ("dqn_online", "DQN (no replay, no target)", 2),
    ("dqn_replay", "DQN + replay", 3),
    ("dqn_cartpole", "DQN + replay + target", 4),
    ("reinforce_cartpole", "REINFORCE", 5),
    ("actor_critic_cartpole", "Actor-Critic (n-step)", 6),
    ("ppo_cartpole", "PPO (CartPole control)", 7),
    ("ppo_lunarlander", "PPO", 7),
]

CARTPOLE_CURVES = [
    ("DQN (no replay, no target)", "dqn_online"),
    ("DQN + replay", "dqn_replay"),
    ("DQN + replay + target", "dqn_cartpole"),
    ("REINFORCE", "reinforce_cartpole"),
    ("Actor-Critic", "actor_critic_cartpole"),
    ("PPO", "ppo_cartpole"),
]


def load_run(run_name: str) -> tuple[dict, dict] | None:
    """Return ``(summary_json, csv_columns)`` for a run, or None if absent."""
    log_dir = resolve_dir(*LOG_DIR)
    json_path = os.path.join(log_dir, f"{run_name}.json")
    csv_path = os.path.join(log_dir, f"{run_name}.csv")
    if not (os.path.exists(json_path) and os.path.exists(csv_path)):
        return None
    with open(json_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    return summary, MetricsRecorder.load_csv(csv_path)


def recorder_from_csv(name: str, cols: dict) -> MetricsRecorder:
    """Wrap raw CSV columns back into a recorder for plotting."""
    rec = MetricsRecorder(run_name=name)
    rec.episodes = [
        {"reward": r, "env_steps": s, "length": l}
        for r, s, l in zip(cols["reward"], cols["env_steps"], cols["length"])
    ]
    return rec


def build_table(rows: list[dict]) -> str:
    header = (
        "| # | algorithm | environment | env steps | episodes | train time | "
        "final greedy eval |\n"
        "|---|---|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        eval_txt = row["eval"] or "n/a"
        lines.append(
            f"| {row['stage']} | {row['name']} | {row['env']} | "
            f"{row['env_steps']:,} | {row['episodes']:,} | "
            f"{row['time']:.0f} s | {eval_txt} |"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args(argv)

    rows = []
    for run_name, display, stage in KNOWN_RUNS:
        loaded = load_run(run_name)
        if loaded is None:
            print(f"  (skipping {run_name}: no logs found)")
            continue
        summary, cols = loaded
        config = summary.get("config", {})
        env = config.get("env_id") or config.get("env") or "GridWorld"
        # Preferred source: the final greedy evaluation the trainer stored.
        # Falls back to the last periodic evaluation for older logs.
        final_eval = config.get("final_eval") or {}
        eval_txt = None
        if "mean_reward" in final_eval:
            eval_txt = f"{final_eval['mean_reward']:.1f} +/- {final_eval['std_reward']:.1f}"
            if final_eval.get("solved"):
                eval_txt += " (solved)"
        elif "final_q_star_gap" in final_eval:
            eval_txt = (f"{final_eval['optimal_actions']}/{final_eval['decision_states']}"
                        " optimal actions")
        elif "eval_reward" in cols and np.isfinite(cols["eval_reward"]).any():
            finite = cols["eval_reward"][np.isfinite(cols["eval_reward"])]
            eval_txt = f"{finite[-1]:.1f}"
        rows.append({
            "stage": stage,
            "name": display,
            "env": env,
            "env_steps": int(summary["env_steps"]),
            "episodes": int(summary["episodes"]),
            "time": float(summary["training_time_sec"]),
            "eval": eval_txt,
            "final_train": summary["final_performance_train"],
        })

    print("\n" + build_table(rows) + "\n")

    curves = {}
    for display, run_name in CARTPOLE_CURVES:
        loaded = load_run(run_name)
        if loaded is not None:
            curves[display] = recorder_from_csv(display, loaded[1])

    if len(curves) > 1:
        plot_comparison(
            curves,
            resolve_path(*PLOT_DIR, "progression_cartpole_training.png"),
            window=args.window,
            title="CartPole-v1: training reward (includes exploration noise)",
            x_axis="env_steps",
        )
    else:
        print("  (not enough CartPole runs for the progression figure)")

    plot_greedy_progression(args)
    return rows


def plot_greedy_progression(args) -> None:
    """Plot the periodic *greedy* evaluations -- the honest comparison.

    Training reward penalises algorithms differently: DQN acts with
    ``epsilon = 0.05``, and on CartPole a single random action can end an
    episode, so its training curve oscillates violently while its greedy policy
    is perfect.  Comparing algorithms on training reward would mostly compare
    their exploration schemes.
    """
    import matplotlib.pyplot as plt

    from evaluation.plotting import PALETTE, save_figure

    from evaluation.metrics import moving_average

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = 0
    for i, (display, run_name) in enumerate(CARTPOLE_CURVES):
        loaded = load_run(run_name)
        if loaded is None:
            continue
        _, cols = loaded
        if "eval_reward" not in cols:
            continue
        mask = np.isfinite(cols["eval_reward"])
        if not mask.any():
            continue
        steps, rewards = cols["env_steps"][mask], cols["eval_reward"][mask]
        colour = PALETTE[i % len(PALETTE)]
        # Faint raw points, bold 3-point average. Each evaluation is only 5
        # episodes, so a single point carries real sampling noise on top of the
        # genuine oscillation -- showing both keeps the figure honest and legible.
        ax.plot(steps, rewards, color=colour, alpha=0.25, linewidth=0.9,
                marker="o", markersize=2.5)
        ax.plot(steps, moving_average(rewards, 3), color=colour,
                linewidth=2.2, label=display)
        plotted += 1

    if plotted < 2:
        plt.close(fig)
        print("  (not enough runs with periodic evaluation for the greedy figure)")
        return

    ax.axhline(475, color="#555", linestyle="--", linewidth=1.2,
               label="CartPole solved threshold (475)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Greedy evaluation reward")
    ax.set_ylim(-15, 545)
    ax.set_title("The progression on one task: greedy policy vs samples consumed")
    ax.grid(alpha=0.25)
    # Outside the axes: at 500 reward the curves cover every in-plot corner.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    save_figure(fig, resolve_path(*PLOT_DIR, "progression_cartpole.png"))


if __name__ == "__main__":
    main()
