"""Controlled experiments backing each step of the progression.

Each script isolates one claim, runs the comparison across several seeds, writes
a figure to ``results/plots/`` and a JSON summary to ``results/logs/``.

    exp1_exploring_starts.py          why coverage, not the update rule, limits
                                      tabular Q-learning
    exp2_replay_ablation.py           what experience replay actually buys
    exp3_target_network_frequency.py  how target-update frequency trades speed
                                      against stability
    exp4_reinforce_vs_actor_critic.py variance: Monte-Carlo vs bootstrapped
    exp5_ppo_clipping.py              what clipping prevents
    run_all.py                        the full progression end to end
"""
