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

