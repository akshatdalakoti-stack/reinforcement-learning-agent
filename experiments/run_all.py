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


