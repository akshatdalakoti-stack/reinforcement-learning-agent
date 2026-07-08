"""Fast correctness checks for the hand-written components.

    python tests_smoke.py

Not a substitute for the training runs. These target the places where a silent
bug would be invisible in a learning curve -- a wrong discount, a target network
that is secretly being trained, a GAE recursion that ignores episode
boundaries. Every check below tests something this project implements by hand
rather than something PyTorch or Gymnasium already guarantees.
"""

from __future__ import annotations

import numpy as np
import torch

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


