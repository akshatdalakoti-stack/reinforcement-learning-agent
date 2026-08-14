"""Matplotlib helpers for learning curves and ablation comparisons."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # scripts run headless; figures go to results/plots/
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.metrics import MetricsRecorder, moving_average  # noqa: E402

PALETTE = ["#2f6f9f", "#c1543a", "#4a8a5c", "#8a6bb1", "#c9922e", "#5b5f66"]


