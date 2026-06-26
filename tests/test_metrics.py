"""Metrics behaviour: NaN handling and known reference values."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.evaluate.metrics import category_accuracy, mae, rmse


def test_mae_ignores_nan():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 3.0, 5.0, 4.0])
    # |1-1| + |2-3| + |4-4| over 3 valid pairs
    assert mae(y, p) == 1 / 3


def test_rmse_simple():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert rmse(y, p) == 1.0


def test_category_accuracy_ignores_unknown():
    y = pd.Series(["Good", "Moderate", "Unknown", "Unhealthy"])
    p = pd.Series(["Good", "Good",     "Good",    "Unhealthy"])
    # 2 correct out of 3 valid pairs
    assert category_accuracy(y, p) == 2 / 3
