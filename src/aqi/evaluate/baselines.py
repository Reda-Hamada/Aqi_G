"""Naive baselines: persistence, climatology, seasonal naive.

All baselines act directly on the tabular DataFrame and produce predictions
in the same (N, P*H) shape as a Forecaster.predict().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import HORIZONS, POLLUTANTS


def persistence_predict(X: pd.DataFrame) -> np.ndarray:
    """ŷ_{t+h} = x_t for each pollutant, broadcast across horizons."""
    preds = np.zeros((len(X), len(POLLUTANTS) * len(HORIZONS)), dtype="float64")
    col = 0
    for p in POLLUTANTS:
        cur = X[p].to_numpy() if p in X.columns else X.get(f"{p}_lag1", np.zeros(len(X))).to_numpy()
        for _ in HORIZONS:
            preds[:, col] = cur
            col += 1
    return preds


def climatology_predict(X: pd.DataFrame, train_df: pd.DataFrame) -> np.ndarray:
    """ŷ_{t+h} = mean(p) at (station, month, hour-of-day+h) from train_df."""
    out = np.zeros((len(X), len(POLLUTANTS) * len(HORIZONS)), dtype="float64")
    clims = {}
    for p in POLLUTANTS:
        tmp = train_df[["station", "timestamp", p]].copy()
        tmp["month"] = tmp["timestamp"].dt.month
        tmp["hour"] = tmp["timestamp"].dt.hour
        clims[p] = tmp.groupby(["station", "month", "hour"])[p].mean()

    stations = X["station"].to_numpy()
    months = X["timestamp"].dt.month.to_numpy()
    hours = X["timestamp"].dt.hour.to_numpy()

    col = 0
    for p in POLLUTANTS:
        for h in HORIZONS:
            target_hour = (hours + h) % 24
            keys = pd.MultiIndex.from_arrays([stations, months, target_hour])
            out[:, col] = clims[p].reindex(keys).to_numpy()
            col += 1
    return out


def seasonal_naive_predict(X: pd.DataFrame) -> np.ndarray:
    """ŷ_{t+h} = value at (t+h-168). Requires `<p>_lag168` to be present."""
    preds = np.zeros((len(X), len(POLLUTANTS) * len(HORIZONS)), dtype="float64")
    col = 0
    for p in POLLUTANTS:
        lag168 = X.get(f"{p}_lag168", X[p]).to_numpy() if f"{p}_lag168" in X.columns or p in X.columns else np.zeros(len(X))
        for _ in HORIZONS:
            preds[:, col] = lag168
            col += 1
    return preds
