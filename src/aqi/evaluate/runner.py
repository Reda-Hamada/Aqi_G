"""Evaluation harness: run a set of models over a held-out period and produce
a comparison report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from aqi.aqi import breakpoints as bp
from aqi.aqi.compute import aqi_from_pollutants
from aqi.config import HORIZONS, POLLUTANTS
from aqi.evaluate.metrics import (
    category_accuracy,
    category_macro_f1,
    mae,
    mape,
    rmse,
)


@dataclass
class EvalResult:
    model: str
    overall: dict
    per_horizon: pd.DataFrame
    per_pollutant: pd.DataFrame
    per_station: pd.DataFrame
    confusion: pd.DataFrame


def _aqi_from_predictions(y_pred: np.ndarray, station_index: np.ndarray) -> pd.DataFrame:
    """Compute AQI/category/dominant for each (row, horizon) cell.

    `y_pred` shape: (N, P*H). Columns are ordered by POLLUTANTS x HORIZONS.
    Returns a long DataFrame with columns: row_idx, horizon, aqi, category,
    dominant.
    """
    n = y_pred.shape[0]
    h_len = len(HORIZONS)
    rows = []
    for i in range(n):
        for j, h in enumerate(HORIZONS):
            vals = {p: y_pred[i, k * h_len + j] for k, p in enumerate(POLLUTANTS)}
            aqi, cat, dom = aqi_from_pollutants(
                vals["PM2.5"], vals["PM10"], vals["SO2"],
                vals["NO2"], vals["CO"], vals["O3"],
            )
            rows.append((i, h, aqi, cat, dom))
    return pd.DataFrame(rows, columns=["row_idx", "horizon", "aqi", "category", "dominant"])


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    station: np.ndarray,
    timestamp: np.ndarray,
    aqi_true_per_horizon: dict[int, np.ndarray] | None = None,
    aqi_cat_true_per_horizon: dict[int, np.ndarray] | None = None,
) -> EvalResult:
    """Score one model's predictions across pollutants, horizons, stations."""
    h_len = len(HORIZONS)
    p_len = len(POLLUTANTS)
    assert y_true.shape == y_pred.shape == (len(station), p_len * h_len), \
        f"y shapes {y_true.shape} / {y_pred.shape} vs expected (N, {p_len * h_len})"

    rows = []
    for k, p in enumerate(POLLUTANTS):
        for j, h in enumerate(HORIZONS):
            c = k * h_len + j
            rows.append(dict(
                pollutant=p, horizon=h,
                mae=mae(y_true[:, c], y_pred[:, c]),
                rmse=rmse(y_true[:, c], y_pred[:, c]),
                mape=mape(y_true[:, c], y_pred[:, c]),
            ))
    long = pd.DataFrame(rows)
    per_horizon = long.groupby("horizon")[["mae", "rmse", "mape"]].mean().reset_index()
    per_pollutant = long.groupby("pollutant")[["mae", "rmse", "mape"]].mean().reset_index()

    per_station_rows = []
    for s in pd.unique(station):
        m = station == s
        if not m.any():
            continue
        per_station_rows.append(dict(
            station=s,
            mae=mae(y_true[m].ravel(), y_pred[m].ravel()),
            rmse=rmse(y_true[m].ravel(), y_pred[m].ravel()),
        ))
    per_station = pd.DataFrame(per_station_rows)

    overall = dict(
        mae=mae(y_true.ravel(), y_pred.ravel()),
        rmse=rmse(y_true.ravel(), y_pred.ravel()),
        mape=mape(y_true.ravel(), y_pred.ravel()),
    )

    # AQI-level metrics if labels supplied.
    confusion = pd.DataFrame()
    if aqi_true_per_horizon and aqi_cat_true_per_horizon:
        aqi_df = _aqi_from_predictions(y_pred, station)
        aqi_maes = []
        cat_accs = []
        cat_f1s = []
        for h in HORIZONS:
            sub = aqi_df[aqi_df["horizon"] == h]
            y_true_aqi = aqi_true_per_horizon[h]
            y_true_cat = pd.Series(aqi_cat_true_per_horizon[h])
            y_pred_aqi = sub["aqi"].to_numpy()
            y_pred_cat = sub["category"].reset_index(drop=True)
            aqi_maes.append(mae(y_true_aqi, y_pred_aqi))
            cat_accs.append(category_accuracy(y_true_cat, y_pred_cat))
            cat_f1s.append(category_macro_f1(y_true_cat, y_pred_cat))
        overall["aqi_mae"] = float(np.nanmean(aqi_maes))
        overall["aqi_category_accuracy"] = float(np.nanmean(cat_accs))
        overall["aqi_category_macro_f1"] = float(np.nanmean(cat_f1s))
        per_horizon["aqi_mae"] = aqi_maes
        per_horizon["aqi_category_accuracy"] = cat_accs

        # Confusion matrix on horizon=24 only for brevity.
        sub24 = aqi_df[aqi_df["horizon"] == 24]
        labels = [c[2] for c in bp.CATEGORIES]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            aqi_cat_true_per_horizon[24], sub24["category"].to_numpy(),
            labels=labels,
        )
        confusion = pd.DataFrame(cm, index=labels, columns=labels)

    return EvalResult(
        model=model_name,
        overall=overall,
        per_horizon=per_horizon,
        per_pollutant=per_pollutant,
        per_station=per_station,
        confusion=confusion,
    )
