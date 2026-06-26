"""Forecast metrics: MAE, RMSE, MAPE on values; accuracy/F1 on AQI category."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    denom = np.maximum(np.abs(y_true[mask]), eps)
    return float(np.abs((y_true[mask] - y_pred[mask]) / denom).mean())


def category_accuracy(y_true_cat: pd.Series, y_pred_cat: pd.Series) -> float:
    mask = (y_true_cat != "Unknown") & (y_pred_cat != "Unknown")
    if not mask.any():
        return float("nan")
    return float((y_true_cat[mask] == y_pred_cat[mask]).mean())


def category_macro_f1(y_true_cat: pd.Series, y_pred_cat: pd.Series) -> float:
    mask = (y_true_cat != "Unknown") & (y_pred_cat != "Unknown")
    if not mask.any():
        return float("nan")
    return float(f1_score(y_true_cat[mask], y_pred_cat[mask], average="macro"))


def category_confusion(y_true_cat: pd.Series, y_pred_cat: pd.Series, labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
