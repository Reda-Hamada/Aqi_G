#!/usr/bin/env python
"""Evaluate a training run against the held-out test split.

Tabular models (rf/xgb/lgbm) and the naive baselines are scored on the tabular
test rows. The LSTM is scored on its sequence test windows (same test period;
the valid-row sets differ slightly because sequences need 72 h of lookback).
Every model is summarised with the same metric matrix and written into one
comparison report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aqi.aqi import breakpoints as bp
from aqi.aqi.compute import aqi_from_pollutants
from aqi.config import (
    HORIZONS,
    MODELS_DIR,
    POLLUTANTS,
    SEQUENCE_LENGTH,
    STATIONS,
    TABULAR_PARQUET,
)
from aqi.evaluate.baselines import (
    climatology_predict,
    persistence_predict,
    seasonal_naive_predict,
)
from aqi.evaluate.report import write_report
from aqi.evaluate.runner import evaluate
from aqi.features.build import build_sequences, feature_columns, target_columns
from aqi.splits.walk_forward import chronological_masks

P_LEN = len(POLLUTANTS)
H_LEN = len(HORIZONS)


def _load_model(model_name: str, run_dir: Path):
    sub = run_dir / model_name
    if model_name == "rf":
        from aqi.models.rf import RandomForestForecaster
        return RandomForestForecaster.load(sub)
    if model_name == "xgb":
        from aqi.models.xgb import XGBForecaster
        return XGBForecaster.load(sub)
    if model_name == "lgbm":
        from aqi.models.lgbm import LGBMForecaster
        return LGBMForecaster.load(sub)
    if model_name == "lstm":
        from aqi.models.lstm import LSTMForecaster
        return LSTMForecaster.load(sub)
    raise ValueError(model_name)


def _aqi_true_per_horizon(y_true: np.ndarray) -> tuple[dict, dict]:
    """Compute AQI value + category per horizon from a (N, P*H) true array.

    Column order is pollutant-major: column k*H_LEN + j is pollutant k at
    horizon j (matching `target_columns()` / the runner).
    """
    aqi_per_h, cat_per_h = {}, {}
    for j, h in enumerate(HORIZONS):
        cols = [k * H_LEN + j for k in range(P_LEN)]
        block = y_true[:, cols]  # (N, P) in POLLUTANTS order
        aqi_vals, cat_vals = [], []
        for row in block:
            d = dict(zip(POLLUTANTS, row))
            aqi, cat, _ = aqi_from_pollutants(
                d["PM2.5"], d["PM10"], d["SO2"], d["NO2"], d["CO"], d["O3"],
            )
            aqi_vals.append(aqi); cat_vals.append(cat)
        aqi_per_h[h] = np.array(aqi_vals)
        cat_per_h[h] = np.array(cat_vals)
    return aqi_per_h, cat_per_h


def _seq_to_runner(arr_nhp: np.ndarray) -> np.ndarray:
    """(N, H, P) -> (N, P*H) pollutant-major (runner/target_columns order)."""
    n = arr_nhp.shape[0]
    return arr_nhp.transpose(0, 2, 1).reshape(n, P_LEN * H_LEN)


def _lstm_test_predict(model, df: pd.DataFrame, feats: list[str], masks):
    """Stream per-station sequence windows in the test period and predict.

    Returns y_true, y_pred (both (N, P*H)), station, timestamp arrays.
    Predicting per station keeps peak memory bounded.
    """
    test_min = df.loc[masks.test, "timestamp"].min()
    test_max = df.loc[masks.test, "timestamp"].max()
    yts, yps, sts, tss = [], [], [], []
    for s in STATIONS:
        sub = df[df["station"] == s]
        pl = build_sequences(sub, sequence_length=SEQUENCE_LENGTH, feature_cols=feats)
        if len(pl["timestamp"]) == 0:
            continue
        ts = pd.to_datetime(pl["timestamp"])
        keep = np.asarray((ts >= test_min) & (ts <= test_max))
        if not keep.any():
            continue
        X = pl["X"][keep]
        y_true = _seq_to_runner(pl["y"][keep])
        station = pl["station"][keep]
        pred_nhp = model.predict(X, station)
        y_pred = _seq_to_runner(pred_nhp)
        yts.append(y_true); yps.append(y_pred)
        sts.append(station); tss.append(np.asarray(pl["timestamp"])[keep])
        del pl, X
    return (
        np.concatenate(yts), np.concatenate(yps),
        np.concatenate(sts), np.concatenate(tss),
    )


def main(run_id: str, models: list[str], notes_path: str | None = None) -> None:
    df = pd.read_parquet(TABULAR_PARQUET)
    masks = chronological_masks(df)
    feats = feature_columns(df)
    tgts = target_columns()
    run_dir = MODELS_DIR / run_id

    tabular_models = [m for m in models if m != "lstm"]
    want_lstm = "lstm" in models

    results = []

    # --- Tabular models + baselines (shared test rows) -------------------
    df_test = df[masks.test & df[tgts].notna().all(axis=1) & df[feats].notna().all(axis=1)].copy()
    X_test = df_test[feats]
    Y_test = df_test[tgts].to_numpy()
    station = df_test["station"].to_numpy()
    ts = df_test["timestamp"].to_numpy()
    print(f"[eval] Tabular test rows: {len(df_test):,}")
    aqi_true_h, aqi_cat_true_h = _aqi_true_per_horizon(Y_test)

    for name in tabular_models:
        print(f"[eval] {name}")
        model = _load_model(name, run_dir)
        y_pred = model.predict(X_test)
        results.append(evaluate(name, Y_test, y_pred, station, ts, aqi_true_h, aqi_cat_true_h))

    df_train = df[masks.train].copy()
    for name, fn in [
        ("persistence", lambda x: persistence_predict(x)),
        ("seasonal_naive", lambda x: seasonal_naive_predict(x)),
        ("climatology", lambda x: climatology_predict(x, df_train)),
    ]:
        print(f"[eval] baseline: {name}")
        y_pred = fn(df_test)
        results.append(evaluate(name, Y_test, y_pred, station, ts, aqi_true_h, aqi_cat_true_h))

    # --- LSTM (sequence test rows) ---------------------------------------
    if want_lstm:
        print("[eval] lstm (sequence windows)")
        model = _load_model("lstm", run_dir)
        y_true_l, y_pred_l, station_l, ts_l = _lstm_test_predict(model, df, feats, masks)
        print(f"[eval] LSTM test windows: {len(y_true_l):,}")
        aqi_true_l, aqi_cat_true_l = _aqi_true_per_horizon(y_true_l)
        results.append(evaluate("lstm", y_true_l, y_pred_l, station_l, ts_l,
                                aqi_true_l, aqi_cat_true_l))

    notes = {}
    if notes_path and Path(notes_path).exists():
        import json
        notes = json.loads(Path(notes_path).read_text())

    out_dir = write_report(results, run_id, notes=notes)
    print(f"[eval] Report written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "lstm"])
    ap.add_argument("--notes", default=None, help="Optional JSON file with report notes.")
    args = ap.parse_args()
    main(args.run_id, args.models, notes_path=args.notes)
