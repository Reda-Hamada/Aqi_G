#!/usr/bin/env python
"""Train one or more model families on the tabular dataset.

Usage:
  python scripts/03_train_models.py --model lgbm
  python scripts/03_train_models.py --model all
  python scripts/03_train_models.py --model rf --quick   # short training subset
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from aqi.config import POLLUTANTS, SEQUENCE_LENGTH, STATIONS, TABULAR_PARQUET, set_global_seed
from aqi.features.build import build_sequences, feature_columns, target_columns
from aqi.persistence.registry import make_run_id, model_dir, write_run_manifest
from aqi.splits.walk_forward import chronological_masks


def _slice_xy(df: pd.DataFrame, feats: list[str], targets: list[str]):
    keep = df[targets].notna().all(axis=1) & df[feats].notna().all(axis=1)
    sub = df[keep]
    return sub[feats], sub[targets]


def train_tabular_model(name: str, df: pd.DataFrame, masks, quick: bool, tree_rows: int | None = None):
    feats = feature_columns(df)
    tgts = target_columns()
    X_train, Y_train = _slice_xy(df[masks.train], feats, tgts)
    X_val, Y_val = _slice_xy(df[masks.val], feats, tgts)

    if quick:
        X_train, Y_train = X_train.iloc[:5000], Y_train.iloc[:5000]
        X_val, Y_val = X_val.iloc[:2000], Y_val.iloc[:2000]
    elif tree_rows and len(X_train) > tree_rows:
        # Approximate (reduced-data) tree fit: random, seeded subsample so the
        # result is a genuine measurement rather than a fabricated number.
        idx = X_train.sample(n=tree_rows, random_state=42).index
        X_train, Y_train = X_train.loc[idx], Y_train.loc[idx]
        v = min(len(X_val), max(2000, tree_rows // 4))
        X_val, Y_val = X_val.iloc[:v], Y_val.iloc[:v]

    # Approximate (reduced-data) mode: fewer estimators so the subsampled fit
    # finishes in minutes. Numbers are genuine but not final-thesis-grade.
    approx = bool(tree_rows)
    if name == "rf":
        from aqi.models.rf import RandomForestForecaster
        model = RandomForestForecaster(n_estimators=150 if approx else 400)
    elif name == "xgb":
        from aqi.models.xgb import XGBForecaster
        model = XGBForecaster(n_estimators=200 if approx else 1000)
    elif name == "lgbm":
        from aqi.models.lgbm import LGBMForecaster
        model = LGBMForecaster(n_estimators=500 if approx else 2000)
    else:
        raise ValueError(name)

    t0 = time.perf_counter()
    model.fit(X_train, Y_train, X_val, Y_val)
    elapsed = time.perf_counter() - t0
    print(f"[train] {name}: {elapsed:.1f}s on {len(X_train):,} rows")
    return model


def _subsample(n: int, cap: int | None, seed: int = 42) -> np.ndarray:
    """Evenly-strided indices into [0, n) capped at `cap` (preserves coverage)."""
    if cap is None or n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).round().astype(int)


def train_lstm(df: pd.DataFrame, masks, quick: bool, epochs: int = 20,
               max_windows: int | None = 80000):
    """Memory-safe LSTM training.

    Builds sequence windows one station at a time and immediately subsamples
    each station's train/val windows before accumulating, so peak RAM stays
    bounded (the full all-station window tensor would be ~30 GB).
    """
    from aqi.models.lstm import LSTMForecaster
    feats = feature_columns(df)
    train_max = df.loc[masks.train, "timestamp"].max()
    val_max = df.loc[masks.val, "timestamp"].max()

    n_stations = len(STATIONS)
    if quick:
        per_tr = 2000 // n_stations + 1
        per_va = 500 // n_stations + 1
    else:
        per_tr = (max_windows // n_stations + 1) if max_windows else None
        per_va = (max_windows // (4 * n_stations) + 1) if max_windows else None

    X_tr_p, y_tr_p, s_tr_p = [], [], []
    X_va_p, y_va_p, s_va_p = [], [], []
    for s in STATIONS:
        sub = df[df["station"] == s]
        pl = build_sequences(sub, sequence_length=SEQUENCE_LENGTH, feature_cols=feats)
        ts = pd.to_datetime(pl["timestamp"])
        tr_m = np.asarray(ts <= train_max)
        va_m = np.asarray((ts > train_max) & (ts <= val_max))
        tr_idx = np.flatnonzero(tr_m)[_subsample(int(tr_m.sum()), per_tr)]
        va_idx = np.flatnonzero(va_m)[_subsample(int(va_m.sum()), per_va)]
        X_tr_p.append(pl["X"][tr_idx]); y_tr_p.append(pl["y"][tr_idx]); s_tr_p.append(pl["station"][tr_idx])
        X_va_p.append(pl["X"][va_idx]); y_va_p.append(pl["y"][va_idx]); s_va_p.append(pl["station"][va_idx])
        del pl

    X_tr = np.concatenate(X_tr_p); y_tr = np.concatenate(y_tr_p); s_tr = np.concatenate(s_tr_p)
    X_va = np.concatenate(X_va_p); y_va = np.concatenate(y_va_p); s_va = np.concatenate(s_va_p)
    del X_tr_p, y_tr_p, X_va_p, y_va_p

    model = LSTMForecaster(n_features=X_tr.shape[-1], epochs=3 if quick else epochs)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, s_tr, X_va, y_va, s_va)
    print(f"[train] lstm: {time.perf_counter()-t0:.1f}s on {len(X_tr):,} train / "
          f"{len(X_va):,} val windows ({epochs} epochs max)")
    return model


def main(model_names: list[str], quick: bool, tree_rows: int | None = None,
         lstm_epochs: int = 20, lstm_max_windows: int | None = 80000,
         run_id: str | None = None) -> str:
    set_global_seed()
    print(f"[train] Loading {TABULAR_PARQUET}")
    df = pd.read_parquet(TABULAR_PARQUET)
    masks = chronological_masks(df)
    run_id = run_id or make_run_id()
    print(f"[train] run_id={run_id}")

    summary = {}
    for name in model_names:
        out_dir = model_dir(run_id, name)
        if name == "lstm":
            model = train_lstm(df, masks, quick=quick, epochs=lstm_epochs,
                               max_windows=lstm_max_windows)
        else:
            model = train_tabular_model(name, df, masks, quick=quick, tree_rows=tree_rows)
        model.save(out_dir)
        summary[name] = {"dir": str(out_dir)}
        print(f"[train] saved {name} -> {out_dir}")

    write_run_manifest(run_id, {
        "models": summary,
        "tabular_path": str(TABULAR_PARQUET),
        "quick_mode": quick,
        "tree_rows": tree_rows,
        "lstm_epochs": lstm_epochs,
        "lstm_max_windows": lstm_max_windows,
    })
    print(f"[train] Done. run_id={run_id}")
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "xgb", "lgbm", "lstm", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tree-rows", type=int, default=None,
                    help="Subsample tree training to N rows (approximate fit).")
    ap.add_argument("--lstm-epochs", type=int, default=20)
    ap.add_argument("--lstm-max-windows", type=int, default=80000,
                    help="Cap total LSTM training windows (memory bound).")
    ap.add_argument("--run-id", default=None, help="Reuse an existing run id.")
    args = ap.parse_args()
    names = ["rf", "xgb", "lgbm", "lstm"] if args.model == "all" else [args.model]
    main(names, quick=args.quick, tree_rows=args.tree_rows,
         lstm_epochs=args.lstm_epochs, lstm_max_windows=args.lstm_max_windows,
         run_id=args.run_id)
