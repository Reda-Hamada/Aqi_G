#!/usr/bin/env python
"""Single inference: produce a 24-hour AQI forecast for a station and timestamp."""
from __future__ import annotations

import argparse

import pandas as pd

from aqi.config import MODELS_DIR, TABULAR_PARQUET, TIMEZONE
from aqi.features.build import feature_columns
from aqi.forecast.pipeline import forecast_tabular


def _load_model(model_name: str, run_id: str):
    sub = MODELS_DIR / run_id / model_name
    if model_name == "rf":
        from aqi.models.rf import RandomForestForecaster
        return RandomForestForecaster.load(sub)
    if model_name == "xgb":
        from aqi.models.xgb import XGBForecaster
        return XGBForecaster.load(sub)
    if model_name == "lgbm":
        from aqi.models.lgbm import LGBMForecaster
        return LGBMForecaster.load(sub)
    raise ValueError(model_name)


def main(run_id: str, model_name: str, station: str, asof: str) -> None:
    df = pd.read_parquet(TABULAR_PARQUET)
    feats = feature_columns(df)
    ts = pd.Timestamp(asof, tz=TIMEZONE)
    snapshot = df[(df["station"] == station) & (df["timestamp"] == ts)][feats]
    if snapshot.empty:
        raise SystemExit(f"No snapshot for station={station} at {ts}")
    model = _load_model(model_name, run_id)
    out = forecast_tabular(
        model, snapshot,
        station=station, issue_time=ts,
        run_id=run_id, model_name=model_name,
    )
    print(out.to_json())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--station", required=True)
    ap.add_argument("--asof", required=True, help="ISO timestamp, e.g. 2017-01-15T08:00")
    args = ap.parse_args()
    main(args.run_id, args.model, args.station, args.asof)
