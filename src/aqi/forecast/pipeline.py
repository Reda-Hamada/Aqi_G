"""Forecast pipeline: snapshot -> features -> model -> AQI output."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aqi.aqi.compute import aqi_from_pollutants
from aqi.config import FORECAST_LOG, HORIZONS, POLLUTANTS
from aqi.persistence.registry import run_dir


@dataclass
class ForecastOutput:
    station: str
    issue_time: str
    run_id: str
    model: str
    horizons: list[dict]
    confidence: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def forecast_tabular(
    forecaster,
    snapshot: pd.DataFrame,
    *,
    station: str,
    issue_time: pd.Timestamp,
    run_id: str,
    model_name: str,
    confidence: str = "ok",
) -> ForecastOutput:
    """Run a single inference and produce the spec's output structure.

    `snapshot` must contain exactly the feature columns the forecaster expects,
    in the same order. Single-row inputs are typical for inference; the
    function works for multi-row inputs too but only the first row's forecast
    is recorded in the output object.
    """
    y_pred = forecaster.predict(snapshot)  # (N, P*H)
    horizons = []
    h_len = len(HORIZONS)
    for j, h in enumerate(HORIZONS):
        vals = {p: float(y_pred[0, k * h_len + j]) for k, p in enumerate(POLLUTANTS)}
        aqi, cat, dom = aqi_from_pollutants(
            vals["PM2.5"], vals["PM10"], vals["SO2"],
            vals["NO2"], vals["CO"], vals["O3"],
        )
        horizons.append({
            "h": h,
            "pollutants": vals,
            "aqi": aqi,
            "category": cat,
            "dominant": dom,
        })

    out = ForecastOutput(
        station=station,
        issue_time=pd.Timestamp(issue_time).isoformat(),
        run_id=run_id,
        model=model_name,
        horizons=horizons,
        confidence=confidence,
    )
    _log_forecast(snapshot, out)
    return out


def _log_forecast(snapshot: pd.DataFrame, out: ForecastOutput) -> None:
    """Append the request/response pair to forecast_log.parquet."""
    snapshot_hash = pd.util.hash_pandas_object(snapshot.head(1).reset_index(drop=True)).sum()
    row = {
        "request_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "station": out.station,
        "issue_time": out.issue_time,
        "run_id": out.run_id,
        "model": out.model,
        "input_snapshot_hash": int(snapshot_hash),
        "output_json": out.to_json(),
    }
    if FORECAST_LOG.exists():
        df = pd.read_parquet(FORECAST_LOG)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_parquet(FORECAST_LOG, index=False)
