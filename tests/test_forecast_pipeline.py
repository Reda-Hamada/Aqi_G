"""End-to-end smoke test: synthetic data -> features -> RF train -> forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqi.features.build import build_tabular, feature_columns, target_columns
from aqi.forecast.pipeline import forecast_tabular
from aqi.models.rf import RandomForestForecaster

from tests._synthetic import make_long_df


@pytest.fixture(scope="module")
def trained_rf_and_data():
    df_long = make_long_df(n_hours=24 * 60, n_stations=3, seed=42)
    df = build_tabular(df_long, include_spatial=True, include_weather=True, include_city=False)

    feats = feature_columns(df)
    tgts = target_columns()
    df = df[df[feats].notna().all(axis=1) & df[tgts].notna().all(axis=1)].reset_index(drop=True)
    assert len(df) > 100, "Synthetic dataset is too small after feature build"

    split = int(len(df) * 0.7)
    X_train, Y_train = df.iloc[:split][feats], df.iloc[:split][tgts]
    X_val,   Y_val   = df.iloc[split:].iloc[:200][feats], df.iloc[split:].iloc[:200][tgts]

    # Tiny RF to keep CI fast.
    model = RandomForestForecaster(n_estimators=10, max_depth=4, n_jobs=1)
    model.fit(X_train, Y_train, X_val, Y_val)
    return model, df, feats


def test_rf_predict_shape(trained_rf_and_data):
    model, df, feats = trained_rf_and_data
    preds = model.predict(df.iloc[:5][feats])
    assert preds.shape == (5, 6 * 24)
    assert not np.isnan(preds).any()


def test_forecast_pipeline_returns_24_horizons(trained_rf_and_data, tmp_path, monkeypatch):
    model, df, feats = trained_rf_and_data
    snapshot = df.iloc[[0]][feats]

    monkeypatch.setattr(
        "aqi.forecast.pipeline.FORECAST_LOG",
        tmp_path / "forecast_log.parquet",
    )
    out = forecast_tabular(
        model, snapshot,
        station="Aotizhongxin",
        issue_time=pd.Timestamp("2014-01-15 08:00", tz="Asia/Shanghai"),
        run_id="test-run",
        model_name="rf",
    )
    assert len(out.horizons) == 24
    h1 = out.horizons[0]
    assert h1["h"] == 1
    assert "pollutants" in h1 and len(h1["pollutants"]) == 6
    assert "aqi" in h1 and "category" in h1 and "dominant" in h1


def test_persistence_baseline_matches_lag1():
    from aqi.evaluate.baselines import persistence_predict
    df_long = make_long_df(n_hours=200, n_stations=1)
    df = build_tabular(df_long, include_spatial=False, include_weather=False, include_city=False)
    df = df.dropna(subset=["PM2.5"]).reset_index(drop=True)
    preds = persistence_predict(df.iloc[:10])
    # First horizon column should equal PM2.5 itself.
    np.testing.assert_allclose(preds[:, 0], df["PM2.5"].iloc[:10].to_numpy())
