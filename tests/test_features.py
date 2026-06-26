"""Feature engineering tests: shape, no future leakage, deterministic lags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import POLLUTANTS
from aqi.features.build import build_tabular, build_targets, feature_columns
from aqi.features.temporal import add_temporal_features

from tests._synthetic import make_long_df


def test_temporal_lag_is_exact_shift():
    df = make_long_df(n_hours=24 * 10, n_stations=2)
    out = add_temporal_features(df.copy(), columns=["PM2.5"])
    # For station 0, row at index 24 should have lag24 == PM2.5 at index 0.
    sub = out[out["station"] == "Aotizhongxin"].reset_index(drop=True)
    assert sub.loc[24, "PM2.5_lag24"] == sub.loc[0, "PM2.5"]
    assert sub.loc[1, "PM2.5_lag1"] == sub.loc[0, "PM2.5"]


def test_temporal_no_leakage_from_future():
    """rolling mean uses shifted values only -> value at t depends only on
    rows strictly before t."""
    df = make_long_df(n_hours=200, n_stations=1)
    out = add_temporal_features(df.copy(), columns=["PM2.5"])
    sub = out[out["station"] == "Aotizhongxin"].reset_index(drop=True)
    # Mutate a future value; the rolling stat at an earlier index must not change.
    orig = sub.loc[50, "PM2.5_rmean24"]
    df2 = df.copy()
    df2.loc[100, "PM2.5"] = 9999
    out2 = add_temporal_features(df2, columns=["PM2.5"])
    sub2 = out2[out2["station"] == "Aotizhongxin"].reset_index(drop=True)
    assert sub2.loc[50, "PM2.5_rmean24"] == orig


def test_targets_are_future_values():
    df = make_long_df(n_hours=100, n_stations=1)
    out = build_targets(df, horizons=[1, 6, 24])
    sub = out[out["station"] == "Aotizhongxin"].reset_index(drop=True)
    assert sub.loc[0, "y_PM2.5_h1"] == sub.loc[1, "PM2.5"]
    assert sub.loc[0, "y_PM2.5_h24"] == sub.loc[24, "PM2.5"]


def test_build_tabular_produces_expected_columns():
    df = make_long_df(n_hours=24 * 21, n_stations=3)
    tab = build_tabular(df, include_spatial=True, include_weather=True, include_city=True)
    # 6 pollutants * 24 horizons = 144 target columns
    target_cols = [c for c in tab.columns if c.startswith("y_")]
    assert len(target_cols) == 6 * 24
    for p in POLLUTANTS:
        assert f"{p}_lag1" in tab.columns
        assert f"{p}_neighmean_lag0" in tab.columns
        assert f"{p}_neighmean_lag24" in tab.columns
    assert "wind_u" in tab.columns and "wind_v" in tab.columns
    assert "hour_sin" in tab.columns
    assert "station_id" in tab.columns
    # CITY pseudo-station included.
    assert (tab["station"] == "CITY").any()


def test_feature_columns_excludes_targets_and_ids():
    df = make_long_df(n_hours=24 * 14, n_stations=2)
    tab = build_tabular(df)
    feats = feature_columns(tab)
    assert "y_PM2.5_h1" not in feats
    assert "timestamp" not in feats
    assert "station" not in feats
    assert "wd" not in feats
    # but station_id should be in (it's numeric)
    assert "station_id" in feats


def test_spatial_neighbor_mean_matches_manual():
    df = make_long_df(n_hours=24 * 7, n_stations=3)
    tab = build_tabular(df, include_spatial=True, include_weather=False, include_city=False)
    # Pick a non-NaN row and check the neighbor mean equals mean of the other 2.
    t = tab["timestamp"].iloc[100]
    row = tab[(tab["timestamp"] == t) & (tab["station"] == "Aotizhongxin")].iloc[0]
    others = df[(df["timestamp"] == t) & (df["station"] != "Aotizhongxin")]["PM2.5"].mean()
    np.testing.assert_allclose(row["PM2.5_neighmean_lag0"], others, rtol=1e-6)
