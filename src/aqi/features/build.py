"""Single orchestrator: cleaned long df -> tabular features + targets.

This is the only place feature-leakage rules live. Any new feature should be
added here so the leakage test (tests/test_features.py) sees it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import HORIZONS, POLLUTANTS, SEQUENCES_DIR, STATIONS, TABULAR_PARQUET
from aqi.features.calendar import add_calendar_features
from aqi.features.spatial import add_spatial_features, build_city_aggregate
from aqi.features.temporal import add_temporal_features
from aqi.features.weather import add_weather_features

LAG_COLUMNS = POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def build_targets(df: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Add y_<pollutant>_h<h> columns = pollutant at t+h, per station."""
    df = df.sort_values(["station", "timestamp"], kind="stable").copy()
    groups = df.groupby("station", sort=False)
    new = {}
    for p in POLLUTANTS:
        for h in horizons:
            new[f"y_{p}_h{h}"] = groups[p].shift(-h)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build_tabular(
    df: pd.DataFrame,
    *,
    include_spatial: bool = True,
    include_weather: bool = True,
    include_city: bool = True,
) -> pd.DataFrame:
    """Build the model-ready tabular DataFrame from a cleaned long df.

    Set `include_spatial=False` for the spatial ablation and
    `include_weather=False` for the weather ablation.
    """
    df = df.copy()

    if include_weather:
        df = add_weather_features(df)
    if include_spatial:
        df = add_spatial_features(df)

    # Add a synthetic "CITY" pseudo-station so a single model can serve both
    # station-level and city-level inference.
    if include_city:
        city = build_city_aggregate(df)
        df = pd.concat([df, city], ignore_index=True)
        df = df.sort_values(["station", "timestamp"], kind="stable")

    df = add_temporal_features(df, columns=LAG_COLUMNS)
    df = add_calendar_features(df)
    df = build_targets(df, HORIZONS)

    # station_id as categorical code for tree models.
    all_stations = STATIONS + (["CITY"] if include_city else [])
    cat = pd.Categorical(df["station"], categories=all_stations, ordered=False)
    df["station_id"] = cat.codes.astype("int16")

    return df


def target_columns() -> list[str]:
    return [f"y_{p}_h{h}" for p in POLLUTANTS for h in HORIZONS]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All non-target, non-identifier columns suitable as model inputs."""
    targets = set(target_columns())
    skip = targets | {"timestamp", "station", "wd"} | {
        "aqi", "aqi_category", "dominant_pollutant",
    }
    cols = [c for c in df.columns if c not in skip]
    # Force numeric dtype check; drop accidental object cols.
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def write_tabular(df: pd.DataFrame, path=TABULAR_PARQUET) -> None:
    """Persist features + targets. Drop rows where ALL targets are NaN."""
    targets = target_columns()
    keep = ~df[targets].isna().all(axis=1)
    out = df[keep].reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(f"[features.build] Wrote {len(out):,} rows × {out.shape[1]} cols to {path}")


# ---------------------------------------------------------------------------
# Sequence dataset for the LSTM
# ---------------------------------------------------------------------------
def build_sequences(
    df: pd.DataFrame,
    *,
    sequence_length: int = 72,
    horizons: list[int] = HORIZONS,
    feature_cols: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build (X, y, station, timestamp) arrays.

    X has shape (N, T, F), y has shape (N, H, P) for P pollutants.
    Rows containing NaNs anywhere in their window are dropped.
    """
    if feature_cols is None:
        feature_cols = feature_columns(df)
    df = df.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True)

    Xs, Ys, stns, tss = [], [], [], []
    H = len(horizons)
    P = len(POLLUTANTS)

    # We iterate per station; sequences are bounded inside each station.
    for s, sub in df.groupby("station", sort=False):
        X_all = sub[feature_cols].to_numpy(dtype="float32")
        Y_all = sub[[f"y_{p}_h{h}" for p in POLLUTANTS for h in horizons]].to_numpy(dtype="float32")
        # reshape Y to (N, P, H) then transpose to (N, H, P) for consistency
        Y_all = Y_all.reshape(len(sub), P, H).transpose(0, 2, 1)
        ts_arr = sub["timestamp"].to_numpy()
        N = len(sub)
        for t in range(sequence_length - 1, N):
            xw = X_all[t - sequence_length + 1: t + 1]
            yw = Y_all[t]
            if np.isnan(xw).any() or np.isnan(yw).any():
                continue
            Xs.append(xw)
            Ys.append(yw)
            stns.append(s)
            tss.append(ts_arr[t])
    return {
        "X": np.stack(Xs) if Xs else np.zeros((0, sequence_length, len(feature_cols)), dtype="float32"),
        "y": np.stack(Ys) if Ys else np.zeros((0, len(horizons), P), dtype="float32"),
        "station": np.array(stns, dtype=object),
        "timestamp": np.array(tss),
        "feature_cols": np.array(feature_cols, dtype=object),
    }


def write_sequences(payload: dict[str, np.ndarray], station: str) -> None:
    out = SEQUENCES_DIR / f"{station}.npz"
    np.savez_compressed(out, **payload)
    print(f"[features.build] Wrote sequence file {out} (X={payload['X'].shape})")
