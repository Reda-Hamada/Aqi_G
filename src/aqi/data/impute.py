"""Gap-aware imputation with missingness flags. Avoids future leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import POLLUTANTS, TRAIN_END, WEATHER_CATEGORICAL, WEATHER_NUMERIC

SHORT_GAP_MAX = 3       # hours → linear interpolation
MEDIUM_GAP_MAX = 24     # hours → climatological hour-of-day mean


def _run_lengths(mask: pd.Series) -> pd.Series:
    """Per row, the length of the contiguous True run containing it (0 if False)."""
    grp = (mask != mask.shift()).cumsum()
    sizes = mask.groupby(grp).transform("size")
    return sizes.where(mask, 0).astype("int32")


def _climatology(train_df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of `col` per (station, month, hour) using only training rows."""
    tmp = train_df[["station", "timestamp", col]].copy()
    tmp["month"] = tmp["timestamp"].dt.month
    tmp["hour"] = tmp["timestamp"].dt.hour
    return tmp.groupby(["station", "month", "hour"])[col].mean()


def impute(df: pd.DataFrame, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Impute NaNs in pollutants + weather using gap-aware strategy.

    - Runs ≤ 3 h: linear interpolation on time within station.
    - Runs 4–24 h: train-only climatology by (station, month, hour).
    - Runs > 24 h: left as NaN.
    - Adds `was_imputed_<col>` and `missing_run_<col>` columns for the six pollutants.
    """
    df = df.copy()
    df = df.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True)
    train_mask = df["timestamp"] <= pd.Timestamp(train_end, tz=df["timestamp"].dt.tz)

    cols = POLLUTANTS + WEATHER_NUMERIC

    # Pre-compute climatology per column from train-only data.
    train_df = df.loc[train_mask, ["station", "timestamp"] + cols]
    climatologies = {c: _climatology(train_df, c) for c in cols}

    for col in cols:
        nan_mask = df[col].isna()
        original_nan = nan_mask.copy()
        runs = df.groupby("station")[col].apply(
            lambda s: _run_lengths(s.isna())
        ).reset_index(level=0, drop=True)
        runs.index = df.index

        # Short gaps: linear interpolation per station.
        short = (runs > 0) & (runs <= SHORT_GAP_MAX) & nan_mask
        if short.any():
            interp = (
                df.set_index("timestamp")
                  .groupby("station")[col]
                  .apply(lambda s: s.interpolate(method="time", limit=SHORT_GAP_MAX))
            )
            interp = interp.reset_index(level=0, drop=True).reindex(df.set_index("timestamp").index)
            interp.index = df.index
            df.loc[short, col] = interp.loc[short]
            nan_mask = df[col].isna()

        # Medium gaps: climatological fill.
        medium = (runs > SHORT_GAP_MAX) & (runs <= MEDIUM_GAP_MAX) & nan_mask
        if medium.any():
            clim = climatologies[col]
            keys = pd.MultiIndex.from_arrays([
                df.loc[medium, "station"].to_numpy(),
                df.loc[medium, "timestamp"].dt.month.to_numpy(),
                df.loc[medium, "timestamp"].dt.hour.to_numpy(),
            ])
            df.loc[medium, col] = clim.reindex(keys).to_numpy()

        if col in POLLUTANTS:
            df[f"was_imputed_{col}"] = (original_nan & df[col].notna()).astype("int8")
            df[f"missing_run_{col}"] = runs.astype("int32")

    # Wind direction: forward-fill within station for short gaps only.
    df["wd"] = df.groupby("station")["wd"].transform(
        lambda s: s.ffill(limit=SHORT_GAP_MAX)
    )

    return df
