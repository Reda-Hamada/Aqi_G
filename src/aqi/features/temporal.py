"""Lag, rolling, and difference features.

All features are computed per (station, column) and use STRICTLY past data only,
i.e. for row at time t the lag uses x_{t-k}.
"""
from __future__ import annotations

import pandas as pd

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [6, 24, 72]


def add_temporal_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Append lag, rolling, and diff features for each column in `columns`.

    Operates per-station so groups don't leak into each other.
    """
    df = df.sort_values(["station", "timestamp"], kind="stable").copy()
    groups = df.groupby("station", sort=False)

    new_cols: dict[str, pd.Series] = {}

    for col in columns:
        for lag in LAG_HOURS:
            new_cols[f"{col}_lag{lag}"] = groups[col].shift(lag)
        for w in ROLL_WINDOWS:
            shifted = groups[col].shift(1)  # exclude current row to avoid leakage
            new_cols[f"{col}_rmean{w}"] = shifted.rolling(window=w, min_periods=max(1, w // 4)).mean()
            new_cols[f"{col}_rstd{w}"] = shifted.rolling(window=w, min_periods=max(2, w // 4)).std()
        # Differences
        new_cols[f"{col}_d1"] = df[col] - groups[col].shift(1)
        new_cols[f"{col}_d24"] = df[col] - groups[col].shift(24)
        new_cols[f"{col}_d168"] = df[col] - groups[col].shift(168)

    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return out
