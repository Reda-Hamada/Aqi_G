"""Calendar features (hour/dow/month sin-cos, weekend, holiday)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype("int8")

    try:
        from chinese_calendar import is_holiday  # type: ignore[import-not-found]
        # chinese_calendar expects naive dates; convert.
        dates = ts.dt.tz_convert(None).dt.date if ts.dt.tz else ts.dt.date
        df["is_holiday"] = dates.map(is_holiday).astype("int8")
    except Exception:
        df["is_holiday"] = 0
    return df
