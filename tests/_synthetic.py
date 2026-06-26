"""Small synthetic dataset matching cleaned_long.parquet schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import POLLUTANTS, STATIONS, TIMEZONE


def make_long_df(
    n_hours: int = 24 * 30,            # 30 days
    n_stations: int = 3,
    start: str = "2014-01-01 00:00",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small but schema-complete long DataFrame for fast tests."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n_hours, freq="h", tz=TIMEZONE)
    rows = []
    for s in STATIONS[:n_stations]:
        # Pollutants and weather with mild structure so rolling stats are non-trivial.
        base = rng.normal(50, 10, n_hours)
        diurnal = 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
        row = pd.DataFrame({"timestamp": ts, "station": s})
        for p in POLLUTANTS:
            row[p] = np.clip(base + diurnal + rng.normal(0, 5, n_hours), 0.1, None)
        row["TEMP"] = 5 + 10 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 30)) + rng.normal(0, 1, n_hours)
        row["PRES"] = 1015 + rng.normal(0, 3, n_hours)
        row["DEWP"] = row["TEMP"] - rng.uniform(2, 10, n_hours)
        row["RAIN"] = 0.0
        row["WSPM"] = np.abs(rng.normal(2, 1, n_hours))
        row["wd"] = rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], n_hours)
        rows.append(row)
    return pd.concat(rows, ignore_index=True)
