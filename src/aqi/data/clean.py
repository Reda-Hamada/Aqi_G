"""Value-range clipping for pollutants and weather."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import COMPASS_16, POLLUTANTS

# Physical bounds. Anything outside these is replaced with NaN.
WEATHER_BOUNDS: dict[str, tuple[float, float]] = {
    "TEMP": (-40.0, 50.0),
    "PRES": (950.0, 1100.0),
    "DEWP": (-50.0, 40.0),
    "RAIN": (0.0, 200.0),
    "WSPM": (0.0, 50.0),
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clip implausible values to NaN. Returns a new DataFrame."""
    df = df.copy()

    for col in POLLUTANTS:
        df.loc[df[col] < 0, col] = np.nan

    for col, (lo, hi) in WEATHER_BOUNDS.items():
        invalid = (df[col] < lo) | (df[col] > hi)
        df.loc[invalid, col] = np.nan

    # Dewpoint must not exceed temperature.
    bad_dew = df["DEWP"] > df["TEMP"]
    df.loc[bad_dew, "DEWP"] = np.nan

    df.loc[~df["wd"].isin(COMPASS_16), "wd"] = np.nan
    return df
