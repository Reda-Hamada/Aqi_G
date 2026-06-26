"""Weather-derived features: wind u/v, T-Dew stability, pressure tendency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import COMPASS_16

# Angle in radians measured from North, going clockwise.
_COMPASS_RAD = {name: i * (2 * np.pi / 16) for i, name in enumerate(COMPASS_16)}


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rad = df["wd"].map(_COMPASS_RAD)
    # Meteorological convention: wind direction is the direction the wind is
    # COMING FROM. To get the vector blowing TOWARD a destination, negate.
    df["wind_u"] = -df["WSPM"] * np.sin(rad)
    df["wind_v"] = -df["WSPM"] * np.cos(rad)

    df["T_minus_Dew"] = df["TEMP"] - df["DEWP"]

    pres_groups = df.groupby("station", sort=False)["PRES"]
    df["PRES_tend_6h"] = df["PRES"] - pres_groups.shift(6)
    df["PRES_tend_24h"] = df["PRES"] - pres_groups.shift(24)
    return df
