"""AQI computation from pollutant concentrations.

Inputs are pollutant concentrations in raw PRSA units (ug/m^3 for all six
pollutants). Internally we convert SO2/NO2/O3 to ppb and CO to ppm using
MOLAR_VOLUME at 25 C / 1 atm before applying EPA breakpoints.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from aqi.aqi import breakpoints as bp
from aqi.config import MOLAR_MASS, MOLAR_VOLUME_25C_1ATM, POLLUTANTS

# ---------------------------------------------------------------------------
# Unit conversions ug/m^3 -> EPA unit
# ---------------------------------------------------------------------------
def ugm3_to_ppb(ugm3: float | np.ndarray, gas: str) -> float | np.ndarray:
    return ugm3 * MOLAR_VOLUME_25C_1ATM / MOLAR_MASS[gas]


def ugm3_to_ppm(ugm3: float | np.ndarray, gas: str) -> float | np.ndarray:
    return ugm3_to_ppb(ugm3, gas) / 1000.0


# ---------------------------------------------------------------------------
# Pollutant truncation per EPA reporting resolution
# ---------------------------------------------------------------------------
def _truncate(value: float, pollutant: str) -> float:
    if value is None or value != value:
        return float("nan")
    if pollutant in ("PM2.5",):
        return np.floor(value * 10) / 10
    if pollutant in ("PM10", "SO2", "NO2"):
        return float(int(value))
    if pollutant in ("CO",):
        return np.floor(value * 10) / 10
    if pollutant in ("O3_8H", "O3_1H"):
        return np.floor(value * 1000) / 1000
    return value


# ---------------------------------------------------------------------------
# Sub-index from a breakpoint table
# ---------------------------------------------------------------------------
def sub_index(value: float, table: list[bp.Bp]) -> float:
    """Piecewise-linear interpolation. Returns NaN if value is NaN or below the
    first breakpoint's low; for values above the last breakpoint, returns 500.
    """
    if value is None or value != value:
        return float("nan")
    if value < table[0][0]:
        # below the lowest valid concentration -> still in the lowest band
        c_lo, c_hi, i_lo, i_hi = table[0]
        return ((i_hi - i_lo) / (c_hi - c_lo)) * (max(value, c_lo) - c_lo) + i_lo
    for c_lo, c_hi, i_lo, i_hi in table:
        if c_lo <= value <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
    # above all breakpoints
    return 500.0


# ---------------------------------------------------------------------------
# Per-pollutant sub-index (after unit conversion + truncation)
# ---------------------------------------------------------------------------
def subindex_pm25(ugm3: float) -> float:
    return sub_index(_truncate(ugm3, "PM2.5"), bp.PM25)


def subindex_pm10(ugm3: float) -> float:
    return sub_index(_truncate(ugm3, "PM10"), bp.PM10)


def subindex_so2(ugm3_1h: float, ugm3_24h: float | None = None) -> float:
    """1-h SO2 in ppb is used for AQI <= 200; otherwise 24-h average is used."""
    ppb_1h = ugm3_to_ppb(ugm3_1h, "SO2")
    idx_1h = sub_index(_truncate(ppb_1h, "SO2"), bp.SO2_1H)
    if not np.isnan(idx_1h) and idx_1h <= 200:
        return idx_1h
    if ugm3_24h is not None and not np.isnan(ugm3_24h):
        ppb_24h = ugm3_to_ppb(ugm3_24h, "SO2")
        return sub_index(_truncate(ppb_24h, "SO2"), bp.SO2_24H)
    # AQI > 200 indicated by 1-h reading but no 24-h available: fall back to 1-h interp.
    return idx_1h


def subindex_no2(ugm3: float) -> float:
    ppb = ugm3_to_ppb(ugm3, "NO2")
    return sub_index(_truncate(ppb, "NO2"), bp.NO2_1H)


def subindex_co(ugm3_8h: float) -> float:
    ppm = ugm3_to_ppm(ugm3_8h, "CO")
    return sub_index(_truncate(ppm, "CO"), bp.CO)


def subindex_o3(ugm3_8h: float, ugm3_1h: float | None = None) -> float:
    """8-hour value is primary; for AQI > 100, take max of 8-h and 1-h indices."""
    ppm_8h = ugm3_to_ppm(ugm3_8h, "O3")
    idx_8h = sub_index(_truncate(ppm_8h, "O3_8H"), bp.O3_8H)
    if ugm3_1h is None or np.isnan(ugm3_1h):
        return idx_8h
    ppm_1h = ugm3_to_ppm(ugm3_1h, "O3")
    idx_1h = sub_index(_truncate(ppm_1h, "O3_1H"), bp.O3_1H)
    candidates = [v for v in (idx_8h, idx_1h) if not np.isnan(v)]
    return max(candidates) if candidates else float("nan")


# ---------------------------------------------------------------------------
# Hourly snapshot -> AQI (helper for forecast post-processing)
# ---------------------------------------------------------------------------
def aqi_from_pollutants(
    pm25: float,
    pm10: float,
    so2: float,
    no2: float,
    co: float,
    o3: float,
) -> Tuple[float, str, str]:
    """Compute AQI from a *single hourly* snapshot. Inputs in ug/m^3.

    This is a convenience for cases where averaging windows (24-h, 8-h) cannot
    be applied (e.g. point forecasts not anchored to a long history); in that
    case the function treats the 1-hour value as the relevant input for every
    pollutant. For training-label AQI, use `aqi_from_dataframe()` which
    correctly applies trailing windows.
    """
    indices = {
        "PM2.5": subindex_pm25(pm25),
        "PM10":  subindex_pm10(pm10),
        "SO2":   subindex_so2(so2),
        "NO2":   subindex_no2(no2),
        "CO":    subindex_co(co),
        "O3":    subindex_o3(o3, ugm3_1h=o3),
    }
    valid = {p: v for p, v in indices.items() if not np.isnan(v)}
    if not valid:
        return float("nan"), "Unknown", "None"
    dominant = max(valid, key=valid.get)
    aqi = valid[dominant]
    return aqi, bp.aqi_to_category(aqi), dominant


# ---------------------------------------------------------------------------
# DataFrame-level AQI: applies the proper averaging windows by station/timestamp
# ---------------------------------------------------------------------------
def aqi_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly AQI, category, dominant pollutant for an hourly long df.

    Required columns: timestamp, station, PM2.5, PM10, SO2, NO2, CO, O3.
    Adds columns: aqi, aqi_category, dominant_pollutant, and per-pollutant
    sub-indices (sub_<p>).

    Uses trailing windows:
      - PM2.5/PM10: 24-h mean
      - O3:         8-h mean (with 1-h fallback)
      - CO:         8-h mean
      - SO2/NO2:    1-h (instantaneous)
    """
    df = df.sort_values(["station", "timestamp"], kind="stable").copy()

    def _rolling(col: str, window: int) -> pd.Series:
        return df.groupby("station")[col].transform(
            lambda s: s.rolling(window=window, min_periods=max(1, window // 2)).mean()
        )

    pm25_24 = _rolling("PM2.5", 24)
    pm10_24 = _rolling("PM10", 24)
    o3_8    = _rolling("O3", 8)
    co_8    = _rolling("CO", 8)
    so2_1   = df["SO2"]
    no2_1   = df["NO2"]
    o3_1    = df["O3"]

    df["sub_PM2.5"] = pm25_24.map(subindex_pm25)
    df["sub_PM10"]  = pm10_24.map(subindex_pm10)
    df["sub_SO2"]   = so2_1.map(subindex_so2)
    df["sub_NO2"]   = no2_1.map(subindex_no2)
    df["sub_CO"]    = co_8.map(subindex_co)
    df["sub_O3"]    = [
        subindex_o3(a, b) for a, b in zip(o3_8.to_numpy(), o3_1.to_numpy())
    ]

    sub_cols = [f"sub_{p}" for p in POLLUTANTS]
    sub_arr = df[sub_cols].to_numpy()
    with np.errstate(invalid="ignore"):
        df["aqi"] = np.nanmax(sub_arr, axis=1)

    dom_idx = np.nanargmax(np.where(np.isnan(sub_arr), -np.inf, sub_arr), axis=1)
    df["dominant_pollutant"] = [POLLUTANTS[i] for i in dom_idx]
    df.loc[df["aqi"].isna(), "dominant_pollutant"] = "None"
    df["aqi_category"] = df["aqi"].map(bp.aqi_to_category)
    return df
