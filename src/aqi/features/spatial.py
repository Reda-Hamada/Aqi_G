"""Spatial features that incorporate the other 11 stations.

For each target station and timestamp, we compute:
  - Mean / std / min / max of each pollutant across the 11 neighbors at lag 0
    and lag 24 (training-safe because we operate on already-loaded history).
  - Inverse-distance-weighted mean using great-circle distances.

A separate "city" pseudo-station is also built (cross-station mean) for
city-level forecasts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aqi.config import POLLUTANTS, STATION_COORDS, STATIONS

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def station_distance_matrix() -> pd.DataFrame:
    rows = []
    for s1 in STATIONS:
        for s2 in STATIONS:
            d = _haversine_km(*STATION_COORDS[s1], *STATION_COORDS[s2])
            rows.append((s1, s2, d))
    return pd.DataFrame(rows, columns=["s1", "s2", "km"])


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add neighbor aggregations and inverse-distance-weighted means.

    Adds, for each pollutant p in POLLUTANTS:
      - `{p}_neighmean_lag0`, `{p}_neighstd_lag0`, `{p}_neighmax_lag0`
      - `{p}_neighmean_lag24`
      - `{p}_idw_lag0` (inverse-distance weighted mean across the other 11)
    """
    df = df.sort_values(["timestamp", "station"], kind="stable").copy()

    # Pivot wide so neighbor stats are vectorised.
    pivots = {p: df.pivot(index="timestamp", columns="station", values=p) for p in POLLUTANTS}

    dist = station_distance_matrix().pivot(index="s1", columns="s2", values="km")
    # Inverse-distance weights, zeroing the self entry.
    inv = 1.0 / dist.replace(0.0, np.nan)
    for s in STATIONS:
        inv.loc[s, s] = 0.0
    weights = inv.div(inv.sum(axis=1), axis=0)  # rows sum to 1

    new = {p: {} for p in POLLUTANTS}
    for p in POLLUTANTS:
        wide = pivots[p]  # (T, 12)
        # Build per-station "neighbor" features by masking the target column.
        for s in STATIONS:
            others = wide.drop(columns=s, errors="ignore")
            new[p].setdefault("mean", {})[s] = others.mean(axis=1)
            new[p].setdefault("std",  {})[s] = others.std(axis=1)
            new[p].setdefault("max",  {})[s] = others.max(axis=1)
            new[p].setdefault("min",  {})[s] = others.min(axis=1)
            # Inverse-distance weighted mean.
            w = weights.loc[s, others.columns].to_numpy()
            new[p].setdefault("idw",  {})[s] = pd.Series(
                (others.to_numpy() * w).sum(axis=1),
                index=others.index,
            )

    # Reshape neighbor features back into long DataFrame keyed on (timestamp, station).
    parts = []
    for p in POLLUTANTS:
        for stat in ("mean", "std", "max", "min", "idw"):
            wide = pd.DataFrame(new[p][stat]).rename_axis(index="timestamp", columns=None)
            wide = wide.reset_index().melt(
                id_vars="timestamp", var_name="station",
                value_name=f"{p}_neigh{stat}_lag0",
            )
            parts.append(wide.set_index(["timestamp", "station"]))
    spatial = pd.concat(parts, axis=1).reset_index()

    df = df.merge(spatial, on=["timestamp", "station"], how="left")

    # 24-h-lagged neighbor mean.
    df = df.sort_values(["station", "timestamp"], kind="stable")
    for p in POLLUTANTS:
        df[f"{p}_neighmean_lag24"] = df.groupby("station")[f"{p}_neighmean_lag0"].shift(24)

    return df


def build_city_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-station mean of pollutants + weather at each timestamp.

    The returned DataFrame is single-station (`station = 'CITY'`) and feeds
    the city-level forecast path.
    """
    cols = POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    city = df.groupby("timestamp")[cols].mean().reset_index()
    # Wind direction averaged via vector components if available.
    if {"wind_u", "wind_v"}.issubset(df.columns):
        city[["wind_u", "wind_v"]] = (
            df.groupby("timestamp")[["wind_u", "wind_v"]].mean().reset_index()[["wind_u", "wind_v"]]
        )
    city["station"] = "CITY"
    return city
