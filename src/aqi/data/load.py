"""Load the 12 per-station CSVs into one long DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from aqi.config import POLLUTANTS, RAW_DIR, STATIONS, TIMEZONE, WEATHER_CATEGORICAL, WEATHER_NUMERIC

RAW_COLUMNS = (
    ["No", "year", "month", "day", "hour"]
    + POLLUTANTS
    + WEATHER_NUMERIC
    + WEATHER_CATEGORICAL
    + ["station"]
)


def _csv_path(station: str) -> Path:
    return RAW_DIR / f"PRSA_Data_{station}_20130301-20170228.csv"


def load_station(station: str) -> pd.DataFrame:
    """Load one station's CSV into a tidy DataFrame with a tz-aware timestamp."""
    path = _csv_path(station)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw CSV for station {station!r}: {path}. "
            f"Run scripts/01_download.py first."
        )

    df = pd.read_csv(path)
    # Some snapshots wrap PM2.5 in quotes; ensure consistent dtype.
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["timestamp"] = ts.dt.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    df["station"] = station

    keep = ["timestamp", "station"] + POLLUTANTS + WEATHER_NUMERIC + WEATHER_CATEGORICAL
    return df[keep].copy()


def load_all() -> pd.DataFrame:
    """Concatenate all 12 stations into one long DataFrame, hourly, tz-aware."""
    parts = [load_station(s) for s in STATIONS]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True)

    # Assert / enforce strict hourly cadence per station by reindexing.
    out_parts = []
    for s, sub in df.groupby("station", sort=False):
        full_idx = pd.date_range(
            sub["timestamp"].min(),
            sub["timestamp"].max(),
            freq="h",
            tz=TIMEZONE,
        )
        sub = sub.set_index("timestamp").reindex(full_idx)
        sub["station"] = s
        sub.index.name = "timestamp"
        out_parts.append(sub.reset_index())
    return pd.concat(out_parts, ignore_index=True)
