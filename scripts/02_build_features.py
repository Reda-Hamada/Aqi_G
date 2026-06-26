#!/usr/bin/env python
"""Read cleaned long df → write tabular + sequence datasets."""
from __future__ import annotations

import argparse
import time

import pandas as pd

from aqi.aqi.compute import aqi_from_dataframe
from aqi.config import CLEANED_PARQUET, POLLUTANTS, SEQUENCE_LENGTH, STATIONS
from aqi.features.build import (
    build_sequences,
    build_tabular,
    feature_columns,
    write_sequences,
    write_tabular,
)


def main(
    spatial: bool = True,
    weather: bool = True,
    sequences: bool = True,
    sequence_length: int = SEQUENCE_LENGTH,
) -> None:
    t0 = time.perf_counter()
    print(f"[02_build_features] Reading {CLEANED_PARQUET}")
    df = pd.read_parquet(CLEANED_PARQUET)
    print(f"[02_build_features] Input: {df.shape}")

    print("[02_build_features] Computing hourly AQI labels...")
    df = aqi_from_dataframe(df)

    print(f"[02_build_features] Building tabular features "
          f"(spatial={spatial}, weather={weather})")
    tab = build_tabular(df, include_spatial=spatial, include_weather=weather)
    write_tabular(tab)

    if sequences:
        feats = feature_columns(tab)
        print(f"[02_build_features] Building sequences (T={sequence_length}, F={len(feats)})")
        for s in STATIONS:
            sub = tab[tab["station"] == s].copy()
            payload = build_sequences(sub, sequence_length=sequence_length, feature_cols=feats)
            write_sequences(payload, station=s)

    elapsed = time.perf_counter() - t0
    print(f"[02_build_features] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-sequences", action="store_true")
    ap.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    args = ap.parse_args()
    main(
        spatial=not args.no_spatial,
        weather=not args.no_weather,
        sequences=not args.no_sequences,
        sequence_length=args.sequence_length,
    )
