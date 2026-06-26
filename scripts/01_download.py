#!/usr/bin/env python
"""End-to-end data ingestion: download → load → clean → impute → parquet."""
from __future__ import annotations

import sys
import time

import pandas as pd

from aqi.config import CLEANED_PARQUET
from aqi.data.clean import clean
from aqi.data.download import download
from aqi.data.impute import impute
from aqi.data.load import load_all


def main(force_download: bool = False) -> None:
    t0 = time.perf_counter()
    download(force=force_download)
    print("[01_download] Loading 12 station CSVs...")
    df = load_all()
    print(f"[01_download] Loaded {len(df):,} rows for {df['station'].nunique()} stations")

    print("[01_download] Clipping out-of-range values to NaN...")
    df = clean(df)

    print("[01_download] Imputing short/medium gaps...")
    df = impute(df)

    print(f"[01_download] Writing {CLEANED_PARQUET}")
    df.to_parquet(CLEANED_PARQUET, index=False)
    elapsed = time.perf_counter() - t0
    print(f"[01_download] Done in {elapsed:.1f}s. Final shape: {df.shape}")


if __name__ == "__main__":
    main(force_download="--force" in sys.argv)
