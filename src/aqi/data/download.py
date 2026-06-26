"""Download the Beijing Multi-Site Air-Quality dataset from UCI."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

from aqi.config import RAW_DIR, STATIONS

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)


def _expected_files() -> list[Path]:
    return [
        RAW_DIR / f"PRSA_Data_{s}_20130301-20170228.csv" for s in STATIONS
    ]


def already_downloaded() -> bool:
    return all(p.exists() for p in _expected_files())


def download(force: bool = False) -> list[Path]:
    """Fetch the UCI archive and extract the 12 station CSVs into RAW_DIR.

    Returns the list of CSV paths now on disk.
    """
    if not force and already_downloaded():
        print(f"[download] All 12 CSVs already present in {RAW_DIR}")
        return _expected_files()

    print(f"[download] Fetching {UCI_URL}")
    response = requests.get(UCI_URL, timeout=300)
    response.raise_for_status()

    outer = zipfile.ZipFile(io.BytesIO(response.content))
    # UCI ships a nested zip in some snapshots; handle both layouts.
    nested_zips = [n for n in outer.namelist() if n.lower().endswith(".zip")]
    if nested_zips:
        for nz in nested_zips:
            with outer.open(nz) as fh:
                inner = zipfile.ZipFile(io.BytesIO(fh.read()))
                _extract_csvs(inner)
    else:
        _extract_csvs(outer)

    missing = [p for p in _expected_files() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"After extraction, the following expected files are missing: {missing}"
        )
    print(f"[download] Wrote {len(_expected_files())} CSVs to {RAW_DIR}")
    return _expected_files()


def _extract_csvs(zf: zipfile.ZipFile) -> None:
    for name in zf.namelist():
        if name.lower().endswith(".csv") and "PRSA_Data_" in name:
            target = RAW_DIR / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
