"""Global configuration: paths, constants, station coordinates, seeds."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"
MODELS_DIR = PROJECT_ROOT / "models_store"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "evaluation"
FIGURES_DIR = REPORTS_DIR / "figures"

CLEANED_PARQUET = INTERIM_DIR / "cleaned_long.parquet"
TABULAR_PARQUET = PROCESSED_DIR / "tabular.parquet"
FORECAST_LOG = REPORTS_DIR / "forecast_log.parquet"

for _p in [
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    SEQUENCES_DIR,
    MODELS_DIR,
    EVAL_DIR,
    FIGURES_DIR,
]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stations (12 Beijing PRSA sites). Approximate WGS84 coordinates.
# ---------------------------------------------------------------------------
STATIONS = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]

STATION_COORDS: dict[str, tuple[float, float]] = {
    "Aotizhongxin":   (39.982, 116.397),
    "Changping":      (40.220, 116.230),
    "Dingling":       (40.290, 116.220),
    "Dongsi":         (39.929, 116.417),
    "Guanyuan":       (39.929, 116.339),
    "Gucheng":        (39.914, 116.184),
    "Huairou":        (40.328, 116.628),
    "Nongzhanguan":   (39.937, 116.461),
    "Shunyi":         (40.127, 116.655),
    "Tiantan":        (39.886, 116.407),
    "Wanliu":         (39.987, 116.287),
    "Wanshouxigong":  (39.878, 116.352),
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_NUMERIC = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WEATHER_CATEGORICAL = ["wd"]
COMPASS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Molar masses for ug/m3 → ppb (SO2/NO2/O3) / ug/m3 → ppm (CO).
MOLAR_MASS = {"SO2": 64.07, "NO2": 46.01, "O3": 48.00, "CO": 28.01}
MOLAR_VOLUME_25C_1ATM = 24.45  # L/mol at 25 C and 1 atm

# ---------------------------------------------------------------------------
# Forecast / split parameters
# ---------------------------------------------------------------------------
HORIZONS = list(range(1, 25))           # 1..24 hours ahead
SEQUENCE_LENGTH = 72                    # LSTM lookback hours

TRAIN_END   = "2015-12-31 23:00"
VAL_START   = "2016-01-01 00:00"
VAL_END     = "2016-06-30 23:00"
TEST_START  = "2016-07-01 00:00"
TEST_END    = "2017-02-28 23:00"
EMBARGO_HOURS = 168  # one week embargo around split boundaries

TIMEZONE = "Asia/Shanghai"

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed numpy, python random, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed; fine
        pass


@dataclass(frozen=True)
class Paths:
    """Convenience bundle of common paths."""
    raw: Path = RAW_DIR
    interim: Path = INTERIM_DIR
    processed: Path = PROCESSED_DIR
    sequences: Path = SEQUENCES_DIR
    models: Path = MODELS_DIR
    reports: Path = REPORTS_DIR
    eval: Path = EVAL_DIR
    figures: Path = FIGURES_DIR
