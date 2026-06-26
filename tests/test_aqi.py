"""EPA reference-value tests for AQI computation.

Reference values cross-checked against EPA-454/B-18-007 Table 5
and the AirNow AQI calculator example outputs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aqi.aqi.compute import (
    aqi_from_pollutants,
    subindex_co,
    subindex_no2,
    subindex_o3,
    subindex_pm10,
    subindex_pm25,
    subindex_so2,
    ugm3_to_ppb,
    ugm3_to_ppm,
)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
def test_no2_ppb_conversion():
    # 100 ug/m^3 NO2 -> ~53.13 ppb at 25 C / 1 atm
    assert ugm3_to_ppb(100, "NO2") == pytest.approx(53.139, rel=1e-3)


def test_co_ppm_conversion():
    # 1000 ug/m^3 CO -> ~0.873 ppm
    assert ugm3_to_ppm(1000, "CO") == pytest.approx(0.873, rel=1e-2)


# ---------------------------------------------------------------------------
# Per-pollutant sub-index boundaries
# ---------------------------------------------------------------------------
def test_pm25_boundary_100():
    # 35.4 ug/m^3 PM2.5 (24-h avg) is exactly AQI 100.
    assert subindex_pm25(35.4) == pytest.approx(100.0, abs=1.0)


def test_pm25_boundary_50():
    # 12.0 ug/m^3 PM2.5 is exactly AQI 50.
    assert subindex_pm25(12.0) == pytest.approx(50.0, abs=1.0)


def test_pm10_boundary_100():
    # 154 ug/m^3 PM10 is exactly AQI 100.
    assert subindex_pm10(154) == pytest.approx(100.0, abs=1.0)


def test_pm10_boundary_50():
    assert subindex_pm10(54) == pytest.approx(50.0, abs=1.0)


def test_o3_8h_boundary_100():
    # 0.070 ppm O3 8-h average = AQI 100. Reverse-convert to ug/m^3.
    ugm3 = 0.070 * 1000 * 48.00 / 24.45  # ~ 137.4 ug/m^3
    idx = subindex_o3(ugm3)
    assert idx == pytest.approx(100.0, abs=1.0)


def test_co_boundary_100():
    # 9.4 ppm CO 8-h average -> AQI 100. Convert ppm to ug/m^3.
    ugm3 = 9.4 * 1000 * 28.01 / 24.45
    assert subindex_co(ugm3) == pytest.approx(100.0, abs=1.0)


def test_no2_boundary_100():
    # 100 ppb NO2 = AQI 100. Convert to ug/m^3.
    ugm3 = 100 * 46.01 / 24.45
    assert subindex_no2(ugm3) == pytest.approx(100.0, abs=1.0)


def test_so2_boundary_100():
    # 75 ppb SO2 1-h = AQI 100.
    ugm3 = 75 * 64.07 / 24.45
    assert subindex_so2(ugm3) == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Aggregate AQI = max of sub-indices, dominant = argmax
# ---------------------------------------------------------------------------
def test_aqi_from_pollutants_pm25_dominant():
    aqi, cat, dom = aqi_from_pollutants(
        pm25=55.5, pm10=50, so2=10, no2=20, co=500, o3=80,
    )
    assert dom == "PM2.5"
    assert 140 < aqi <= 200
    assert cat in {"Unhealthy for Sensitive Groups", "Unhealthy"}


def test_aqi_nan_when_all_inputs_nan():
    aqi, cat, dom = aqi_from_pollutants(
        pm25=float("nan"), pm10=float("nan"), so2=float("nan"),
        no2=float("nan"), co=float("nan"), o3=float("nan"),
    )
    assert math.isnan(aqi)
    assert cat == "Unknown"
    assert dom == "None"


def test_aqi_dominant_handles_partial_nan():
    aqi, _, dom = aqi_from_pollutants(
        pm25=10, pm10=float("nan"), so2=5, no2=10, co=200, o3=200,
    )
    assert dom in {"PM2.5", "SO2", "NO2", "CO", "O3"}
    assert not math.isnan(aqi)


# ---------------------------------------------------------------------------
# Monotonicity sanity check
# ---------------------------------------------------------------------------
def test_pm25_monotone():
    values = np.linspace(0, 300, 50)
    indices = [subindex_pm25(v) for v in values]
    assert all(b >= a - 1e-6 for a, b in zip(indices, indices[1:]))
