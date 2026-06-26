"""U.S. EPA AQI breakpoint tables.

Reference: EPA-454/B-18-007 "Technical Assistance Document for the Reporting of
Daily Air Quality" (Sept 2018), Table 5.

Each entry: (C_low, C_high, I_low, I_high) in the pollutant's native EPA unit.
- PM2.5, PM10:   ug/m^3  (24-hour avg)
- O3 (8-hr):     ppm     (8-hour avg)
- O3 (1-hr):     ppm     (1-hour avg)  -- only used for AQI >= 101
- CO:            ppm     (8-hour avg)
- SO2 (1-hr):    ppb     (1-hour avg)  -- 1-hour is used through index 200
- SO2 (24-hr):   ppb     (24-hour avg) -- only index >= 201
- NO2 (1-hr):    ppb     (1-hour avg)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Bp = Tuple[float, float, int, int]


PM25: List[Bp] = [
    (0.0,   12.0,    0, 50),
    (12.1,  35.4,   51, 100),
    (35.5,  55.4,  101, 150),
    (55.5, 150.4,  151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10: List[Bp] = [
    (0,    54,    0, 50),
    (55,  154,   51, 100),
    (155, 254,  101, 150),
    (255, 354,  151, 200),
    (355, 424,  201, 300),
    (425, 504,  301, 400),
    (505, 604,  401, 500),
]

O3_8H: List[Bp] = [
    (0.000, 0.054,   0,  50),
    (0.055, 0.070,  51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
    # 8-hour O3 not defined > 300; fall back to 1-hour O3.
]

O3_1H: List[Bp] = [
    (0.125, 0.164, 101, 150),
    (0.165, 0.204, 151, 200),
    (0.205, 0.404, 201, 300),
    (0.405, 0.504, 301, 400),
    (0.505, 0.604, 401, 500),
]

CO: List[Bp] = [
    (0.0,   4.4,    0,  50),
    (4.5,   9.4,   51, 100),
    (9.5,  12.4,  101, 150),
    (12.5, 15.4,  151, 200),
    (15.5, 30.4,  201, 300),
    (30.5, 40.4,  301, 400),
    (40.5, 50.4,  401, 500),
]

SO2_1H: List[Bp] = [
    (0,    35,    0,  50),
    (36,   75,   51, 100),
    (76,  185,  101, 150),
    (186, 304,  151, 200),
    # > 200 uses 24-hour SO2.
]

SO2_24H: List[Bp] = [
    (305, 604,  201, 300),
    (605, 804,  301, 400),
    (805, 1004, 401, 500),
]

NO2_1H: List[Bp] = [
    (0,    53,    0,  50),
    (54,  100,   51, 100),
    (101, 360,  101, 150),
    (361, 649,  151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

CATEGORIES: List[Tuple[int, int, str]] = [
    (0,   50,  "Good"),
    (51,  100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def aqi_to_category(aqi: float) -> str:
    """Map an AQI value to its EPA category."""
    if aqi != aqi:  # NaN
        return "Unknown"
    a = int(round(aqi))
    for lo, hi, name in CATEGORIES:
        if lo <= a <= hi:
            return name
    return "Hazardous" if a > 500 else "Unknown"


ALL_TABLES: Dict[str, List[Bp]] = {
    "PM2.5":   PM25,
    "PM10":    PM10,
    "O3_8H":   O3_8H,
    "O3_1H":   O3_1H,
    "CO":      CO,
    "SO2_1H":  SO2_1H,
    "SO2_24H": SO2_24H,
    "NO2":     NO2_1H,
}
