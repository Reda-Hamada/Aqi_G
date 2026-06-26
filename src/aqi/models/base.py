"""Forecaster ABC: every model family exposes the same interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from aqi.config import HORIZONS, POLLUTANTS


def target_columns(pollutants: list[str] = POLLUTANTS, horizons: list[int] = HORIZONS) -> list[str]:
    return [f"y_{p}_h{h}" for p in pollutants for h in horizons]


class Forecaster(ABC):
    """Predicts hourly pollutant concentrations 1..H hours ahead.

    fit(): trains on (X_train, Y_train) with optional (X_val, Y_val) early stopping.
    predict(): returns predicted Y (N, P*H) in the same column order as targets.
    save() / load(): persist artefacts.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        Y_val: pd.DataFrame | None = None,
    ) -> "Forecaster": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def save(self, dir_path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, dir_path: Path) -> "Forecaster": ...

    @property
    def target_cols(self) -> list[str]:
        return target_columns()
