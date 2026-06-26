"""Random Forest forecaster.

Trains one MultiOutputRegressor sharing trees across horizons per pollutant.
Inputs must be NaN-free, so callers should impute or drop before fit().
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from aqi.config import HORIZONS, POLLUTANTS, SEED
from aqi.models.base import Forecaster, target_columns


class RandomForestForecaster(Forecaster):
    name = "rf"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        max_features: str | float = "sqrt",
        n_jobs: int = -1,
        random_state: int = SEED,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        # One regressor per pollutant -> outputs all 24 horizons together.
        self.models: dict[str, RandomForestRegressor] = {}
        self.feature_cols_: list[str] | None = None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        self.feature_cols_ = list(X_train.columns)
        for p in POLLUTANTS:
            cols = [f"y_{p}_h{h}" for h in HORIZONS]
            est = RandomForestRegressor(**self.params)
            est.fit(X_train.to_numpy(), Y_train[cols].to_numpy())
            self.models[p] = est
        return self

    def predict(self, X):
        X_arr = X[self.feature_cols_].to_numpy() if self.feature_cols_ else X.to_numpy()
        preds = []
        for p in POLLUTANTS:
            preds.append(self.models[p].predict(X_arr))  # (N, H)
        return np.concatenate(preds, axis=1)  # (N, P*H)

    def save(self, dir_path: Path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for p, est in self.models.items():
            joblib.dump(est, dir_path / f"{p.replace('.', '_')}.joblib")
        manifest = {
            "name": self.name,
            "params": self.params,
            "pollutants": POLLUTANTS,
            "horizons": HORIZONS,
            "feature_cols": self.feature_cols_,
            "target_cols": target_columns(),
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, dir_path: Path):
        dir_path = Path(dir_path)
        manifest = json.loads((dir_path / "manifest.json").read_text())
        inst = cls(**manifest["params"])
        for p in manifest["pollutants"]:
            inst.models[p] = joblib.load(dir_path / f"{p.replace('.', '_')}.joblib")
        inst.feature_cols_ = manifest["feature_cols"]
        return inst
