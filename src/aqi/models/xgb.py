"""XGBoost forecaster (per pollutant, multi-output across horizons)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aqi.config import HORIZONS, POLLUTANTS, SEED
from aqi.models.base import Forecaster, target_columns


class XGBForecaster(Forecaster):
    name = "xgb"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        early_stopping_rounds: int = 50,
        random_state: int = SEED,
    ):
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "xgboost is required for XGBForecaster. "
                "Install with `pip install xgboost`."
            ) from e
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            random_state=random_state,
            multi_strategy="multi_output_tree",
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.models: dict = {}
        self.feature_cols_: list[str] | None = None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        from xgboost import XGBRegressor
        self.feature_cols_ = list(X_train.columns)

        for p in POLLUTANTS:
            cols = [f"y_{p}_h{h}" for h in HORIZONS]
            est = XGBRegressor(**self.params)
            fit_kwargs = {}
            if X_val is not None and Y_val is not None:
                fit_kwargs["eval_set"] = [(X_val.to_numpy(), Y_val[cols].to_numpy())]
                # newer xgboost requires early_stopping_rounds on the constructor;
                # accept either path.
                try:
                    est.set_params(early_stopping_rounds=self.early_stopping_rounds)
                except Exception:
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            est.fit(X_train.to_numpy(), Y_train[cols].to_numpy(), **fit_kwargs)
            self.models[p] = est
        return self

    def predict(self, X):
        X_arr = X[self.feature_cols_].to_numpy() if self.feature_cols_ else X.to_numpy()
        return np.concatenate([self.models[p].predict(X_arr) for p in POLLUTANTS], axis=1)

    def save(self, dir_path: Path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for p, est in self.models.items():
            est.save_model(str(dir_path / f"{p.replace('.', '_')}.json"))
        manifest = {
            "name": self.name,
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "pollutants": POLLUTANTS,
            "horizons": HORIZONS,
            "feature_cols": self.feature_cols_,
            "target_cols": target_columns(),
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, dir_path: Path):
        from xgboost import XGBRegressor
        dir_path = Path(dir_path)
        manifest = json.loads((dir_path / "manifest.json").read_text())
        # `multi_strategy` is set internally by __init__, not a constructor arg.
        ctor_params = {k: v for k, v in manifest["params"].items() if k != "multi_strategy"}
        inst = cls(**{**ctor_params,
                      "early_stopping_rounds": manifest["early_stopping_rounds"]})
        for p in manifest["pollutants"]:
            est = XGBRegressor()
            est.load_model(str(dir_path / f"{p.replace('.', '_')}.json"))
            inst.models[p] = est
        inst.feature_cols_ = manifest["feature_cols"]
        return inst
