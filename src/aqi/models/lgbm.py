"""LightGBM forecaster (per pollutant per horizon)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aqi.config import HORIZONS, POLLUTANTS, SEED
from aqi.models.base import Forecaster, target_columns


class LGBMForecaster(Forecaster):
    name = "lgbm"

    def __init__(
        self,
        n_estimators: int = 2000,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_data_in_leaf: int = 50,
        early_stopping_rounds: int = 50,
        random_state: int = SEED,
        n_jobs: int = -1,
    ):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for LGBMForecaster. "
                "Install with `pip install lightgbm`."
            ) from e
        self.params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_data_in_leaf=min_data_in_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        self.early_stopping_rounds = early_stopping_rounds
        # One booster per (pollutant, horizon). Reasonable for 6 * 24 = 144 models.
        self.models: dict[tuple[str, int], object] = {}
        self.feature_cols_: list[str] | None = None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        import lightgbm as lgb
        self.feature_cols_ = list(X_train.columns)

        for p in POLLUTANTS:
            for h in HORIZONS:
                target = f"y_{p}_h{h}"
                est = lgb.LGBMRegressor(**self.params)
                kwargs = {}
                callbacks = []
                if X_val is not None and Y_val is not None:
                    kwargs["eval_set"] = [(X_val.to_numpy(), Y_val[target].to_numpy())]
                    callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
                est.fit(
                    X_train.to_numpy(),
                    Y_train[target].to_numpy(),
                    callbacks=callbacks,
                    **kwargs,
                )
                self.models[(p, h)] = est
        return self

    def predict(self, X):
        X_arr = X[self.feature_cols_].to_numpy() if self.feature_cols_ else X.to_numpy()
        preds = np.zeros((X_arr.shape[0], len(POLLUTANTS) * len(HORIZONS)), dtype="float64")
        col = 0
        for p in POLLUTANTS:
            for h in HORIZONS:
                preds[:, col] = self.models[(p, h)].predict(X_arr)
                col += 1
        return preds

    def save(self, dir_path: Path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for (p, h), est in self.models.items():
            joblib.dump(est, dir_path / f"{p.replace('.', '_')}_h{h}.joblib")
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
        dir_path = Path(dir_path)
        manifest = json.loads((dir_path / "manifest.json").read_text())
        # `verbosity` is set internally by __init__, not a constructor arg.
        ctor_params = {k: v for k, v in manifest["params"].items() if k != "verbosity"}
        inst = cls(**{**ctor_params,
                      "early_stopping_rounds": manifest["early_stopping_rounds"]})
        for p in manifest["pollutants"]:
            for h in manifest["horizons"]:
                inst.models[(p, h)] = joblib.load(
                    dir_path / f"{p.replace('.', '_')}_h{h}.joblib"
                )
        inst.feature_cols_ = manifest["feature_cols"]
        return inst
