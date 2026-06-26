"""Simple weighted-average ensemble of base forecasters' predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EnsembleAverage:
    """Average predictions from several base models, optionally weighted.

    Operates purely on prediction arrays. Persistence is just a weights file;
    callers must independently load the base models.
    """

    name = "ensemble"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}

    def predict(self, base_preds: dict[str, np.ndarray]) -> np.ndarray:
        names = list(base_preds.keys())
        w = np.array([self.weights.get(n, 1.0) for n in names], dtype="float64")
        w = w / w.sum()
        stacked = np.stack([base_preds[n] for n in names], axis=0)  # (M, N, T)
        return np.tensordot(w, stacked, axes=(0, 0))

    def save(self, dir_path: Path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "manifest.json").write_text(json.dumps(
            {"name": self.name, "weights": self.weights}, indent=2
        ))

    @classmethod
    def load(cls, dir_path: Path):
        manifest = json.loads(Path(dir_path, "manifest.json").read_text())
        return cls(weights=manifest["weights"])
