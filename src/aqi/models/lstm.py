"""LSTM forecaster: encoder + multi-horizon dense head.

Inputs are sequence arrays produced by features.build.build_sequences:
  X: (N, T, F)  — historical features over T hours
  y: (N, H, P)  — future pollutant concentrations over H horizons, P pollutants
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aqi.config import HORIZONS, POLLUTANTS, SEED, SEQUENCE_LENGTH


class _LSTMNet:
    """Lazy torch import: builds an LSTM encoder + dense head when constructed."""

    def __init__(self, n_features: int, hidden: int = 128, n_stations: int = 13, station_emb: int = 8):
        import torch
        import torch.nn as nn
        self.torch = torch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.station_emb = nn.Embedding(n_stations, station_emb)
                self.lstm = nn.LSTM(
                    input_size=n_features + station_emb,
                    hidden_size=hidden,
                    num_layers=2,
                    dropout=0.2,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(HORIZONS) * len(POLLUTANTS)),
                )

            def forward(self, x, station_id):
                # x: (B, T, F);  station_id: (B,)
                emb = self.station_emb(station_id)            # (B, E)
                emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
                inp = self.torch_cat([x, emb], dim=-1)        # (B, T, F+E)
                out, (h, _) = self.lstm(inp)
                z = h[-1]                                     # (B, hidden)
                y = self.head(z)                              # (B, H*P)
                return y.view(-1, len(HORIZONS), len(POLLUTANTS))

            def torch_cat(self, tensors, dim):
                import torch as _t
                return _t.cat(tensors, dim=dim)

        self.net = Net()


class LSTMForecaster:
    """Wrapper exposing the Forecaster-like interface for sequence inputs.

    Note: this class deliberately does NOT inherit from `Forecaster` because the
    tabular signature does not fit a sequence model; the evaluation runner
    branches on `inst.kind == 'sequence'`.
    """

    name = "lstm"
    kind = "sequence"

    def __init__(
        self,
        n_features: int,
        n_stations: int = 13,
        hidden: int = 128,
        station_emb: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 5,
        device: str | None = None,
        random_state: int = SEED,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch is required for LSTMForecaster. "
                "Install with `pip install torch`."
            ) from e
        self.params = dict(
            n_features=n_features,
            n_stations=n_stations,
            hidden=hidden,
            station_emb=station_emb,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            random_state=random_state,
        )
        self.device = device
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_std_: np.ndarray | None = None
        self.station_lookup_: dict[str, int] = {}
        self.model = None  # type: ignore[assignment]

    def _device(self):
        import torch
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(self, arr, dtype=None):
        import torch
        t = torch.as_tensor(arr)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            flat = X.reshape(-1, X.shape[-1])
            self.scaler_mean_ = flat.mean(axis=0)
            self.scaler_std_ = flat.std(axis=0)
            self.scaler_std_[self.scaler_std_ == 0] = 1.0
        # Cast stats to X's dtype so float32 inputs are not promoted to float64
        # (a full-tensor float64 copy OOMs on large window sets).
        mean = self.scaler_mean_.astype(X.dtype, copy=False)
        std = self.scaler_std_.astype(X.dtype, copy=False)
        return (X - mean) / std

    def _station_codes(self, names: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.station_lookup_ = {s: i for i, s in enumerate(sorted(set(names.tolist())))}
        return np.array([self.station_lookup_[s] for s in names], dtype="int64")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        station_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        station_val: np.ndarray | None = None,
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.params["random_state"])
        Xs = self._scale(X_train, fit=True).astype("float32")
        sids = self._station_codes(station_train, fit=True)

        n_stations = max(self.params["n_stations"], len(self.station_lookup_))
        self.params["n_stations"] = n_stations

        net_wrap = _LSTMNet(
            n_features=self.params["n_features"],
            hidden=self.params["hidden"],
            n_stations=n_stations,
            station_emb=self.params["station_emb"],
        )
        self.model = net_wrap.net.to(self._device())
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        loss_fn = nn.MSELoss()

        ds = TensorDataset(
            self._to_tensor(Xs, torch.float32),
            self._to_tensor(sids, torch.long),
            self._to_tensor(y_train.astype("float32"), torch.float32),
        )
        dl = DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True, drop_last=False)

        val_dl = None
        if X_val is not None and y_val is not None and station_val is not None:
            Xv = self._scale(X_val).astype("float32")
            sv = self._station_codes(station_val)
            val_dl = DataLoader(
                TensorDataset(
                    self._to_tensor(Xv, torch.float32),
                    self._to_tensor(sv, torch.long),
                    self._to_tensor(y_val.astype("float32"), torch.float32),
                ),
                batch_size=self.params["batch_size"],
            )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        for epoch in range(self.params["epochs"]):
            self.model.train()
            for xb, sb, yb in dl:
                xb = xb.to(self._device()); sb = sb.to(self._device()); yb = yb.to(self._device())
                opt.zero_grad()
                pred = self.model(xb, sb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            if val_dl is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    n = 0
                    for xb, sb, yb in val_dl:
                        xb = xb.to(self._device()); sb = sb.to(self._device()); yb = yb.to(self._device())
                        val_loss += loss_fn(self.model(xb, sb), yb).item() * xb.size(0)
                        n += xb.size(0)
                    val_loss /= max(n, 1)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.params["patience"]:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray, station: np.ndarray) -> np.ndarray:
        """Returns (N, H, P) predictions."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        self.model.eval()
        Xs = self._scale(X).astype("float32")
        sids = self._station_codes(station)
        ds = TensorDataset(self._to_tensor(Xs, torch.float32), self._to_tensor(sids, torch.long))
        dl = DataLoader(ds, batch_size=self.params["batch_size"])
        outs = []
        with torch.no_grad():
            for xb, sb in dl:
                xb = xb.to(self._device()); sb = sb.to(self._device())
                outs.append(self.model(xb, sb).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def save(self, dir_path: Path):
        import torch
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), dir_path / "model.pt")
        np.savez(
            dir_path / "scaler.npz",
            mean=self.scaler_mean_,
            std=self.scaler_std_,
        )
        manifest = {
            "name": self.name,
            "kind": self.kind,
            "params": self.params,
            "station_lookup": self.station_lookup_,
            "pollutants": POLLUTANTS,
            "horizons": HORIZONS,
            "sequence_length": SEQUENCE_LENGTH,
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, dir_path: Path):
        import torch
        dir_path = Path(dir_path)
        manifest = json.loads((dir_path / "manifest.json").read_text())
        inst = cls(**manifest["params"])
        inst.station_lookup_ = manifest["station_lookup"]
        s = np.load(dir_path / "scaler.npz")
        inst.scaler_mean_ = s["mean"]
        inst.scaler_std_ = s["std"]
        net_wrap = _LSTMNet(
            n_features=manifest["params"]["n_features"],
            hidden=manifest["params"]["hidden"],
            n_stations=manifest["params"]["n_stations"],
            station_emb=manifest["params"]["station_emb"],
        )
        inst.model = net_wrap.net.to(inst._device())
        inst.model.load_state_dict(torch.load(dir_path / "model.pt", map_location=inst._device()))
        return inst
