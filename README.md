# Beijing AQI Forecasting System

24-hour-ahead hourly AQI forecasts for Beijing using the
[Beijing Multi-Site Air-Quality dataset](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
(12 stations, 2013-03 → 2017-02). Compares four model families
(**Random Forest**, **XGBoost**, **LightGBM**, **LSTM**) on the same
train/validation/test split, with **spatial features from all 12 stations**
and **weather inputs** integrated into every pipeline.

See `specs/001-aqi-forecast-beijing/spec.md` for the full feature
specification.

## Project layout

```
src/aqi/
  config.py        # paths, constants, station coordinates, seeds
  data/            # download, load, clean, impute
  aqi/             # EPA breakpoints + AQI/category/dominant computation
  features/        # temporal, calendar, weather, spatial; orchestrator
  splits/          # chronological + walk-forward splits with embargo
  models/          # Forecaster ABC + RF / XGB / LGBM / LSTM / ensemble
  evaluate/        # metrics, baselines, runner, report
  forecast/        # snapshot -> features -> model -> AQI output
  persistence/     # run/model registry
scripts/           # CLI entry points (01_download, 02_build_features, ...)
tests/             # pytest suite (28 tests)
data/raw/          # downloaded CSVs (12 stations)
data/interim/      # cleaned_long.parquet
data/processed/    # tabular.parquet + sequences/*.npz
models_store/      # per-run model artefacts + manifests
reports/           # evaluation reports and figures
```

## Quickstart

```bash
# 1. Create a Python 3.10+ venv and install dependencies.
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the tests (synthetic data only; no network required).
pytest

# 3. Download the dataset and build the cleaned long parquet (~25 min CPU).
python scripts/01_download.py

# 4. Build features and sequences (~5-15 min).
python scripts/02_build_features.py

# 5. Train models. --quick uses a 5k-row subset for a smoke check.
python scripts/03_train_models.py --model lgbm --quick
python scripts/03_train_models.py --model all          # full run

# 6. Evaluate against the held-out test split.
python scripts/04_evaluate.py --run-id <RUN_ID_FROM_TRAIN>

# 7. Single inference.
python scripts/05_forecast.py --run-id <RUN_ID> \
    --model lgbm --station Aotizhongxin --asof 2017-01-15T08:00

# 8. Ablation studies (spatial-off, weather-off).
python scripts/06_run_ablations.py --model lgbm
```

## Forecast formulation

Because every hourly forecast must carry a **dominant pollutant** (FR-002,
acceptance scenario 1.1), the system predicts the six pollutant
concentrations individually and composes AQI from them via U.S. EPA
breakpoints (`src/aqi/aqi/compute.py`). For each row at time t and station s,
the targets are:

```
y_{p,h} = p^{s}_{t+h}   for p ∈ {PM2.5, PM10, SO2, NO2, CO, O3}
                      and h ∈ {1..24}
```

→ 144 targets per row.

Trees produce one estimator per pollutant (RF/XGB) or per (pollutant, horizon)
(LightGBM). The LSTM uses an encoder + dense head that emits the full
(24 horizons × 6 pollutants) tensor in one shot.

All four model families share the same `Forecaster` interface
(`src/aqi/models/base.py`), so the evaluation harness iterates them in a
single loop.

## Time-series validation

Chronological split — random k-fold is **forbidden**:

| split | window               |
|-------|----------------------|
| train | 2013-03 → 2015-12    |
| val   | 2016-01 → 2016-06    |
| test  | 2016-07 → 2017-02    |

A one-week embargo around each boundary prevents lag features from leaking
across splits. Inner walk-forward folds on the training window are used for
hyperparameter tuning (`src/aqi/splits/walk_forward.py`).

## Evaluation

For each model:

- **MAE / RMSE / MAPE** on each pollutant × horizon
- **MAE on AQI value**, **category accuracy**, **macro-F1**
- **Per-horizon MAE curve** for hour 1..24
- **Per-station MAE table**
- Comparison against three naive baselines: persistence, climatology, and
  seasonal naive (week-ago)

Ablations (`scripts/06_run_ablations.py`) quantify:

- **SC-005**: spatial features (all 12 stations vs. target station only)
- **SC-006**: weather feature contribution

Every forecast is logged to `reports/forecast_log.parquet` with the model
run-id and an input snapshot hash, so any reported forecast can be
reproduced bit-for-bit (FR-014, SC-008).

## Testing

```bash
pytest -q
```

Covers AQI breakpoint correctness against EPA reference values, feature
no-leakage (mutating a future value cannot change a past row's rolling
stats), chronological-split ordering, baseline correctness, and end-to-end
RF train+forecast on synthetic data — no network required.
