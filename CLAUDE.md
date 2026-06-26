<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

# CLAUDE.md — Beijing AQI Forecasting

## What this project is

A research-mode AQI forecasting system for Beijing that compares Random Forest,
XGBoost, LightGBM, and an LSTM on the same 24-hour-ahead forecast task using
the Beijing Multi-Site Air-Quality dataset (12 stations, 2013-03 → 2017-02).

Specification: `specs/001-aqi-forecast-beijing/spec.md`.
Implementation plan: `/home/reda/.claude/plans/create-a-detailed-implementation-zesty-hickey.md`.

## Project map

```
src/aqi/
  config.py        # paths, station list/coordinates, constants, seeds
  data/            # download + load + clean + impute
  aqi/             # EPA AQI breakpoints + computation
  features/        # temporal/calendar/weather/spatial + orchestrator (build.py)
  splits/          # chronological + walk-forward splits with 1-week embargo
  models/          # Forecaster ABC + RF/XGB/LGBM/LSTM/ensemble wrappers
  evaluate/        # metrics, baselines, runner, report
  forecast/        # snapshot -> features -> model -> AQI output JSON
  persistence/     # run/model registry with manifest.json
scripts/           # 01_download → 06_run_ablations CLIs
tests/             # 28 pytest tests, no network required
```

## Conventions

- **Python ≥ 3.10**, type-hinted, package installed editable from `src/`.
- **Single timezone**: all timestamps are `Asia/Shanghai` (UTC+8).
- **No future leakage**: lag/rolling features always use `groupby().shift(1)`
  before `rolling()`. The leakage test (`tests/test_features.py`) enforces this.
- **One AQI standard**: U.S. EPA breakpoints in `src/aqi/aqi/breakpoints.py`.
  Reference values for boundary checks are encoded in `tests/test_aqi.py`.
- **One Forecaster interface**: every model exposes `fit / predict / save / load`
  (`src/aqi/models/base.py`), so the eval runner stays in one loop.
- **Two prediction shapes**: tabular models output `(N, P*H) = (N, 144)`.
  The LSTM uses sequence inputs and outputs `(N, H, P) = (N, 24, 6)`.
- **Splits**: chronological only. `chronological_masks(df)` in
  `src/aqi/splits/walk_forward.py` is the single source of truth.
- **Run IDs**: `models_store/<YYYYMMDDTHHMMSS>_<git_sha>/` per training run.
  Every artefact stores a `manifest.json`.

## Running things

```bash
source .venv/bin/activate
pytest                                        # full test suite
python scripts/01_download.py                 # one-time, ~25 min
python scripts/02_build_features.py
python scripts/03_train_models.py --model all
python scripts/04_evaluate.py --run-id <id>
```

## Where to add code

- New feature → add to `src/aqi/features/<area>.py` AND register inputs in
  `src/aqi/features/build.py:LAG_COLUMNS`. Add a leakage test if it uses
  history.
- New model family → subclass `Forecaster` in `src/aqi/models/`, implement
  `fit/predict/save/load`, and add a `_load_model` branch in
  `scripts/04_evaluate.py` and `scripts/05_forecast.py`.
- New baseline → add to `src/aqi/evaluate/baselines.py` and append to the
  baselines loop in `scripts/04_evaluate.py`.
