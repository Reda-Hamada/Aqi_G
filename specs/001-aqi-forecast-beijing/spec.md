# Feature Specification: Beijing AQI Forecasting System

**Feature Branch**: `001-aqi-forecast-beijing`

**Created**: 2026-06-17

**Status**: Draft

**Input**: User description: "Build an AQI forecasting system for Beijing using AQI calculation, Random Forest, XGBoost, LightGBM, LSTM, spatial features from 12 stations, and weather data."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - City-Wide AQI Forecast for Beijing (Priority: P1)

A Beijing resident, environmental researcher, or public health official wants to know what the air quality will be in Beijing over the next 24 hours so that they can plan outdoor activities, issue health advisories, or study air pollution patterns. They open the forecasting system, request a forecast for Beijing, and receive an hourly AQI prediction along with the dominant pollutant category (e.g., "Unhealthy — PM2.5 driven").

**Why this priority**: This is the core value proposition. Without a credible, city-level AQI forecast there is no product. Every other capability (model comparison, station-level breakdown, historical analysis) is an enhancement on top of this primary forecast.

**Independent Test**: Can be fully tested by submitting the most recent observed pollutant + weather snapshot to the system and verifying it produces an hourly AQI forecast covering the next 24 hours, with each hour reporting (a) a numeric AQI value, (b) an AQI category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy / Very Unhealthy / Hazardous), and (c) the dominant pollutant. Forecast accuracy can be evaluated against a held-out historical period.

**Acceptance Scenarios**:

1. **Given** a complete observation snapshot of recent pollutant and weather conditions for Beijing, **When** the user requests a 24-hour AQI forecast, **Then** the system returns 24 hourly AQI values, each with category label and dominant pollutant, plus a single overall daily AQI summary.
2. **Given** historical data has been processed and models have been trained, **When** the user evaluates the forecast against actual measurements from a held-out test period, **Then** the system reports accuracy metrics (e.g., mean absolute error, category-classification accuracy) for the forecast.
3. **Given** the most recent station readings include partial gaps (one or two pollutants missing for a few hours), **When** the user requests a forecast, **Then** the system still produces a forecast and indicates which inputs were imputed or how confidence may be affected.

---

### User Story 2 - Compare Forecasting Models to Select the Best One (Priority: P2)

A data scientist or researcher wants to understand which forecasting approach performs best for Beijing AQI so they can recommend the most reliable model (or ensemble) for production use. They run the system in evaluation mode, see side-by-side performance of multiple forecasting approaches on the same test data, and select a "winning" model based on objective metrics.

**Why this priority**: Building confidence in the forecast requires showing that the chosen approach outperforms alternatives. This story turns the system from a black box into a defensible recommendation and is essential for academic / publication contexts, but is not required to deliver a working forecast.

**Independent Test**: Can be fully tested by running an evaluation over a fixed historical test window and verifying the system produces a comparison report that ranks each approach by at least two error metrics and one classification metric, along with per-pollutant breakdown.

**Acceptance Scenarios**:

1. **Given** a fixed train/test split of historical Beijing data, **When** the user runs model evaluation, **Then** the system produces a comparison table showing each candidate model's forecast error (e.g., MAE, RMSE) and category-classification accuracy on the same test set.
2. **Given** the comparison report from the prior step, **When** the user selects the best-performing model as the production forecaster, **Then** subsequent forecast requests use that selected model and the choice is recorded for auditability.
3. **Given** a single chosen model significantly underperforms an ensemble of multiple models, **When** the user inspects the evaluation, **Then** the system clearly shows that the ensemble offers a measurable improvement (or does not) so the user can decide whether to forecast with a single model or an ensemble.

---

### User Story 3 - Station-Level Forecasts and Spatial Insight (Priority: P3)

A user investigating local air quality differences across Beijing (e.g., a parent choosing a school district, a planner studying pollution hotspots) wants to see how the forecast differs between individual monitoring stations rather than only a city average. They request a forecast for a specific station or compare several, and observe how using readings from neighboring stations improves predictions over relying on one station alone.

**Why this priority**: Granular, spatially-aware forecasting is a meaningful improvement over a single city number, but most users will be satisfied with the city forecast (P1). This story extends the system once the core forecast is reliable and demonstrates the spatial value of the 12-station network.

**Independent Test**: Can be fully tested by requesting forecasts for two named stations on the same day, verifying each returns its own 24-hour hourly AQI series, and that an ablation test (forecasting station X using only X's own data vs. using all 12 stations) shows a measurable accuracy difference on a held-out period.

**Acceptance Scenarios**:

1. **Given** the user specifies one of the 12 monitoring stations, **When** they request a 24-hour AQI forecast, **Then** the system returns the forecast specific to that station along with a confidence indicator.
2. **Given** the user requests forecasts for all 12 stations, **When** the request completes, **Then** the system returns 12 station-level forecast series plus an aggregated city-level series, and the relationship between them is reproducible.
3. **Given** an evaluation comparing "single-station" vs. "spatial (all-stations)" feature configurations on the same test window, **When** the user views results, **Then** the report quantifies the accuracy gain (or loss) from including spatial features.

---

### Edge Cases

- **Missing or partial observations**: How does the system behave when one or more of the 12 stations is offline for several hours, or when specific pollutant or weather columns are missing in the input snapshot?
- **Extreme pollution events**: How does the forecast behave during rare, severe smog episodes that fall outside the bulk of training data (heavy-tail AQI values)? Confidence must be communicated.
- **Out-of-range inputs**: How does the system respond when weather or pollutant values fall outside historical ranges (e.g., temperature far below training minimum)?
- **Daylight savings / timezone shifts**: How are hourly timestamps handled around clock changes so the 24-hour horizon is unambiguous?
- **Model staleness**: How does the system behave when the production model has not been retrained for a long period and recent pollution patterns may have drifted?
- **Forecast at horizon limits**: Forecast quality at hour 24 will be lower than at hour 1; the system must communicate uncertainty growth across the horizon.
- **Station relocation or new data sources**: What happens when one of the 12 stations is decommissioned or a new station is added — does the system still produce a forecast?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest historical hourly observations of air pollutants (PM2.5, PM10, SO2, NO2, CO, O3) and meteorological variables (temperature, pressure, dew point, precipitation, wind speed, wind direction) from each of the 12 Beijing monitoring stations covering at least 4 years of data.
- **FR-002**: System MUST compute the Air Quality Index (AQI) and the dominant pollutant for any given hourly pollutant snapshot following an established AQI breakpoint standard (e.g., U.S. EPA or Chinese MEP), and record which standard is used.
- **FR-003**: System MUST classify any AQI value into a category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy / Very Unhealthy / Hazardous) based on the chosen AQI standard.
- **FR-004**: System MUST produce a 24-hour-ahead hourly AQI forecast for Beijing at the city level, including category and dominant pollutant for each hour.
- **FR-005**: System MUST support forecasts at the individual-station level for each of the 12 Beijing monitoring stations.
- **FR-006**: System MUST train and evaluate four distinct forecasting model families — Random Forest, XGBoost, LightGBM, and an LSTM-based deep learning model — on the same train/validation/test split so they are directly comparable.
- **FR-007**: System MUST construct spatial features that incorporate observations from neighboring stations (not only the target station) when forecasting a given station or the city average.
- **FR-008**: System MUST incorporate weather features (current and lagged) as inputs to every forecasting model, and the contribution of weather inputs MUST be evaluable through a feature-removal ablation.
- **FR-009**: System MUST produce a model-comparison report for any chosen evaluation window that includes, per model, at least one error metric (e.g., MAE or RMSE) on the AQI value and one classification metric (e.g., category accuracy) on the AQI category.
- **FR-010**: System MUST persist the trained model artifacts and the corresponding evaluation metrics together, so that any reported forecast can be traced back to a specific model version and training run.
- **FR-011**: System MUST handle missing values in input observations through a documented imputation strategy and MUST flag forecasts whose inputs required substantial imputation.
- **FR-012**: System MUST validate that observation timestamps form a regular hourly series in a known timezone, and MUST reject or repair inputs that violate this contract.
- **FR-013**: System MUST allow a user to re-run training on an updated historical window (e.g., when new months of data become available) and produce a new comparison report without overwriting prior model artifacts.
- **FR-014**: System MUST log every forecast request along with the model version used, the input snapshot identifier, and the forecast output, to support reproducibility and post-hoc accuracy review.
- **FR-015**: System MUST clearly indicate the units, the AQI standard, the timezone, and the forecast issue time on every forecast output.

### Key Entities *(include if feature involves data)*

- **Monitoring Station**: One of the 12 Beijing air-quality monitoring stations. Has a stable identifier, a geographic location, and produces hourly readings. Spatial relationships between stations matter for feature construction.
- **Pollutant Reading**: An hourly observation at a station containing concentrations of PM2.5, PM10, SO2, NO2, CO, and O3. Forms the basis for AQI computation.
- **Weather Reading**: An hourly observation at (or associated with) a station containing temperature, pressure, dew point, precipitation, wind speed, and wind direction.
- **AQI Value**: A computed scalar derived from pollutant concentrations at an hourly timestamp; carries a category label and an identified dominant pollutant.
- **Forecast**: An ordered sequence of hourly AQI values, categories, and dominant pollutants for a future window (up to 24 hours), produced for either a station or the city. Linked to the model version, the issue time, and the input snapshot.
- **Model**: A trained predictor (Random Forest, XGBoost, LightGBM, or LSTM) with a version, a training configuration, a training data window, and a recorded evaluation metric set.
- **Evaluation Report**: The output of running candidate models on a held-out window. Holds per-model, per-pollutant, and per-horizon metrics so models can be ranked.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a 24-hour-ahead Beijing city forecast on a held-out test period, the chosen production model achieves a Mean Absolute Error on the AQI value materially lower than a naive persistence baseline (i.e., "tomorrow equals today"), with the percent improvement reported in the evaluation report.
- **SC-002**: For the 6-category AQI classification on the same held-out test period, the chosen production model correctly identifies the category in at least the majority of hours, and clearly outperforms a baseline of always predicting the most-common category.
- **SC-003**: Forecast accuracy is reported separately for short horizons (hour 1–6) and longer horizons (hour 19–24), and the short-horizon accuracy is meaningfully better than the long-horizon accuracy, confirming the system communicates uncertainty growth honestly.
- **SC-004**: The model-comparison report covers all four model families (Random Forest, XGBoost, LightGBM, LSTM) on the same data split and clearly ranks them by at least two metrics so a non-expert reader can identify the winner without re-running anything.
- **SC-005**: An ablation comparing "single-station only" features vs. "all 12 stations" spatial features is documented and quantifies the accuracy contribution of the spatial feature set on the same test window.
- **SC-006**: An ablation removing weather inputs is documented and quantifies the accuracy contribution of weather features on the same test window.
- **SC-007**: A user can request a forecast for the city or any of the 12 stations and receive a result within a few seconds once models are trained (i.e., inference is fast enough for interactive use, distinct from offline training time).
- **SC-008**: Every forecast output is traceable: given the recorded model version and input snapshot identifier, the same forecast can be regenerated, supporting reproducibility for research and audit.

## Assumptions

- **Geographic scope**: Forecasts are for Beijing only and use exactly the 12 well-known Beijing monitoring stations (e.g., Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong) consistent with the publicly available Beijing Multi-Site Air-Quality dataset (2013–2017).
- **Temporal resolution**: Inputs and forecasts are hourly. The forecast horizon is 24 hours ahead unless explicitly extended in a future iteration.
- **Pollutants in scope**: PM2.5, PM10, SO2, NO2, CO, and O3 — the standard six pollutants required for AQI computation. Other pollutants are out of scope.
- **Weather features in scope**: Temperature, atmospheric pressure, dew point, precipitation, wind speed, and wind direction. Additional meteorological inputs (e.g., solar radiation, boundary-layer height) are out of scope for v1.
- **AQI standard**: The U.S. EPA AQI breakpoint standard is the default for computation; the Chinese MEP standard may be supported as an alternative but is not required for v1.
- **Use mode**: The system operates primarily in batch evaluation / research mode against historical data. A real-time data feed and a public-facing UI are out of scope for v1; the forecast API/interface is sufficient for evaluation and demonstration.
- **Users**: Primary users are researchers, data scientists, and analysts. A general-public consumer interface is out of scope for v1.
- **Model family scope**: Exactly the four model families (Random Forest, XGBoost, LightGBM, LSTM) are compared. Other architectures (e.g., Transformer, Prophet, ARIMA) are not part of v1 but are not precluded later.
- **Ensembling**: An optional ensemble of two or more of the four models may be considered as a fifth candidate, but is not strictly required for v1.
- **Data availability**: The historical Beijing multi-site dataset is available, accessible, and licensed for use in this work.
- **Missing data threshold**: Hours with a small fraction of missing pollutant values are imputable; hours with most pollutants missing are excluded from training and flagged at inference time.
- **Compute environment**: A workstation- or single-server-class environment with a single GPU (for the LSTM) is available; multi-node distributed training is not required.
