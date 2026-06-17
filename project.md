
# AQI Forecasting Project Specification

## Project Overview

This project aims to build a machine learning and deep learning system for forecasting Air Quality Index (AQI) in Beijing.

The forecasting horizons are:

* 1 Hour Ahead
* 3 Hours Ahead
* 6 Hours Ahead
* 24 Hours Ahead

The objective is to compare traditional machine learning models against deep learning approaches and evaluate the impact of spatial information and meteorological variables on prediction performance.

---

# Dataset Description

The dataset contains measurements collected from 12 monitoring stations in Beijing.

Available pollutants may include:

* PM2.5
* PM10
* SO2
* NO2
* CO
* O3

The dataset does not contain AQI directly.

AQI must be calculated from pollutant concentrations according to the official AQI standard before model training.

---

# Development Roadmap

## Phase 1 — Environment Setup

### Objectives

Prepare the development environment and project structure.

### Tasks

1. Create project directory structure.

```text
AQI-Forecasting/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── figures/
└── experiments/
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm shap
pip install tensorflow keras
pip install jupyter
```

3. Configure reproducibility:

* Set random seeds
* Create configuration file
* Create logging utilities

---

# Phase 2 — Data Understanding

## Objectives

Understand the dataset before any modeling.

### Tasks

Analyze:

* Number of rows
* Number of stations
* Time range
* Sampling frequency
* Missing values
* Outliers
* Data distributions

Generate:

* Summary statistics
* Missing value report
* Correlation matrix
* Station-level statistics

Deliverables:

* EDA Report
* Data Quality Report

---

# Phase 3 — AQI Calculation

## Objectives

Compute AQI from pollutant concentrations.

### Tasks

1. Implement AQI calculation.

2. Compute sub-index for:

* PM2.5
* PM10
* SO2
* NO2
* CO
* O3

3. Compute final AQI:

```text
AQI = max(SubIndices)
```

4. Validate AQI values.

5. Store results in:

```text
AQI
AQI_Category
```

Deliverables:

* AQI Calculator Module
* Validation Report

---

# Phase 4 — Baseline Dataset (City Average AQI)

## Objectives

Create a simple forecasting dataset.

### Tasks

Aggregate all stations.

For each timestamp:

```text
City_AQI = Mean(AQI of all stations)
```

Result:

Single time series representing city-level air quality.

Deliverables:

* city_average_aqi.csv

---

# Phase 5 — Feature Engineering

## Objectives

Create forecasting features.

### Lag Features

Create:

```text
AQI_lag_1
AQI_lag_3
AQI_lag_6
AQI_lag_12
AQI_lag_24
```

### Rolling Statistics

Create:

```text
rolling_mean_3
rolling_mean_6
rolling_mean_12
rolling_mean_24

rolling_std_3
rolling_std_6
rolling_std_12
rolling_std_24
```

### Time Features

Create:

```text
hour
day
month
day_of_week
week_of_year
is_weekend
```

Deliverables:

* Feature Engineering Pipeline

---

# Phase 6 — Forecast Targets

## Objectives

Create prediction targets.

Generate:

```text
AQI_t+1
AQI_t+3
AQI_t+6
AQI_t+24
```

These correspond to:

* 1 Hour Forecast
* 3 Hour Forecast
* 6 Hour Forecast
* 24 Hour Forecast

---

# Phase 7 — Baseline Machine Learning Models

## Objectives

Build strong baseline models.

### Model 1

Random Forest Regressor

### Model 2

XGBoost Regressor

### Model 3

LightGBM Regressor

### Training Strategy

Use:

```text
TimeSeriesSplit
```

Avoid:

```text
Random Train/Test Split
```

to prevent data leakage.

### Evaluation Metrics

Calculate:

* MAE
* RMSE
* R²

Deliverables:

Performance comparison table.

---

# Phase 8 — Deep Learning Models

## Objectives

Compare ML against sequence models.

### Model 4

LSTM

### Model 5

BiLSTM

### Sequence Length Experiments

Try:

```text
24
48
72
```

time steps.

### Hyperparameters

Tune:

* Hidden units
* Learning rate
* Batch size
* Number of layers
* Dropout

Deliverables:

Deep Learning Results Report.

---

# Phase 9 — Spatial Forecasting

## Objectives

Investigate whether spatial information improves performance.

### Option 2

Instead of city average:

Use all 12 stations.

### Create Multivariate Dataset

Features:

```text
Station_1_AQI
Station_2_AQI
...
Station_12_AQI
```

Train:

* XGBoost
* LightGBM
* LSTM

Compare against:

```text
City Average AQI
```

Research Question:

> Does spatial information improve AQI forecasting accuracy?

Deliverables:

Spatial Analysis Report.

---

# Phase 10 — Meteorological Features

## Objectives

Study weather influence.

### Add

* Temperature
* Humidity
* Pressure
* Wind Speed
* Wind Direction

### Experiments

Train:

1. AQI Only Models
2. AQI + Weather Models

Compare results.

Research Question:

> How much does meteorological information improve AQI prediction?

Deliverables:

Weather Feature Analysis Report.

---

# Phase 11 — Explainability

## Objectives

Interpret model behavior.

### Feature Importance

For:

* Random Forest
* XGBoost
* LightGBM

### SHAP Analysis

Generate:

* Summary Plot
* Dependence Plot
* Global Feature Ranking

Research Questions:

* Which features most influence AQI?
* Which lag values matter most?

Deliverables:

Explainability Report.

---

# Phase 12 — Model Comparison

Create a final comparison table.

| Model         | Horizon | MAE | RMSE | R² |
| ------------- | ------- | --- | ---- | -- |
| Random Forest | 1h      |     |      |    |
| Random Forest | 3h      |     |      |    |
| Random Forest | 6h      |     |      |    |
| Random Forest | 24h     |     |      |    |
| XGBoost       | 1h      |     |      |    |
| XGBoost       | 3h      |     |      |    |
| XGBoost       | 6h      |     |      |    |
| XGBoost       | 24h     |     |      |    |
| LightGBM      | 1h      |     |      |    |
| LightGBM      | 3h      |     |      |    |
| LightGBM      | 6h      |     |      |    |
| LightGBM      | 24h     |     |      |    |
| LSTM          | 1h      |     |      |    |
| LSTM          | 3h      |     |      |    |
| LSTM          | 6h      |     |      |    |
| LSTM          | 24h     |     |      |    |
| BiLSTM        | 1h      |     |      |    |
| BiLSTM        | 3h      |     |      |    |
| BiLSTM        | 6h      |     |      |    |
| BiLSTM        | 24h     |     |      |    |

---

# Phase 13 — Graduation Project Deliverables

Generate:

## Architecture Diagram

Show:

* Data Sources
* AQI Computation
* Feature Engineering
* Models
* Evaluation

---

## Methodology Section

Include:

* Dataset
* AQI Calculation
* Feature Engineering
* Modeling
* Evaluation

---

## Experimental Setup

Include:

* Hardware
* Software
* Libraries
* Hyperparameters

---

## Results Section

Include:

* Tables
* Charts
* Analysis

---

## Conclusion

Summarize:

* Best model
* Best horizon
* Major findings

---

## Future Work

Potential extensions:

* Transformer Models
* Graph Neural Networks
* Real-time Deployment
* Satellite Data Integration

---

# Development Rules for Claude

You must:

1. Work one phase at a time.
2. Never skip validation.
3. Explain every technical decision.
4. Generate production-quality code.
5. Prevent data leakage.
6. Follow time-series forecasting best practices.
7. Provide detailed reasoning before implementation.
8. Prefer simple baselines before advanced models.
9. Keep experiments reproducible.
10. Document all assumptions.

---

# Recommended Execution Order

1. Environment Setup
2. Data Understanding
3. AQI Calculation
4. City Average AQI
5. Feature Engineering
6. Random Forest
7. XGBoost
8. LightGBM
9. LSTM
10. BiLSTM
11. Spatial Modeling
12. Weather Features
13. SHAP Analysis
14. Final Comparison
15. Thesis Deliverables

Success Criterion:

Achieve the best possible forecasting performance while maintaining interpretability and scientific rigor suitable for a graduation project.
