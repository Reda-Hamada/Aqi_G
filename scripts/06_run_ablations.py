#!/usr/bin/env python
"""Run spatial-off and weather-off ablations.

Builds three tabular datasets (baseline, no-spatial, no-weather), trains the
specified model on each, and writes a side-by-side comparison report.
"""
from __future__ import annotations

import argparse

import pandas as pd

from aqi.aqi.compute import aqi_from_dataframe
from aqi.config import CLEANED_PARQUET, PROCESSED_DIR, set_global_seed
from aqi.features.build import build_tabular, feature_columns, target_columns, write_tabular
from aqi.persistence.registry import make_run_id, model_dir, write_run_manifest
from aqi.splits.walk_forward import chronological_masks


def _train(name: str, df: pd.DataFrame, masks, quick: bool):
    feats = feature_columns(df)
    tgts = target_columns()
    keep_tr = df[masks.train][tgts].notna().all(axis=1) & df[masks.train][feats].notna().all(axis=1)
    keep_va = df[masks.val][tgts].notna().all(axis=1) & df[masks.val][feats].notna().all(axis=1)
    df_tr = df[masks.train][keep_tr]
    df_va = df[masks.val][keep_va]
    if quick:
        df_tr = df_tr.iloc[:5000]
        df_va = df_va.iloc[:2000]

    if name == "lgbm":
        from aqi.models.lgbm import LGBMForecaster
        m = LGBMForecaster(n_estimators=400 if quick else 2000)
    elif name == "rf":
        from aqi.models.rf import RandomForestForecaster
        m = RandomForestForecaster(n_estimators=100 if quick else 400)
    elif name == "xgb":
        from aqi.models.xgb import XGBForecaster
        m = XGBForecaster(n_estimators=300 if quick else 1000)
    else:
        raise ValueError(name)
    m.fit(df_tr[feats], df_tr[tgts], df_va[feats], df_va[tgts])
    return m


def main(model_name: str, quick: bool) -> None:
    set_global_seed()
    raw = pd.read_parquet(CLEANED_PARQUET)
    raw = aqi_from_dataframe(raw)

    configs = {
        "baseline":    dict(include_spatial=True,  include_weather=True),
        "no_spatial":  dict(include_spatial=False, include_weather=True),
        "no_weather":  dict(include_spatial=True,  include_weather=False),
    }

    run_id = make_run_id() + "_ablate"
    rows = []
    for cfg_name, kwargs in configs.items():
        print(f"[ablate] Building tabular for {cfg_name}")
        df = build_tabular(raw, **kwargs)
        path = PROCESSED_DIR / f"tabular_{cfg_name}.parquet"
        write_tabular(df, path=path)
        df = pd.read_parquet(path)
        masks = chronological_masks(df)
        model = _train(model_name, df, masks, quick=quick)
        out = model_dir(run_id, f"{model_name}_{cfg_name}")
        model.save(out)
        rows.append({"config": cfg_name, "model_dir": str(out), "tabular": str(path)})

    write_run_manifest(run_id, {"ablations": rows, "model": model_name, "quick_mode": quick})
    print(f"[ablate] Done. run_id={run_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(args.model, args.quick)
