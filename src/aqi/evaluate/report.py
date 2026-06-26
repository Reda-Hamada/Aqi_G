"""Compose the model comparison report (CSV + figures + Markdown).

The report is built from a list of ``EvalResult`` objects (one per model /
baseline) and writes, under ``reports/evaluation/<run_id>/``:

- ``comparison.csv``               overall metric matrix, one row per model
- ``<model>_per_horizon.csv`` etc. per-cut metric tables
- a suite of PNG figures (per-horizon curves, per-pollutant / per-station
  bars, overall comparison bars, confusion-matrix heatmaps)
- ``REPORT.md``                    a self-contained, human-readable report that
  embeds the tables and figures, suitable for pasting into a project document.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aqi.config import EVAL_DIR
from aqi.evaluate.runner import EvalResult

# Metrics we surface in the overall comparison table, in display order.
_OVERALL_COLS = [
    "mae",
    "rmse",
    "mape",
    "aqi_mae",
    "aqi_category_accuracy",
    "aqi_category_macro_f1",
]

_PRETTY = {
    "mae": "MAE (ug/m3)",
    "rmse": "RMSE (ug/m3)",
    "mape": "MAPE",
    "aqi_mae": "AQI MAE",
    "aqi_category_accuracy": "AQI cat. acc.",
    "aqi_category_macro_f1": "AQI macro-F1",
}


def comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.model}
        row.update(r.overall)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Stable column order: model first, then known metrics that are present.
    cols = ["model"] + [c for c in _OVERALL_COLS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols].sort_values("mae").reset_index(drop=True)


def write_report(
    results: list[EvalResult],
    run_id: str,
    *,
    notes: dict | None = None,
) -> Path:
    """Write CSVs, figures and a Markdown report. Returns the output dir."""
    out_dir = Path(EVAL_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    table = comparison_table(results)
    table.to_csv(out_dir / "comparison.csv", index=False)

    for r in results:
        r.per_horizon.to_csv(out_dir / f"{r.model}_per_horizon.csv", index=False)
        r.per_pollutant.to_csv(out_dir / f"{r.model}_per_pollutant.csv", index=False)
        r.per_station.to_csv(out_dir / f"{r.model}_per_station.csv", index=False)
        if not r.confusion.empty:
            r.confusion.to_csv(out_dir / f"{r.model}_confusion_h24.csv")

    figures: list[str] = []
    try:
        figures = _draw_figures(results, out_dir)
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"[report] Figure rendering skipped: {e}")

    try:
        _write_markdown(results, table, figures, out_dir, run_id, notes or {})
    except Exception as e:  # pragma: no cover
        print(f"[report] Markdown rendering skipped: {e}")

    return out_dir


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _draw_figures(results: list[EvalResult], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})
    written: list[str] = []

    def _save(fig, name: str):
        fig.savefig(out_dir / name)
        plt.close(fig)
        written.append(name)

    # Models vs baselines colouring: models solid, baselines dashed/grey.
    baseline_names = {"persistence", "climatology", "seasonal_naive"}

    # 1. Per-horizon concentration MAE -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        style = dict(marker="o", ms=3)
        if r.model in baseline_names:
            style.update(linestyle="--", alpha=0.6)
        ax.plot(r.per_horizon["horizon"], r.per_horizon["mae"], label=r.model, **style)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (concentration, ug/m^3)")
    ax.set_title("Per-horizon pollutant MAE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "per_horizon_mae.png")

    # 2. Per-horizon AQI MAE ----------------------------------------------
    if any("aqi_mae" in r.per_horizon.columns for r in results):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            if "aqi_mae" not in r.per_horizon.columns:
                continue
            style = dict(marker="o", ms=3)
            if r.model in baseline_names:
                style.update(linestyle="--", alpha=0.6)
            ax.plot(r.per_horizon["horizon"], r.per_horizon["aqi_mae"], label=r.model, **style)
        ax.set_xlabel("Forecast horizon (hours ahead)")
        ax.set_ylabel("AQI MAE (index points)")
        ax.set_title("Per-horizon AQI error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        _save(fig, "per_horizon_aqi_mae.png")

    # 3. Per-horizon AQI category accuracy --------------------------------
    if any("aqi_category_accuracy" in r.per_horizon.columns for r in results):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            if "aqi_category_accuracy" not in r.per_horizon.columns:
                continue
            style = dict(marker="o", ms=3)
            if r.model in baseline_names:
                style.update(linestyle="--", alpha=0.6)
            ax.plot(r.per_horizon["horizon"], r.per_horizon["aqi_category_accuracy"],
                    label=r.model, **style)
        ax.set_xlabel("Forecast horizon (hours ahead)")
        ax.set_ylabel("AQI category accuracy")
        ax.set_title("Per-horizon AQI category accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        _save(fig, "per_horizon_category_accuracy.png")

    # 4. Overall comparison bars (MAE / RMSE / AQI MAE) -------------------
    table = comparison_table(results)
    metrics_for_bars = [m for m in ["mae", "rmse", "aqi_mae"] if m in table.columns]
    if metrics_for_bars:
        fig, axes = plt.subplots(1, len(metrics_for_bars),
                                 figsize=(4.5 * len(metrics_for_bars), 4.5))
        if len(metrics_for_bars) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics_for_bars):
            t = table.sort_values(m)
            colors = ["#bbbbbb" if mod in baseline_names else "#3477b8" for mod in t["model"]]
            ax.barh(t["model"], t[m], color=colors)
            ax.set_xlabel(_PRETTY.get(m, m))
            ax.set_title(_PRETTY.get(m, m))
            ax.invert_yaxis()
            ax.grid(alpha=0.3, axis="x")
        fig.suptitle("Overall test-set comparison (lower is better)")
        _save(fig, "overall_comparison.png")

    # 5. Per-pollutant MAE grouped bars -----------------------------------
    model_results = [r for r in results if r.model not in baseline_names] or results
    pol_order = list(model_results[0].per_pollutant["pollutant"])
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(model_results)
    width = 0.8 / max(n, 1)
    x = np.arange(len(pol_order))
    for i, r in enumerate(model_results):
        pp = r.per_pollutant.set_index("pollutant").reindex(pol_order)
        ax.bar(x + i * width, pp["mae"].to_numpy(), width, label=r.model)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(pol_order)
    ax.set_ylabel("MAE (ug/m^3)")
    ax.set_title("Per-pollutant MAE by model")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, "per_pollutant_mae.png")

    # 6. Per-station MAE for the best model -------------------------------
    best = table.iloc[0]["model"]
    best_r = next((r for r in results if r.model == best), None)
    if best_r is not None and not best_r.per_station.empty:
        ps = best_r.per_station.sort_values("mae")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(ps["station"], ps["mae"], color="#3477b8")
        ax.set_xlabel("MAE (ug/m^3)")
        ax.set_title(f"Per-station MAE — {best} (best model)")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        _save(fig, "per_station_mae_best.png")

    # 7. Confusion-matrix heatmaps (h=24) for each model with one --------
    for r in results:
        if r.confusion.empty:
            continue
        cm = r.confusion.to_numpy(dtype="float64")
        labels = list(r.confusion.index)
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        short = [l.replace("Unhealthy for Sensitive Groups", "USG") for l in labels]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("True category")
        ax.set_title(f"AQI category confusion (h=24) — {r.model}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if row_sums[i] > 0:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
        _save(fig, f"confusion_h24_{r.model}.png")

    return written


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _md_table(df: pd.DataFrame, rename: dict | None = None) -> str:
    d = df.copy()
    if rename:
        d = d.rename(columns=rename)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    lines = [header, sep]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _write_markdown(
    results: list[EvalResult],
    table: pd.DataFrame,
    figures: list[str],
    out_dir: Path,
    run_id: str,
    notes: dict,
) -> None:
    baseline_names = {"persistence", "climatology", "seasonal_naive"}
    model_results = [r for r in results if r.model not in baseline_names]
    lines: list[str] = []

    lines.append(f"# Model comparison report — run `{run_id}`\n")
    if notes.get("intro"):
        lines.append(notes["intro"] + "\n")

    # --- Overall matrix ---
    lines.append("## 1. Overall test-set metric matrix\n")
    lines.append("Sorted by pollutant MAE (lower is better). "
                 "Concentration metrics are averaged over the 6 pollutants and "
                 "24 horizons; AQI metrics are computed from the predicted "
                 "concentrations.\n")
    disp = table.copy()
    lines.append(_md_table(disp, rename=_PRETTY))
    lines.append("")

    if figures and "overall_comparison.png" in figures:
        lines.append("![Overall comparison](overall_comparison.png)\n")

    # --- SC-001 persistence comparison ---
    pers = next((r for r in results if r.model == "persistence"), None)
    if pers is not None and model_results:
        pmae = pers.overall.get("mae", float("nan"))
        lines.append("## 2. Skill vs. persistence baseline (SC-001)\n")
        lines.append("Relative MAE reduction against the persistence baseline "
                     f"(persistence MAE = {pmae:.3f}). The spec target is a "
                     "≥ 20 % reduction.\n")
        rows = []
        for r in model_results:
            mmae = r.overall.get("mae", float("nan"))
            imp = (pmae - mmae) / pmae * 100 if pmae and not np.isnan(pmae) else float("nan")
            rows.append({"model": r.model, "MAE": mmae,
                         "improvement vs persistence (%)": imp,
                         "passes ≥20%": "yes" if imp >= 20 else "no"})
        lines.append(_md_table(pd.DataFrame(rows)))
        lines.append("")

    # --- Per-horizon ---
    lines.append("## 3. Error growth with horizon\n")
    lines.append("Forecast error as a function of how far ahead we predict "
                 "(1–24 h). A well-behaved model degrades smoothly with horizon.\n")
    for fig in ["per_horizon_mae.png", "per_horizon_aqi_mae.png",
                "per_horizon_category_accuracy.png"]:
        if fig in figures:
            lines.append(f"![{fig}]({fig})\n")

    # --- Per-pollutant ---
    lines.append("## 4. Per-pollutant performance\n")
    if "per_pollutant_mae.png" in figures:
        lines.append("![Per-pollutant MAE](per_pollutant_mae.png)\n")
    if model_results:
        best = table.iloc[0]["model"]
        best_r = next((r for r in results if r.model == best), None)
        if best_r is not None:
            lines.append(f"Per-pollutant MAE/RMSE/MAPE for the best model "
                         f"(**{best}**):\n")
            lines.append(_md_table(best_r.per_pollutant))
            lines.append("")

    # --- Per-station ---
    lines.append("## 5. Per-station performance (best model)\n")
    if "per_station_mae_best.png" in figures:
        lines.append("![Per-station MAE](per_station_mae_best.png)\n")

    # --- Confusion ---
    conf_figs = [f for f in figures if f.startswith("confusion_h24_")]
    if conf_figs:
        lines.append("## 6. AQI category confusion (24 h ahead)\n")
        lines.append("Row-normalised confusion matrices over the six EPA AQI "
                     "categories at the 24-hour horizon.\n")
        for f in conf_figs:
            lines.append(f"![{f}]({f})\n")

    # --- Methodology / notes ---
    if notes.get("methodology"):
        lines.append("## 7. Methodology & caveats\n")
        lines.append(notes["methodology"] + "\n")

    (out_dir / "REPORT.md").write_text("\n".join(lines))
