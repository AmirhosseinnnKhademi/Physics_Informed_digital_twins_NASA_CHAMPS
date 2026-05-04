"""Stage - evaluate: Compare XGBoost (base) vs XGBoost-Rolling vs GRU-Simple.

Engine-level split: test engines are completely held out during training.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import savgol_filter
from dotenv import load_dotenv
import mlflow

from mlflow_setup import setup_mlflow
from features import add_rolling_features
from models import SimpleGRUModel, RULWindowDataset

load_dotenv()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS  = Path("reports/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

C_XGB         = "#2980b9"    # blue
C_XGB_ROLLING = "#e67e22"    # orange
C_GRU_SIMPLE  = "#8e44ad"    # purple
C_ACTUAL      = "#27ae60"
C_TRAIN       = "#2ecc71"
C_VAL         = "#f39c12"
C_TEST        = "#e74c3c"

MODEL_STYLES = [
    ("RUL_xgb",         "XGBoost",        C_XGB,         ":"),
    ("RUL_xgb_rolling", "XGBoost-Rolling", C_XGB_ROLLING, "-."),
    ("RUL_gru_simple",  "GRU-Simple",     C_GRU_SIMPLE,  "--"),
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y, yh): return float(np.sqrt(np.mean((yh - y) ** 2)))
def mae(y, yh):  return float(np.mean(np.abs(yh - y)))
def r2(y, yh):
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((yh - y) ** 2) / ss_tot) if ss_tot > 0 else float("nan")

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum())


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_xgb_rul(model, scaler_pkg, all_df, feature_cols, test_units):
    test_df = all_df[all_df["unit"].isin(test_units)].copy()
    X = scaler_pkg["scaler"].transform(test_df[feature_cols])
    df = test_df[["unit", "cycle", "RUL"]].rename(columns={"RUL": "RUL_actual"}).copy()
    df["unit"]    = df["unit"].astype(int)
    df["cycle"]   = df["cycle"].astype(int)
    df["RUL_xgb"] = model.predict(X)
    return df.reset_index(drop=True)


def predict_xgb_rolling_rul(model, scaler_pkg, all_df, test_units):
    stat_cols        = scaler_pkg["stat_cols"]
    all_feature_cols = scaler_pkg["all_feature_cols"]
    W                = scaler_pkg["rolling_window"]
    enriched         = add_rolling_features(all_df, stat_cols, W)
    test_enriched    = enriched[enriched["unit"].isin(set(test_units))].copy()
    X                = scaler_pkg["scaler"].transform(test_enriched[all_feature_cols])
    df = test_enriched[["unit", "cycle", "RUL"]].rename(columns={"RUL": "RUL_actual"}).copy()
    df["unit"]            = df["unit"].astype(int)
    df["cycle"]           = df["cycle"].astype(int)
    df["RUL_xgb_rolling"] = model.predict(X)
    return df.reset_index(drop=True)


def predict_gru_simple_rul(model, scaler_pkg, all_df, test_units):
    feature_cols = scaler_pkg["feature_cols"]
    W            = scaler_pkg["window_size"]
    rul_cap      = scaler_pkg["rul_cap"]

    test_df = all_df[all_df["unit"].isin(test_units)].copy()
    scaled  = test_df.copy()
    scaled[feature_cols] = scaler_pkg["scaler"].transform(test_df[feature_cols])

    ds     = RULWindowDataset(scaled, feature_cols, W)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            p_norm = model(X.to(DEVICE)).cpu().numpy()
            preds.extend(np.clip(p_norm * rul_cap, 0, None))
    preds = np.array(preds)

    records = []
    for _, unit_df in test_df.groupby("unit", sort=True):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        n = len(unit_df)
        for i in range(W, n + 1):
            records.append({
                "unit":       int(unit_df["unit"].iloc[0]),
                "cycle":      int(unit_df["cycle"].iloc[i - 1]),
                "RUL_actual": float(unit_df["RUL"].iloc[i - 1]),
            })
    df = pd.DataFrame(records).reset_index(drop=True)
    df["RUL_gru_simple"] = preds
    return df


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_rul_all_engines(all_df, pred_dfs, out_path, train_units, val_units, test_units):
    units = sorted(all_df["unit"].unique().astype(int))
    ncols = 3
    nrows = (len(units) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    axes = np.array(axes).flatten()

    role_color = {u: C_TRAIN for u in train_units}
    role_color.update({u: C_VAL  for u in val_units})
    role_color.update({u: C_TEST for u in test_units})
    role_label = {u: "TRAIN" for u in train_units}
    role_label.update({u: "VAL"  for u in val_units})
    role_label.update({u: "TEST" for u in test_units})

    for ax, unit in zip(axes, units):
        full = all_df[all_df["unit"] == unit].sort_values("cycle")
        ax.plot(full["cycle"], full["RUL"], color=C_ACTUAL,
                linewidth=2.2, label="Actual RUL", zorder=3)

        if unit in test_units:
            for col, name, color, ls in MODEL_STYLES:
                df = pred_dfs[col]
                u_df = df[df["unit"] == unit].sort_values("cycle")
                if not u_df.empty:
                    ax.plot(u_df["cycle"], u_df[col], color=color,
                            linewidth=1.8, linestyle=ls, label=name, zorder=4)

        tc = role_color.get(unit, "#7f8c8d")
        ax.set_title(f"Engine {unit}  [{role_label.get(unit,'?')}]",
                     fontsize=11, color=tc, fontweight="bold")
        ax.set_xlabel("Cycle"); ax.set_ylabel("RUL (cycles)")
        if unit in test_units:
            ax.legend(fontsize=7, ncol=2)

    for ax in axes[len(units):]:
        ax.set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_TRAIN, label=f"Train engines {sorted(train_units)}"),
        mpatches.Patch(color=C_VAL,   label=f"Val engines {sorted(val_units)}"),
        mpatches.Patch(color=C_TEST,
                       label=f"Test engines {sorted(test_units)} — predictions shown"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.015), fontsize=10)
    fig.suptitle("RUL Forecast — All Engines  (engine-level split)",
                 fontsize=12, fontweight="bold", y=1.06)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_test_engines_detail(all_df, rul_df, test_units, out_path):
    n = len(test_units)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), squeeze=False)
    axes = axes.flatten()

    for ax, unit in zip(axes, sorted(test_units)):
        full = all_df[all_df["unit"] == unit].sort_values("cycle")
        ax.plot(full["cycle"], full["RUL"], color=C_ACTUAL,
                linewidth=2.5, label="Actual RUL", zorder=3)

        for col, name, color, ls in MODEL_STYLES:
            sub = rul_df[rul_df["unit"] == unit].dropna(subset=[col]).sort_values("cycle")
            if not sub.empty:
                r = rmse(sub["RUL_actual"].values, sub[col].values)
                ax.plot(sub["cycle"], sub[col], color=color, linewidth=2.0, linestyle=ls,
                        label=f"{name} (RMSE={r:.1f})", zorder=4)

        ax.set_title(f"Test Engine {unit} — Predicted vs Actual RUL",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Cycle"); ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=10)

    fig.suptitle("Test Set Evaluation — Completely unseen during training",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_scatter(rul_common, out_path, gru_window):
    """All three models evaluated on the same n cycles (GRU window onwards)."""
    n = len(MODEL_STYLES)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    max_val = rul_common["RUL_actual"].max()
    lim = (-1, max_val * 1.05)
    n_pts = len(rul_common)
    for ax, (col, name, color, _) in zip(axes, MODEL_STYLES):
        yt = rul_common["RUL_actual"].values
        yp = rul_common[col].values
        ax.scatter(yt, yp, alpha=0.45, s=14, color=color, rasterized=True)
        ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_title(f"{name}  (n={n_pts})\nR²={r2(yt, yp):.3f}  RMSE={rmse(yt, yp):.2f}")
        ax.set_xlabel("Actual RUL"); ax.set_ylabel("Predicted RUL")
        ax.legend()
    fig.suptitle(f"Scatter: Predicted vs Actual RUL — same {n_pts} cycles for all models "
                 f"(cycle {gru_window}+ per engine, fair comparison)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_error_dist(rul_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, name, color, _ in MODEL_STYLES:
        err = rul_df[col].dropna() - rul_df.loc[rul_df[col].notna(), "RUL_actual"]
        ax.hist(err, bins=60, alpha=0.55, label=name, color=color, edgecolor="none")
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (predicted - actual) [cycles]")
    ax.set_ylabel("Count")
    ax.set_title("RUL Prediction Error Distribution — Test Engines", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_nasa_scores(m_dict, out_path, best_model=""):
    models = [m for m in m_dict if not m.startswith("_")]
    scores = [m_dict[m]["nasa_score"] for m in models]
    color_map = {name: color for _, name, color, _ in MODEL_STYLES}
    colors = [color_map.get(m, "#7f8c8d") for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=0.8)
    for bar, sc, name in zip(bars, scores, models):
        is_best = best_model and best_model in name
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(scores)*0.01,
                f"{sc:,.0f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold" if is_best else "normal")
        if is_best:
            bar.set_edgecolor("gold"); bar.set_linewidth(3)
    ax.set_ylabel("NASA PHM Score (lower = better)")
    ax.set_title("NASA Asymmetric PHM Score — Test Engines (fine-tuned)\n"
                 f"Best model: {best_model}", fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_residuals(rul_common, out_path):
    """Residual (pred − actual) vs actual RUL, one panel per model."""
    n   = len(MODEL_STYLES)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    all_res = np.concatenate([
        rul_common[col].values - rul_common["RUL_actual"].values
        for col, *_ in MODEL_STYLES
    ])
    ylim = max(abs(all_res.min()), abs(all_res.max())) * 1.1

    for ax, (col, name, color, _) in zip(axes, MODEL_STYLES):
        yt  = rul_common["RUL_actual"].values
        res = rul_common[col].values - yt

        ax.scatter(yt, res, alpha=0.55, s=18, color=color, rasterized=True, zorder=3)
        ax.axhline(0, color="black", lw=1.5, linestyle="--", zorder=2)

        # Linear trend
        z  = np.polyfit(yt, res, 1)
        xr = np.linspace(yt.min(), yt.max(), 200)
        ax.plot(xr, np.polyval(z, xr), color="crimson", lw=2,
                label=f"Trend  slope={z[0]:+.3f}")

        # Mean bias line
        bias = res.mean()
        ax.axhline(bias, color=color, lw=1.2, linestyle=":",
                   label=f"Mean bias = {bias:+.2f} cyc")

        ax.set_xlim(-1, yt.max() * 1.05)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel("Actual RUL (cycles)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Residual  (predicted − actual)  [cycles]", fontsize=11)
        ax.set_title(name, fontweight="bold", fontsize=12)
        ax.legend(fontsize=9)
        ax.fill_between([0, yt.max()], -2, 2, alpha=0.07, color="green",
                        label="±2 cycle band")

    fig.suptitle("Residuals vs Actual RUL — Test Engines (fine-tuned)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_error_cdf(rul_common, out_path):
    """CDF of |error| with a summary table below the chart."""
    thresholds = [3, 5, 10, 15]
    fig = plt.figure(figsize=(12, 8))
    gs  = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.55)
    ax  = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")

    max_err   = 0.0
    table_rows = []

    for col, name, color, _ in MODEL_STYLES:
        abs_err    = np.abs(rul_common[col].values - rul_common["RUL_actual"].values)
        sorted_err = np.sort(abs_err)
        cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        ax.plot(sorted_err, cdf, color=color, lw=2.5, label=name)
        max_err = max(max_err, sorted_err[-1])
        table_rows.append(
            [name] + [f"{np.mean(abs_err <= t) * 100:.0f}%" for t in thresholds]
        )

    for t in thresholds:
        ax.axvline(t, color="grey", lw=1, linestyle=":", alpha=0.6)
        ax.text(t, 102, f"{t} cyc", ha="center", va="bottom", fontsize=9, color="grey")

    ax.set_xlim(0, max_err * 1.05)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Absolute Error (cycles)", fontsize=11)
    ax.set_ylabel("Cumulative % of Predictions", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.set_title("CDF of Absolute Prediction Error — Test Engines (fine-tuned)",
                 fontweight="bold", fontsize=12)

    col_labels = ["Model", "≤3 cyc", "≤5 cyc", "≤10 cyc", "≤15 cyc"]
    tbl = ax_t.table(cellText=table_rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("#bdc3c7")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


# Sensors most sensitive to HPC + LPT degradation in N-CMAPSS DS03
DEGRADATION_SENSORS = [
    ("T50",  "T50 — LPT Exit Temperature  [°R]",        "rises",  "#c0392b"),
    ("T30",  "T30 — HPC Exit Temperature  [°R]",        "rises",  "#e67e22"),
    ("Ps30", "Ps30 — HPC Static Pressure  [psia]",      "drops",  "#2980b9"),
    ("Wf",   "Wf — Fuel Flow  [pps]",                   "rises",  "#8e44ad"),
    ("T48",  "T48 — LPT Inlet Temperature  [°R]",       "rises",  "#16a085"),
]


def _sg_window(n, frac=0.10, min_win=9):
    """Savitzky-Golay window: ~frac of series length, odd, at least min_win."""
    w = max(min_win, int(n * frac))
    return w if w % 2 == 1 else w + 1


def plot_sensor_trajectories(all_df, rul_df, test_unit, out_path):
    """Raw + SG-smoothed sensor signals for one test engine, with RUL reference."""
    unit_df = all_df[all_df["unit"] == test_unit].sort_values("cycle").reset_index(drop=True)
    cycles  = unit_df["cycle"].values
    n       = len(cycles)
    win     = _sg_window(n)

    n_sensors = len(DEGRADATION_SENSORS)
    fig, axes = plt.subplots(n_sensors + 1, 1, figsize=(14, 3.6 * (n_sensors + 1)),
                             sharex=True)

    # ── Row 0: RUL reference (actual + model predictions) ────────────────────
    ax0 = axes[0]
    ax0.plot(cycles, unit_df["RUL"].values, color=C_ACTUAL, lw=2.5,
             label="Actual RUL", zorder=5)
    for col, name, color, ls in MODEL_STYLES:
        sub = rul_df[(rul_df["unit"] == test_unit)].dropna(subset=[col]).sort_values("cycle")
        if not sub.empty:
            ax0.plot(sub["cycle"], sub[col], color=color, lw=1.8, linestyle=ls,
                     label=name, zorder=4)
    ax0.set_ylabel("RUL (cycles)", fontsize=10)
    ax0.set_title("Actual vs Predicted RUL  [reference]", fontsize=10, style="italic")
    ax0.legend(fontsize=8, ncol=2, loc="upper right")
    ax0.grid(True, alpha=0.25)

    # ── Rows 1–5: one sensor each ────────────────────────────────────────────
    for ax, (sensor, label, direction, color) in zip(axes[1:], DEGRADATION_SENSORS):
        col = f"{sensor}_mean"
        raw = unit_df[col].values.astype(np.float64)
        smooth = savgol_filter(raw, window_length=win, polyorder=3)

        ax.scatter(cycles, raw, alpha=0.20, s=10, color=color, zorder=2)
        ax.plot(cycles, smooth, color=color, lw=2.2, label=f"Smoothed (SG win={win})", zorder=3)
        ax.plot(cycles, raw, color=color, lw=0.7, alpha=0.45, zorder=2, label="Raw")

        # Mark start / end of window
        ax.axvline(cycles[0],  color="grey", lw=0.8, linestyle=":", alpha=0.6)
        ax.axvline(cycles[-1], color="grey", lw=0.8, linestyle=":", alpha=0.6)

        arrow_txt = f"Degradation: {direction} ↑" if direction == "rises" else f"Degradation: {direction} ↓"
        ax.text(0.01, 0.96, arrow_txt, transform=ax.transAxes, fontsize=8,
                va="top", color="crimson", alpha=0.85)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Cycle", fontsize=11)
    fig.suptitle(f"Degradation-Relevant Sensor Trajectories — Engine {test_unit} (Test)\n"
                 f"Raw vs Savitzky-Golay Smoothed  (window = {win} cycles, order = 3)",
                 fontweight="bold", fontsize=12, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_sensor_slopes(all_df, rul_df, test_unit, out_path):
    """Local d(sensor)/d(cycle) — nonlinear degradation rate, with RUL slope reference."""
    unit_df = all_df[all_df["unit"] == test_unit].sort_values("cycle").reset_index(drop=True)
    cycles  = unit_df["cycle"].values.astype(np.float64)
    n       = len(cycles)
    win     = _sg_window(n)
    win_sl  = _sg_window(n, frac=0.15)  # slightly wider window for slope smoothing

    n_sensors = len(DEGRADATION_SENSORS)
    fig, axes = plt.subplots(n_sensors + 1, 1, figsize=(14, 3.6 * (n_sensors + 1)),
                             sharex=True)

    # ── Row 0: RUL slope — actual vs models ──────────────────────────────────
    ax0 = axes[0]
    rul_raw    = unit_df["RUL"].values.astype(np.float64)
    rul_smooth = savgol_filter(rul_raw, window_length=win, polyorder=3)
    rul_slope  = np.gradient(rul_smooth, cycles)

    ax0.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.4)
    ax0.axhline(-1, color="grey", lw=1.0, linestyle=":", alpha=0.6, label="Ideal slope (−1)")
    ax0.plot(cycles, rul_slope, color=C_ACTUAL, lw=2.2, label="Actual RUL slope", zorder=4)

    for col, name, color, ls in MODEL_STYLES:
        sub = rul_df[(rul_df["unit"] == test_unit)].dropna(subset=[col]).sort_values("cycle")
        if not sub.empty:
            cy   = sub["cycle"].values.astype(np.float64)
            vals = sub[col].values.astype(np.float64)
            if len(vals) >= win:
                sm  = savgol_filter(vals, window_length=_sg_window(len(vals)), polyorder=3)
                sl  = np.gradient(sm, cy)
                ax0.plot(cy, sl, color=color, lw=1.6, linestyle=ls, label=f"{name} slope", zorder=3)

    ax0.set_ylabel("dRUL/d(cycle)", fontsize=10)
    ax0.set_title("RUL local slope: ideal = −1  (deviations = nonlinear degradation)",
                  fontsize=10, style="italic")
    ax0.legend(fontsize=8, ncol=2, loc="upper right")
    ax0.grid(True, alpha=0.25)

    # ── Rows 1–5: sensor slopes ───────────────────────────────────────────────
    for ax, (sensor, label, direction, color) in zip(axes[1:], DEGRADATION_SENSORS):
        col    = f"{sensor}_mean"
        raw    = unit_df[col].values.astype(np.float64)
        smooth = savgol_filter(raw, window_length=win,    polyorder=3)
        slope  = np.gradient(smooth, cycles)
        slope_sm = savgol_filter(slope, window_length=win_sl, polyorder=2)

        ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.45, zorder=1)

        # Shade: green = slope in the healthy direction, red = degrading direction
        if direction == "rises":
            ax.fill_between(cycles, slope_sm, 0,
                            where=(slope_sm > 0), alpha=0.18, color="crimson", label="Degrading (rising)")
            ax.fill_between(cycles, slope_sm, 0,
                            where=(slope_sm < 0), alpha=0.12, color="steelblue", label="Recovering / stable")
        else:
            ax.fill_between(cycles, slope_sm, 0,
                            where=(slope_sm < 0), alpha=0.18, color="crimson", label="Degrading (dropping)")
            ax.fill_between(cycles, slope_sm, 0,
                            where=(slope_sm > 0), alpha=0.12, color="steelblue", label="Recovering / stable")

        ax.plot(cycles, slope,    color=color, lw=0.8, alpha=0.35, zorder=2)
        ax.plot(cycles, slope_sm, color=color, lw=2.2, zorder=3,
                label=f"d({sensor})/d(cycle)  [SG win={win_sl}]")

        ax.set_ylabel(f"d({sensor})/d(cyc)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Cycle", fontsize=11)
    fig.suptitle(f"Local Degradation Rate  dSensor/dCycle — Engine {test_unit} (Test)\n"
                 "Red shading = sensor moving in the degrading direction",
                 fontweight="bold", fontsize=12, y=1.005)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


def plot_gru_loss_curves(out_path):
    hp = Path("metrics/gru_simple_history.json")
    if not hp.exists(): return
    h = json.load(open(hp))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h["train_loss"], color=C_GRU_SIMPLE, label="Train loss (Huber, normalised)")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Huber Loss")
    a1.set_title("Training Loss"); a1.legend()
    a2.plot(h["val_rmse"], color=C_GRU_SIMPLE, label="Val RMSE (cycles)")
    a2.set_xlabel("Epoch"); a2.set_ylabel("RMSE (cycles)")
    a2.set_title("Validation RMSE"); a2.legend()
    fig.suptitle("GRU-Simple Training Curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"  saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params       = yaml.safe_load(open("params.yaml"))
    rul_cap      = params["prepare"]["rul_cap"]
    meta         = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]
    test_units   = meta["test_units"]

    # Best-model info written by the finetune stage
    best_info  = json.load(open("models/best_model.json"))
    best_model = best_info["best_model"]
    print(f"[evaluate] best model from finetune: {best_model}  "
          f"(val RMSE={best_info['best_val_rmse']})")

    all_df  = pd.read_parquet("data/processed/all_cycles.parquet")
    test_df = all_df[all_df["unit"].isin(test_units)]
    print(f"[evaluate] test engines: {sorted(test_df['unit'].unique().astype(int))}  "
          f"({len(test_df)} cycles)")

    # ── Load fine-tuned models ────────────────────────────────────────────
    with open("models/ft_xgb_scaler.pkl", "rb") as f: xgb_pkg = pickle.load(f)
    xgb_rul = xgb.XGBRegressor()
    xgb_rul.load_model("models/ft_xgb_rul.ubj")

    with open("models/ft_xgb_rolling_scaler.pkl", "rb") as f: xgb_roll_pkg = pickle.load(f)
    xgb_roll = xgb.XGBRegressor()
    xgb_roll.load_model("models/ft_xgb_rolling_rul.ubj")

    with open("models/ft_gru_simple_scaler.pkl", "rb") as f: gru_pkg = pickle.load(f)
    gru_model = SimpleGRUModel(
        input_size  = len(feature_cols),
        hidden_size = gru_pkg["hidden_size"],
        dense_size  = gru_pkg["dense_size"],
        dropout     = gru_pkg["dropout"],
    ).to(DEVICE)
    gru_model.load_state_dict(
        torch.load("models/ft_gru_simple_rul.pt", map_location=DEVICE))
    gru_ws = gru_pkg["window_size"]

    # Build display names — winning model gets a ★
    col_to_base = {
        "RUL_xgb":         "XGBoost",
        "RUL_xgb_rolling": "XGBoost-Rolling",
        "RUL_gru_simple":  "GRU-Simple",
    }
    global MODEL_STYLES
    MODEL_STYLES = [
        (col, (name + "  [BEST]" if col_to_base[col] == best_model else name), color, ls)
        for col, name, color, ls in [
            ("RUL_xgb",         "XGBoost",        C_XGB,         ":"),
            ("RUL_xgb_rolling", "XGBoost-Rolling", C_XGB_ROLLING, "-."),
            ("RUL_gru_simple",  "GRU-Simple",     C_GRU_SIMPLE,  "--"),
        ]
    ]

    # ── Predictions ───────────────────────────────────────────────────────
    print("[evaluate] XGBoost predictions (fine-tuned)...")
    xgb_df = predict_xgb_rul(xgb_rul, xgb_pkg, all_df, feature_cols, test_units)

    print("[evaluate] XGBoost-Rolling predictions (fine-tuned)...")
    xgb_roll_df = predict_xgb_rolling_rul(xgb_roll, xgb_roll_pkg, all_df, test_units)

    print("[evaluate] GRU-Simple predictions (fine-tuned)...")
    gru_df = predict_gru_simple_rul(gru_model, gru_pkg, all_df, test_units)

    # Full merge (GRU rows start at cycle W, so early cycles have NaN for GRU)
    rul_df = xgb_df \
        .merge(xgb_roll_df[["unit", "cycle", "RUL_xgb_rolling"]], on=["unit", "cycle"], how="left") \
        .merge(gru_df[["unit", "cycle", "RUL_gru_simple"]], on=["unit", "cycle"], how="left")

    # Common subset: only cycles where ALL three models have predictions.
    rul_common = rul_df.dropna(subset=["RUL_gru_simple"]).copy()
    print(f"[evaluate] common evaluation window: {len(rul_common)} cycles "
          f"(GRU window={gru_ws}, first {gru_ws} cycles per engine excluded)")

    # Per-model prediction dicts for the overview time-series plot (full range)
    pred_dfs = {
        "RUL_xgb":         xgb_df,
        "RUL_xgb_rolling": xgb_roll_df,
        "RUL_gru_simple":  gru_df,
    }

    # ── Metrics (on common cycles only) ───────────────────────────────────
    metrics = {}
    for col, name, _, _ in MODEL_STYLES:
        yt = rul_common["RUL_actual"].values
        yp = rul_common[col].values
        metrics[name] = {
            "rmse":       round(rmse(yt, yp), 4),
            "mae":        round(mae(yt, yp),  4),
            "r2":         round(r2(yt, yp),   4),
            "nasa_score": round(nasa_score(yt, yp), 2),
            "n_samples":  int(len(rul_common)),
        }
    metrics["_best_model"] = best_model

    print("\n[evaluate] -- Fine-tuned RUL Regression (test engines) --")
    print(f"{'Model':<24} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'NASA Score':>12}")
    print("-"*64)
    for m, v in metrics.items():
        if m.startswith("_"): continue
        print(f"{m:<24} {v['rmse']:>8.3f} {v['mae']:>8.3f} "
              f"{v['r2']:>8.4f} {v['nasa_score']:>12,.1f}")

    Path("metrics").mkdir(exist_ok=True)
    json.dump(metrics, open("metrics/evaluation.json", "w"), indent=2)
    print("\n[evaluate] saved metrics/evaluation.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("[evaluate] Generating plots...")
    plot_rul_all_engines(all_df, pred_dfs, PLOTS/"01_rul_all_engines.png",
                         train_units, val_units, test_units)
    plot_test_engines_detail(all_df, rul_common, test_units, PLOTS/"02_rul_test_engines.png")
    plot_scatter(rul_common,  PLOTS/"03_scatter.png", gru_ws)
    plot_error_dist(rul_common, PLOTS/"04_error_dist.png")
    plot_nasa_scores(metrics,  PLOTS/"05_nasa_scores.png",  best_model)
    plot_gru_loss_curves(      PLOTS/"06_gru_loss_curves.png")
    plot_residuals(rul_common, PLOTS/"07_residuals.png")
    plot_error_cdf(rul_common, PLOTS/"08_error_cdf.png")
    test_unit = sorted(test_units)[0]
    plot_sensor_trajectories(all_df, rul_df, test_unit, PLOTS/"09_sensor_trajectories.png")
    plot_sensor_slopes(all_df, rul_df, test_unit,       PLOTS/"10_sensor_slopes.png")

    # ── MLflow ────────────────────────────────────────────────────────────
    setup_mlflow("Evaluation")
    with mlflow.start_run(run_name="evaluate"):
        def safe_key(s):
            return re.sub(r"[^A-Za-z0-9_.\-/ ]", "_", s).replace(" ", "_")
        flat = {safe_key(f"{m}_{k}"): v
                for m, mv in metrics.items()
                if not m.startswith("_") and isinstance(mv, dict)
                for k, v in mv.items() if k != "n_samples"}
        mlflow.log_metrics(flat)
        for p_path in PLOTS.glob("*.png"):
            mlflow.log_artifact(str(p_path))

    print("[evaluate] Done. Plots -> reports/plots/")


if __name__ == "__main__":
    main()
