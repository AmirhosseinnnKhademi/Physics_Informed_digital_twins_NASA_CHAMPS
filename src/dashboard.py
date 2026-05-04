#!/usr/bin/env python3
"""Generate reports/dashboard.html — self-contained HTML digital twin panel.

Usage:  .venv\\Scripts\\python.exe src/dashboard.py
Output: reports/dashboard.html
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
import yaml
from dotenv import load_dotenv

from models import SimpleGRUModel, RULWindowDataset
from features import add_rolling_features

load_dotenv()
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

# Sensor catalogue: (key, label, unit, degradation_dir)
# degradation_dir: +1 = rises with wear, -1 = drops with wear
SENSORS = [
    ("T24",  "LPC Exit Temp",       "°R",   +1),
    ("T30",  "HPC Exit Temp",       "°R",   +1),
    ("T48",  "HPT Exit Temp",       "°R",   +1),
    ("T50",  "LPT Exit Temp",       "°R",   +1),
    ("P15",  "Fan Static Press",    "psia", -1),
    ("P2",   "Inlet Total Press",   "psia", -1),
    ("P21",  "Fan Total Press",     "psia", -1),
    ("P24",  "LPC Exit Press",      "psia", -1),
    ("Ps30", "HPC Static Press",    "psia", -1),
    ("P40",  "HPT Inlet Press",     "psia", -1),
    ("P50",  "LPT Exit Press",      "psia", -1),
    ("Nf",   "Fan Speed",           "rpm",  -1),
    ("Nc",   "Core Speed",          "rpm",  -1),
    ("Wf",   "Fuel Flow",           "pps",  +1),
]

OP_COLS = [
    ("alt",  "Altitude",    "ft"),
    ("Mach", "Mach",        ""),
    ("TRA",  "Throttle",    "%"),
    ("T2",   "Inlet Temp",  "°R"),
]

# Sparkline sensors (6 most diagnostic)
SPARK_SENSORS = ["T50", "T30", "Ps30", "Wf", "Nf", "T48"]


# ── NASA scoring ─────────────────────────────────────────────────────────────
def nasa_score(actual, predicted):
    d = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(s))


# ── Load everything ───────────────────────────────────────────────────────────
def load_data():
    with open("data/processed/metadata.json") as f:
        meta = json.load(f)
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    test_units = meta["test_units"]            # [15]
    train_units = meta["train_units"]
    val_units   = meta["val_units"]
    all_units   = train_units + val_units + test_units

    all_df = pd.read_parquet("data/processed/all_cycles.parquet")

    return all_df, meta, test_units


def predict_gru(all_df, meta):
    with open("models/ft_gru_simple_scaler.pkl", "rb") as f:
        pkg = pickle.load(f)
    feature_cols = pkg["feature_cols"]
    window_size  = pkg["window_size"]
    hidden_size  = pkg["hidden_size"]
    dense_size   = pkg["dense_size"]
    dropout      = pkg["dropout"]
    rul_cap      = pkg["rul_cap"]
    scaler       = pkg["scaler"]

    test_df = all_df[all_df["unit"].isin(meta["test_units"])].sort_values(["unit","cycle"]).copy()
    # Save original identifiers before scaling (feature_cols includes "cycle")
    orig_unit  = test_df["unit"].values.copy()
    orig_cycle = test_df["cycle"].values.copy()
    orig_rul   = test_df["RUL"].values.copy()

    X_scaled = scaler.transform(test_df[feature_cols].values)
    test_df_sc = test_df.copy()
    test_df_sc[feature_cols] = X_scaled

    dataset = RULWindowDataset(test_df_sc, feature_cols, window_size)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)

    model = SimpleGRUModel(
        input_size  = len(feature_cols),
        hidden_size = hidden_size,
        dense_size  = dense_size,
        dropout     = dropout,
    ).to(DEVICE)
    state = torch.load("models/ft_gru_simple_rul.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy().ravel())
    preds = np.concatenate(preds) * rul_cap   # denormalize

    n = len(dataset)
    rows = []
    for i in range(n):
        rows.append({
            "unit":       int(orig_unit[-n:][i]),
            "cycle":      int(round(orig_cycle[-n:][i])),
            "RUL_actual": float(orig_rul[-n:][i]),
            "RUL_gru":    float(max(0, preds[i])),
        })
    return pd.DataFrame(rows)


def predict_xgb(all_df, meta):
    with open("models/ft_xgb_scaler.pkl", "rb") as f:
        pkg = pickle.load(f)
    feature_cols = pkg["feature_cols"]
    scaler       = pkg["scaler"]

    test_df = all_df[all_df["unit"].isin(meta["test_units"])].sort_values(["unit","cycle"]).copy()
    X = scaler.transform(test_df[feature_cols])
    model = xgb.XGBRegressor()
    model.load_model("models/ft_xgb_rul.ubj")
    preds = model.predict(X)

    df = test_df[["unit", "cycle", "RUL"]].rename(columns={"RUL": "RUL_actual"}).copy()
    df["unit"]    = df["unit"].astype(int)
    df["cycle"]   = df["cycle"].astype(int)
    df["RUL_xgb"] = preds.astype(float)
    return df.reset_index(drop=True)


def predict_xgb_rolling(all_df, meta):
    with open("models/ft_xgb_rolling_scaler.pkl", "rb") as f:
        pkg = pickle.load(f)
    stat_cols        = pkg["stat_cols"]
    all_feature_cols = pkg["all_feature_cols"]
    W                = pkg["rolling_window"]
    scaler           = pkg["scaler"]

    enriched      = add_rolling_features(all_df, stat_cols, W)
    test_enriched = enriched[enriched["unit"].isin(meta["test_units"])].sort_values(["unit","cycle"]).copy()
    X = scaler.transform(test_enriched[all_feature_cols])

    model = xgb.XGBRegressor()
    model.load_model("models/ft_xgb_rolling_rul.ubj")
    preds = model.predict(X)

    df = test_enriched[["unit", "cycle", "RUL"]].rename(columns={"RUL": "RUL_actual"}).copy()
    df["unit"]          = df["unit"].astype(int)
    df["cycle"]         = df["cycle"].astype(int)
    df["RUL_xgb_roll"]  = preds.astype(float)
    return df.reset_index(drop=True)


# ── Assemble JSON payload ─────────────────────────────────────────────────────
def build_payload(all_df, meta, gru_df, xgb_df, xgbr_df):
    unit = meta["test_units"][0]   # engine 15
    unit_df = all_df[all_df["unit"] == unit].sort_values("cycle").reset_index(drop=True)

    cycles      = unit_df["cycle"].tolist()
    rul_actual  = unit_df["RUL"].tolist()

    # Align predictions to cycle index
    def align(pred_df, col):
        merged = unit_df[["cycle"]].merge(
            pred_df[pred_df["unit"] == unit][["cycle", col]], on="cycle", how="left")
        return [round(float(v), 2) if pd.notna(v) else None for v in merged[col]]

    rul_gru  = align(gru_df,  "RUL_gru")
    rul_xgb  = align(xgb_df,  "RUL_xgb")
    rul_xgbr = align(xgbr_df, "RUL_xgb_roll")

    # Ensemble (mean/std of 3 predictions per cycle)
    def to_float_list(lst):
        out = []
        for x in lst:
            try:
                v = float(x)
                out.append(v if not np.isnan(v) else np.nan)
            except (TypeError, ValueError):
                out.append(np.nan)
        return out

    arr = np.array([to_float_list(rul_gru),
                    to_float_list(rul_xgb),
                    to_float_list(rul_xgbr)], dtype=float)
    ens_mean_np = np.nanmean(arr, axis=0)
    ens_std_np  = np.nanstd(arr,  axis=0)
    # Replace NaN with None for JSON serialisation
    ens_mean = [round(float(v), 2) if not np.isnan(v) else None for v in ens_mean_np]
    ens_std  = [round(float(v), 2) if not np.isnan(v) else None for v in ens_std_np]

    # Sensor time series (mean statistic per cycle)
    sensors = {}
    for key, *_ in SENSORS:
        col = f"{key}_mean"
        if col in unit_df.columns:
            sensors[key] = unit_df[col].round(4).tolist()

    # Operational conditions
    ops = {}
    for key, *_ in OP_COLS:
        col = f"{key}_mean"
        if col in unit_df.columns:
            ops[key] = unit_df[col].round(4).tolist()

    # Baselines: mean of first 5 cycles
    baselines = {}
    for key, *_ in SENSORS:
        col = f"{key}_mean"
        if col in unit_df.columns:
            baselines[key] = float(np.nanmean(unit_df[col].values[:5]))

    # Model performance metrics
    perf = {}
    for model_name, pred_col, pred_df_col_df in [
        ("GRU-Simple",    "RUL_gru",       gru_df),
        ("XGBoost",       "RUL_xgb",       xgb_df),
        ("XGBoost-Roll",  "RUL_xgb_roll",  xgbr_df),
    ]:
        sub = pred_df_col_df[pred_df_col_df["unit"] == unit].copy()
        if sub.empty:
            continue
        act = sub["RUL_actual"].values
        prd = sub[pred_col].values
        rmse = float(np.sqrt(np.mean((prd - act) ** 2)))
        mae  = float(np.mean(np.abs(prd - act)))
        ss_res = np.sum((act - prd) ** 2)
        ss_tot = np.sum((act - act.mean()) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        ns   = nasa_score(act, prd)
        perf[model_name] = {"rmse": round(rmse, 3), "mae": round(mae, 3),
                             "r2": round(r2, 4), "nasa_score": round(ns, 1)}

    return {
        "engine_code": f"ENG-0{unit}",
        "n_cycles":    len(cycles),
        "cycles":      cycles,
        "rul_actual":  rul_actual,
        "rul_gru":     rul_gru,
        "rul_xgb":     rul_xgb,
        "rul_xgbr":    rul_xgbr,
        "ens_mean":    ens_mean,
        "ens_std":     ens_std,
        "sensors":     sensors,
        "ops":         ops,
        "baselines":   baselines,
        "sensor_meta": [[k, lbl, unit, d] for k, lbl, unit, d in SENSORS],
        "op_meta":     [[k, lbl, u]       for k, lbl, u in OP_COLS],
        "spark_keys":  SPARK_SENSORS,
        "perf":        perf,
        "generated":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NASA CHAMPS — Predictive Maintenance Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#06101e;--s1:#0b1929;--border:#1a3352;--text:#c8dff0;--muted:#4a6a88;
  --cyan:#00cfff;--purple:#a855f7;--orange:#ff8c42;--green:#00e676;
  --red:#ff4545;--amber:#ffbb33;
  font-family:'Courier New',monospace;font-size:13px;
}
html,body{height:100%;background:var(--bg);color:var(--text);overflow:hidden}
/* ── 5-row layout: header | kpi-bar | engine | main | sparks   slider is fixed ── */
.layout{
  display:grid;
  grid-template-rows:52px 110px 1fr 170px 130px;
  height:100vh;gap:5px;padding:5px 5px 52px;
}
/* header */
.header{background:var(--s1);border:1px solid var(--border);border-radius:6px;
  display:flex;align-items:center;justify-content:space-between;padding:0 18px}
.logo{font-size:1.25rem;font-weight:700;color:#fff;letter-spacing:.05em}
.logo span{color:var(--cyan)}
.badge{font-size:.8rem;padding:2px 10px;border-radius:3px;
  background:rgba(0,207,255,.12);color:var(--cyan);border:1px solid rgba(0,207,255,.3)}
.hdr-r{display:flex;align-items:center;gap:20px;font-size:.85rem;color:var(--muted)}
.hdr-r b{color:var(--text)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 6px var(--green);display:inline-block;margin-right:6px;
  animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
/* kpi bar — 8 cards in one row */
.kpi-bar{display:grid;grid-template-columns:repeat(8,1fr);gap:5px}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:6px;
  padding:8px 10px;display:flex;flex-direction:column;justify-content:center;
  position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--cyan))}
.kpi-lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.kpi-val{font-size:1.45rem;font-weight:700;line-height:1;color:var(--accent,var(--cyan))}
.kpi-sub{font-size:.7rem;color:var(--muted);margin-top:3px}
/* engine + main side by side */
.mid-row{display:grid;grid-template-columns:420px 1fr;gap:5px}
/* panels */
.panel{background:var(--s1);border:1px solid var(--border);border-radius:6px;
  padding:8px 12px;display:flex;flex-direction:column;overflow:hidden}
.ptitle{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin-bottom:6px;display:flex;align-items:center;gap:6px}
.ptitle::after{content:'';flex:1;height:1px;background:var(--border)}
.chart-wrap{flex:1;position:relative;min-height:0}
/* engine panel */
.eng-panel{background:var(--s1);border:1px solid var(--border);border-radius:6px;
  padding:6px 10px;display:flex;flex-direction:column;overflow:hidden}
#engine-svg{width:100%;flex:1;display:block}
/* sensor table */
.stbl{width:100%;border-collapse:collapse;font-size:.8rem}
.stbl th{color:var(--muted);font-size:.72rem;padding:3px 5px;
  border-bottom:1px solid var(--border);font-weight:normal;text-align:left}
.stbl td{padding:4px 5px;border-bottom:1px solid rgba(26,51,82,.5)}
.stbl tr:hover td{background:rgba(255,255,255,.03)}
/* sparks */
.sparks{display:grid;grid-template-columns:repeat(6,1fr);gap:5px}
.spk{background:var(--s1);border:1px solid var(--border);border-radius:6px;
  padding:5px 8px;display:flex;flex-direction:column}
.spk-lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase}
.spk-val{font-size:1.05rem;font-weight:700;line-height:1.2}
.spk-dev{font-size:.7rem;margin-top:1px}
.spk-c{flex:1;position:relative;min-height:0;height:36px;margin-top:3px}
/* slider */
.slider-bar{position:fixed;bottom:5px;left:5px;right:5px;height:42px;z-index:99;
  background:var(--s1);border:1px solid var(--border);border-radius:6px;
  display:flex;align-items:center;gap:10px;padding:0 14px}
#cyc-sl{flex:1;-webkit-appearance:none;height:3px;background:var(--border);border-radius:2px;outline:none}
#cyc-sl::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--cyan);cursor:pointer;box-shadow:0 0 6px var(--cyan)}
.btn{background:none;border:1px solid var(--border);color:var(--text);border-radius:4px;
  padding:4px 12px;cursor:pointer;font-size:.85rem;font-family:inherit;
  transition:border-color .15s,color .15s}
.btn:hover,.btn.on{border-color:var(--cyan);color:var(--cyan);background:rgba(0,207,255,.08)}
#cyc-lbl{font-size:.9rem;min-width:78px;text-align:center}
#rul-bdg{font-size:.88rem;padding:3px 10px;border-radius:3px;
  border:1px solid;font-weight:700;min-width:100px;text-align:center}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>
<div class="layout">

<!-- HEADER -->
<header class="header">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="logo">NASA <span>CHAMPS</span></div>
    <div class="badge">N-CMAPSS DS03-012</div>
    <div class="badge" id="hdr-engine">ENG-015</div>
  </div>
  <div class="hdr-r">
    <span><span class="dot"></span>Live Simulation</span>
    <span>GRU-Simple + XGBoost Ensemble</span>
    <span id="hdr-gen" style="font-size:.72rem"></span>
  </div>
</header>

<!-- KPI BAR (8 cards) -->
<div class="kpi-bar">
  <div class="kpi" style="--accent:var(--green)">
    <div class="kpi-lbl">Best Model RUL</div>
    <div class="kpi-val" id="kpi-rul-pred">--</div>
    <div class="kpi-sub">GRU-Simple</div>
  </div>
  <div class="kpi" style="--accent:var(--cyan)">
    <div class="kpi-lbl">Actual RUL</div>
    <div class="kpi-val" id="kpi-rul-actual">--</div>
    <div class="kpi-sub">Ground truth</div>
  </div>
  <div class="kpi" style="--accent:var(--amber)">
    <div class="kpi-lbl">Ensemble ±σ</div>
    <div class="kpi-val" id="kpi-ens-std">--</div>
    <div class="kpi-sub">3-model spread</div>
  </div>
  <div class="kpi" style="--accent:var(--red)">
    <div class="kpi-lbl">Risk Level</div>
    <div class="kpi-val" id="kpi-risk" style="font-size:1.1rem">--</div>
    <div class="kpi-sub" id="kpi-risk-sub">GRU prediction</div>
  </div>
  <div class="kpi" style="--accent:var(--orange)">
    <div class="kpi-lbl">HPC Temp T30</div>
    <div class="kpi-val" id="kpi-T30">--</div>
    <div class="kpi-sub" id="kpi-T30-dev">°R</div>
  </div>
  <div class="kpi" style="--accent:#e879f9">
    <div class="kpi-lbl">LPT Temp T50</div>
    <div class="kpi-val" id="kpi-T50">--</div>
    <div class="kpi-sub" id="kpi-T50-dev">°R</div>
  </div>
  <div class="kpi" style="--accent:var(--cyan)">
    <div class="kpi-lbl">Flight Phase</div>
    <div class="kpi-val" id="kpi-phase" style="font-size:1rem">--</div>
    <div class="kpi-sub" id="kpi-phase-sub">--</div>
  </div>
  <div class="kpi" style="--accent:var(--amber)">
    <div class="kpi-lbl">Dominant Fault</div>
    <div class="kpi-val" id="kpi-fault" style="font-size:1rem">--</div>
    <div class="kpi-sub" id="kpi-fault-sub">--</div>
  </div>
</div>

<!-- ENGINE + MAIN CHARTS side by side -->
<div class="mid-row">

  <!-- ENGINE DIGITAL TWIN -->
  <div class="eng-panel">
    <div class="ptitle">Digital Twin — Engine Cross-Section</div>
    <svg id="engine-svg" viewBox="0 0 420 310" preserveAspectRatio="xMidYMid meet">
      <defs>
        <polygon id="fblade" points="22,-3 88,-11 88,-3.5 22,3.5" fill="#7a8492" stroke="#50586a" stroke-width="0.4"/>
        <radialGradient id="fan-bg" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#1a4a72" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#0a1e35" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="comb-glow" cx="50%" cy="50%" r="55%">
          <stop offset="0%" stop-color="#ff7700" stop-opacity="0.7"/>
          <stop offset="50%" stop-color="#ff3300" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#aa1100" stop-opacity="0"/>
        </radialGradient>
        <marker id="ai" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#00cfff"/></marker>
        <marker id="ae" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#ff5500"/></marker>
        <marker id="ab" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#1e4a70"/></marker>
      </defs>

      <!-- Cold air background (fan section) -->
      <rect x="16" y="4" width="130" height="302" rx="5" fill="#0d2540"/>
      <circle cx="81" cy="155" r="130" fill="url(#fan-bg)"/>

      <!-- Outer nacelle shell -->
      <path d="M 16,155 C 16,28 48,4 81,4 L 358,10 C 392,12 414,60 414,155 C 414,250 392,298 358,300 L 81,306 C 48,306 16,282 16,155 Z" fill="none" stroke="#1e4060" stroke-width="2"/>

      <!-- Fan disc (large) -->
      <circle cx="81" cy="155" r="118" fill="#0c1f35" stroke="#1e3a55" stroke-width="1.5"/>
      <!-- 24 fan blades -->
      <g transform="translate(81,155)">
        <use href="#fblade" transform="rotate(0)"/>   <use href="#fblade" transform="rotate(15)"/>
        <use href="#fblade" transform="rotate(30)"/>  <use href="#fblade" transform="rotate(45)"/>
        <use href="#fblade" transform="rotate(60)"/>  <use href="#fblade" transform="rotate(75)"/>
        <use href="#fblade" transform="rotate(90)"/>  <use href="#fblade" transform="rotate(105)"/>
        <use href="#fblade" transform="rotate(120)"/> <use href="#fblade" transform="rotate(135)"/>
        <use href="#fblade" transform="rotate(150)"/> <use href="#fblade" transform="rotate(165)"/>
        <use href="#fblade" transform="rotate(180)"/> <use href="#fblade" transform="rotate(195)"/>
        <use href="#fblade" transform="rotate(210)"/> <use href="#fblade" transform="rotate(225)"/>
        <use href="#fblade" transform="rotate(240)"/> <use href="#fblade" transform="rotate(255)"/>
        <use href="#fblade" transform="rotate(270)"/> <use href="#fblade" transform="rotate(285)"/>
        <use href="#fblade" transform="rotate(300)"/> <use href="#fblade" transform="rotate(315)"/>
        <use href="#fblade" transform="rotate(330)"/> <use href="#fblade" transform="rotate(345)"/>
        <circle r="24" fill="#2a3040" stroke="#404858" stroke-width="1.5"/>
        <path d="M -24,0 C -20,-10 -4,-16 8,-16 L 8,16 C -4,16 -20,10 -24,0 Z" fill="#3a424e" stroke="#505868" stroke-width="1"/>
      </g>
      <!-- Fan sensors -->
      <text id="eng-Nf"  x="118" y="148" text-anchor="middle" font-family="'Courier New',monospace" font-size="11" font-weight="bold" fill="#00cfff">Nf: --</text>
      <text id="eng-P15" x="118" y="164" text-anchor="middle" font-family="'Courier New',monospace" font-size="10" fill="#3a8aa8">P15: --</text>

      <!-- Bypass ducts -->
      <polygon points="145,4 358,10 358,46 145,48" fill="#091a2c" stroke="#1a3852" stroke-width="0.8"/>
      <polygon points="145,262 358,264 358,300 145,306" fill="#091a2c" stroke="#1a3852" stroke-width="0.8"/>
      <!-- bypass arrows -->
      <g stroke="#1a4060" stroke-width="1">
        <line x1="168" y1="27" x2="198" y2="27" marker-end="url(#ab)"/>
        <line x1="240" y1="29" x2="270" y2="29" marker-end="url(#ab)"/>
        <line x1="168" y1="283" x2="198" y2="283" marker-end="url(#ab)"/>
        <line x1="240" y1="281" x2="270" y2="281" marker-end="url(#ab)"/>
      </g>

      <!-- LPC -->
      <polygon id="eng-comp-lpc" points="145,48 210,54 210,256 145,262" fill="#0e2234" stroke="#1a3852" stroke-width="1"/>
      <g fill="#2e4060" stroke="#486080" stroke-width="0.8">
        <ellipse cx="162" cy="155" rx="4" ry="96"/>
        <ellipse cx="178" cy="155" rx="4" ry="90"/>
        <ellipse cx="196" cy="155" rx="4" ry="84"/>
      </g>
      <text id="eng-T24" x="175" y="148" text-anchor="middle" font-family="'Courier New',monospace" font-size="10" font-weight="bold" fill="#c8dff0">T24:--</text>
      <text id="eng-P24" x="175" y="162" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">P24:--</text>
      <text x="175" y="273" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">LPC</text>

      <!-- HPC -->
      <polygon id="eng-comp-hpc" points="210,54 270,60 270,250 210,256" fill="#0e2438" stroke="#1a3852" stroke-width="1"/>
      <g fill="#283848" stroke="#3a5068" stroke-width="0.8">
        <ellipse cx="226" cy="155" rx="4" ry="78"/>
        <ellipse cx="244" cy="155" rx="4" ry="72"/>
        <ellipse cx="262" cy="155" rx="4" ry="66"/>
      </g>
      <text id="eng-T30"  x="240" y="148" text-anchor="middle" font-family="'Courier New',monospace" font-size="10" font-weight="bold" fill="#c8dff0">T30:--</text>
      <text id="eng-Ps30" x="240" y="162" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">Ps30:--</text>
      <text x="240" y="267" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">HPC</text>

      <!-- COMBUSTOR -->
      <polygon id="eng-comp-comb" points="270,40 330,40 330,270 270,270" fill="#220e00" stroke="#773300" stroke-width="1.2"/>
      <ellipse cx="300" cy="155" rx="42" ry="86" fill="url(#comb-glow)"/>
      <g stroke="#bb4400" stroke-width="1.5" opacity="0.7">
        <line x1="278" y1="46" x2="278" y2="58"/><line x1="292" y1="43" x2="292" y2="55"/>
        <line x1="308" y1="43" x2="308" y2="55"/><line x1="322" y1="46" x2="322" y2="58"/>
        <line x1="278" y1="264" x2="278" y2="252"/><line x1="292" y1="267" x2="292" y2="255"/>
        <line x1="308" y1="267" x2="308" y2="255"/><line x1="322" y1="264" x2="322" y2="252"/>
      </g>
      <text id="eng-Wf"   x="300" y="146" text-anchor="middle" font-family="'Courier New',monospace" font-size="11" font-weight="bold" fill="#ff8c42">Wf:--</text>
      <text id="eng-T48c" x="300" y="164" text-anchor="middle" font-family="'Courier New',monospace" font-size="10" fill="#ffbb33">T48:--</text>
      <text x="300" y="281" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#774400">COMB</text>

      <!-- HPT -->
      <polygon id="eng-comp-hpt" points="330,60 370,66 370,244 330,250" fill="#2a0a00" stroke="#773300" stroke-width="1"/>
      <g fill="#6a2808" stroke="#9a4020" stroke-width="0.8">
        <ellipse cx="344" cy="155" rx="4" ry="62"/>
        <ellipse cx="360" cy="155" rx="4" ry="58"/>
      </g>
      <text id="eng-T48" x="350" y="148" text-anchor="middle" font-family="'Courier New',monospace" font-size="10" font-weight="bold" fill="#ff8c42">T48:--</text>
      <text id="eng-P40" x="350" y="163" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#664400">P40:--</text>
      <text x="350" y="260" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#663300">HPT</text>

      <!-- LPT -->
      <polygon id="eng-comp-lpt" points="370,66 400,62 400,248 370,244" fill="#1e0800" stroke="#553300" stroke-width="1"/>
      <g fill="#501e06" stroke="#7a3218" stroke-width="0.8">
        <ellipse cx="383" cy="155" rx="4" ry="58"/>
        <ellipse cx="396" cy="155" rx="4" ry="60"/>
      </g>
      <text id="eng-T50" x="385" y="148" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" font-weight="bold" fill="#ff8c42">T50:--</text>
      <text id="eng-P50" x="385" y="161" text-anchor="middle" font-family="'Courier New',monospace" font-size="8" fill="#553300">P50:--</text>
      <text x="385" y="256" text-anchor="middle" font-family="'Courier New',monospace" font-size="8" fill="#553300">LPT</text>

      <!-- NOZZLE -->
      <polygon points="400,10 414,155 400,300" fill="#0c0400" stroke="#442200" stroke-width="1.5"/>
      <polygon id="eng-comp-noz" points="400,62 412,155 400,248" fill="#2e0e00" stroke="#773300" stroke-width="1"/>
      <text id="eng-Nc" x="406" y="150" text-anchor="middle" font-family="'Courier New',monospace" font-size="8" font-weight="bold" fill="#ff8c42">Nc:--</text>

      <!-- Intake arrows (left) -->
      <g stroke="#00cfff" stroke-width="2.5" opacity="0.8">
        <line x1="2" y1="110" x2="14" y2="110" marker-end="url(#ai)"/>
        <line x1="2" y1="155" x2="14" y2="155" marker-end="url(#ai)"/>
        <line x1="2" y1="200" x2="14" y2="200" marker-end="url(#ai)"/>
      </g>
      <!-- Exhaust arrows (right) -->
      <g stroke="#ff5500" stroke-width="2.5" opacity="0.85">
        <line x1="414" y1="138" x2="418" y2="138" marker-end="url(#ae)"/>
        <line x1="414" y1="155" x2="418" y2="155" marker-end="url(#ae)"/>
        <line x1="414" y1="172" x2="418" y2="172" marker-end="url(#ae)"/>
      </g>

      <!-- Component labels -->
      <text x="81"  y="304" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">Fan</text>
      <text x="175" y="285" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">LPC</text>
      <text x="240" y="279" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#4a6a88">HPC</text>
      <text x="300" y="283" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#996600">Combustor</text>
      <text x="360" y="272" text-anchor="middle" font-family="'Courier New',monospace" font-size="9" fill="#774400">Turbine</text>
    </svg>
  </div>

  <!-- RIGHT: RUL chart stacked on sensor table -->
  <div style="display:grid;grid-template-rows:1fr 1fr;gap:5px">
    <div class="panel">
      <div class="ptitle">RUL Timeline</div>
      <div class="chart-wrap"><canvas id="rul-chart"></canvas></div>
    </div>
    <div class="panel" style="padding:6px 10px">
      <div class="ptitle">Sensor Telemetry</div>
      <div style="flex:1;overflow-y:auto">
        <table class="stbl">
          <thead><tr>
            <th>Sensor</th><th style="text-align:right">Value</th>
            <th style="text-align:right">Unit</th><th style="text-align:right">Dev%</th>
          </tr></thead>
          <tbody id="sensor-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- SPARKLINES ROW -->
<div class="sparks" id="sparks-row"></div>

<!-- BOTTOM: metrics + residuals -->
<div style="display:grid;grid-template-columns:340px 1fr;gap:5px">
  <div class="panel" style="padding:6px 10px">
    <div class="ptitle">Model Performance</div>
    <table class="stbl">
      <thead><tr>
        <th>Model</th><th style="text-align:right">RMSE</th>
        <th style="text-align:right">MAE</th><th style="text-align:right">R²</th>
        <th style="text-align:right">NASA</th>
      </tr></thead>
      <tbody id="metrics-tbody"></tbody>
    </table>
  </div>
  <div class="panel">
    <div class="ptitle">Prediction Residuals</div>
    <div class="chart-wrap"><canvas id="res-chart"></canvas></div>
  </div>
</div>

</div><!-- /layout -->

<!-- SLIDER BAR -->
<div class="slider-bar">
  <button class="btn on" id="btn-play">▶ Play</button>
  <button class="btn" id="btn-reset">↺ Reset</button>
  <input type="range" id="cyc-sl" min="0" step="1" value="0">
  <div id="cyc-lbl">Cycle 1</div>
  <div id="rul-bdg">RUL --</div>
</div>

<script>
const D = /*DTDATA*/null;

/* ── Constants ── */
const C_GRU  = '#00cfff', C_XGB = '#ff8c42', C_XGBR = '#a855f7';
const C_ACT  = '#00e676', C_ENS = 'rgba(0,207,255,0.15)';
const C_RED  = '#ff4545', C_AMBER = '#ffbb33', C_GREEN = '#00e676', C_MUTED = '#4a6a88';
const SPARK_COLORS = {T50:'#e879f9',T30:'#ff8c42',Ps30:'#00cfff',Wf:'#00e676',Nf:'#ffbb33',T48:'#a855f7'};

/* ── Chart defaults ── */
Chart.defaults.color='#4a6a88';Chart.defaults.borderColor='#1a3352';
Chart.defaults.font.family="'Courier New',monospace";Chart.defaults.font.size=11;

/* ── Vertical cursor plugin ── */
const vline={id:'vline',afterDraw(c,_,o){
  if(o.xi===undefined)return;
  const m=c.getDatasetMeta(0);if(!m||!m.data[o.xi])return;
  const x=m.data[o.xi].x,{ctx,chartArea:{top,bottom}}=c;
  ctx.save();ctx.strokeStyle=o.color||'rgba(255,255,255,0.4)';
  ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.restore();
}};

/* ── RUL Chart ── */
const N=D.n_cycles;
const ensUp=D.ens_mean.map((v,i)=>v!==null?+(v+D.ens_std[i]).toFixed(2):null);
const ensLo=D.ens_mean.map((v,i)=>v!==null?+(v-D.ens_std[i]).toFixed(2):null);
const rulChart=new Chart(document.getElementById('rul-chart'),{
  type:'line',plugins:[vline],
  data:{labels:D.cycles,datasets:[
    {label:'Ens Upper',data:ensUp,borderWidth:0,pointRadius:0,fill:'+1',backgroundColor:C_ENS,tension:.3,order:10},
    {label:'Ens Lower',data:ensLo,borderWidth:0,pointRadius:0,fill:false,tension:.3,order:10},
    {label:'Actual',data:D.rul_actual,borderColor:C_ACT,borderWidth:2,borderDash:[6,3],pointRadius:0,tension:.2,order:1},
    {label:'GRU-Simple',data:D.rul_gru,borderColor:C_GRU,borderWidth:2,pointRadius:0,tension:.3,order:2},
    {label:'XGBoost-Roll',data:D.rul_xgbr,borderColor:C_XGBR,borderWidth:1.5,borderDash:[3,2],pointRadius:0,tension:.3,order:3},
    {label:'XGBoost',data:D.rul_xgb,borderColor:C_XGB,borderWidth:1.5,borderDash:[2,4],pointRadius:0,tension:.3,order:4},
    {label:'Critical (30)',data:Array(N).fill(30),borderColor:C_RED,borderWidth:1,borderDash:[8,4],pointRadius:0,fill:false,order:9},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    interaction:{mode:'index',intersect:false},
    scales:{
      x:{title:{display:true,text:'Flight Cycle',color:C_MUTED,font:{size:10}},ticks:{maxTicksLimit:8},grid:{color:'rgba(26,51,82,0.5)'}},
      y:{title:{display:true,text:'RUL (cycles)',color:C_MUTED,font:{size:10}},min:0,grid:{color:'rgba(26,51,82,0.5)'}}
    },
    plugins:{
      legend:{position:'top',labels:{boxWidth:14,padding:10,color:'#c8dff0',font:{size:10}}},
      tooltip:{callbacks:{title:i=>`Cycle ${i[0].label}`,label:i=>i.dataset.label.startsWith('Ens')?null:`${i.dataset.label}: ${(+i.raw).toFixed(1)} cyc`}},
      vline:{xi:0,color:'rgba(0,207,255,0.5)'}
    }}
});

/* ── Residuals Chart ── */
const resChart=new Chart(document.getElementById('res-chart'),{
  type:'line',
  data:{labels:D.cycles,datasets:[
    {label:'GRU-Simple',data:D.rul_gru.map((v,i)=>v!==null?+(v-D.rul_actual[i]).toFixed(2):null),borderColor:C_GRU,borderWidth:1.5,pointRadius:0,tension:.3},
    {label:'XGBoost-Roll',data:D.rul_xgbr.map((v,i)=>v!==null?+(v-D.rul_actual[i]).toFixed(2):null),borderColor:C_XGBR,borderWidth:1.5,borderDash:[3,2],pointRadius:0,tension:.3},
    {label:'XGBoost',data:D.rul_xgb.map((v,i)=>v!==null?+(v-D.rul_actual[i]).toFixed(2):null),borderColor:C_XGB,borderWidth:1.5,borderDash:[2,4],pointRadius:0,tension:.3},
    {label:'Zero',data:Array(N).fill(0),borderColor:C_MUTED,borderWidth:1,borderDash:[6,4],pointRadius:0},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
    scales:{x:{ticks:{maxTicksLimit:8},grid:{color:'rgba(26,51,82,0.5)'}},y:{grid:{color:'rgba(26,51,82,0.5)'}}},
    plugins:{legend:{position:'top',labels:{boxWidth:12,padding:8,color:'#c8dff0',font:{size:10}}},
      tooltip:{callbacks:{label:i=>i.dataset.label==='Zero'?null:`${i.dataset.label}: ${(+i.raw).toFixed(1)} cyc`}}}}
});

/* ── Sparklines ── */
const sparkCharts={};
D.spark_keys.forEach(key=>{
  const color=SPARK_COLORS[key]||'#00cfff';
  const div=document.createElement('div');div.className='spk';
  div.innerHTML=`<div class="spk-lbl">${key}</div><div class="spk-val" id="spk-v-${key}">--</div><div class="spk-dev" id="spk-d-${key}" style="color:${C_MUTED}">--</div><div class="spk-c"><canvas id="spk-${key}"></canvas></div>`;
  document.getElementById('sparks-row').appendChild(div);
  sparkCharts[key]=new Chart(document.getElementById(`spk-${key}`),{
    type:'line',data:{labels:D.cycles,datasets:[{data:D.sensors[key]||[],borderColor:color,borderWidth:1.5,pointRadius:0,tension:.4,fill:true,backgroundColor:color+'22'}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:false},y:{display:false}},plugins:{legend:{display:false},tooltip:{enabled:false}}}
  });
});

/* ── Sensor table ── */
(function(){
  const tb=document.getElementById('sensor-tbody');
  D.sensor_meta.forEach(([key,label,unit])=>{
    const tr=document.createElement('tr');tr.id=`sr-${key}`;
    tr.innerHTML=`<td>${key} <span style="color:var(--muted);font-size:.72rem">${label}</span></td><td style="text-align:right;font-weight:700" id="sv-${key}">--</td><td style="text-align:right;color:var(--muted)">${unit}</td><td style="text-align:right" id="sd-${key}">--</td>`;
    tb.appendChild(tr);
  });
})();

/* ── Metrics table ── */
(function(){
  const tb=document.getElementById('metrics-tbody');
  const models=Object.keys(D.perf);
  const bestR=Math.min(...models.map(m=>D.perf[m].rmse));
  const MC={GRU-Simple:C_GRU,XGBoost:C_XGB};MC['XGBoost-Roll']=C_XGBR;
  models.forEach(m=>{
    const p=D.perf[m],best=p.rmse===bestR,c=MC[m]||'#c8dff0';
    const tr=document.createElement('tr');
    tr.innerHTML=`<td style="color:${c}">${m}${best?' ★':''}</td><td style="text-align:right" class="${best?'best':''}">${p.rmse.toFixed(3)}</td><td style="text-align:right">${p.mae.toFixed(3)}</td><td style="text-align:right">${p.r2.toFixed(4)}</td><td style="text-align:right">${p.nasa_score.toFixed(0)}</td>`;
    tb.appendChild(tr);
  });
  const s=document.createElement('style');s.textContent='.best{color:var(--green)}';document.head.appendChild(s);
})();

/* ── Dev helpers ── */
function devColor(pct,dir){const isDeg=dir>0?pct>0:pct<0,a=Math.abs(pct);if(a<0.3)return C_MUTED;if(!isDeg)return C_GREEN;return a>3?C_RED:C_AMBER}
function fmtDev(cur,key){
  const m=D.sensor_meta.find(x=>x[0]===key);if(!m)return{text:'--',color:C_MUTED};
  const base=D.baselines[key];if(!base||Math.abs(base)<1e-6)return{text:'--',color:C_MUTED};
  const pct=(cur-base)/Math.abs(base)*100,color=devColor(pct,m[3]);
  return{text:(pct>=0?'+':'')+pct.toFixed(2)+'%',color};
}
function riskInfo(r){if(r===null)return{label:'Unknown',color:C_MUTED};if(r>50)return{label:'LOW',color:C_GREEN};if(r>30)return{label:'MEDIUM',color:C_AMBER};if(r>15)return{label:'HIGH',color:C_XGB};return{label:'CRITICAL',color:C_RED}}
function flightPhase(idx){
  const alt=(D.ops['alt']||[])[idx],mach=(D.ops['Mach']||[])[idx],tra=(D.ops['TRA']||[])[idx];
  if(alt==null)return{label:'Unknown',color:C_MUTED,sub:'--'};
  const k=(alt/1000).toFixed(0),ms=(mach||0).toFixed(2);
  if(alt<3000&&(mach||0)<0.15)return{label:'Ground',color:'#64748b',sub:`M ${ms}`};
  if(alt<15000&&(tra||0)>82)return{label:'Climb',color:C_GREEN,sub:`${k}k ft`};
  if(alt>28000&&(mach||0)>0.65)return{label:'Cruise',color:C_GRU,sub:`${k}k ft · M ${ms}`};
  if(alt<15000&&(tra||0)<68)return{label:'Approach',color:C_AMBER,sub:`${k}k ft`};
  return{label:'Transit',color:C_XGBR,sub:`${k}k ft`};
}
function dominantFault(idx){
  let mx=0,res={key:'Nominal',label:'No significant deviation',color:C_GREEN};
  D.sensor_meta.forEach(([key,label,unit,dir])=>{
    const val=(D.sensors[key]||[])[idx],base=D.baselines[key];
    if(val==null||!base||Math.abs(base)<1e-6)return;
    const pct=(val-base)/Math.abs(base)*100,isDeg=dir>0?pct>0:pct<0,deg=isDeg?Math.abs(pct):0;
    if(deg>mx){mx=deg;const s=pct>=0?'+':'';res={key,label:`${s}${pct.toFixed(2)}% ${label}`,color:deg>5?C_RED:deg>2?C_AMBER:C_MUTED};}
  });return res;
}

/* ── Engine SVG update ── */
const ENG_LABELS=[
  ['eng-Nf','Nf','Nf',0],['eng-P15','P15','P15',1],
  ['eng-T24','T24','T24',1],['eng-P24','P24','P24',1],
  ['eng-T30','T30','T30',1],['eng-Ps30','Ps30','Ps30',1],
  ['eng-Wf','Wf','Wf',2],['eng-T48c','T48','T48',1],
  ['eng-T48','T48','T48',1],['eng-P40','P40','P40',1],
  ['eng-T50','T50','T50',1],['eng-P50','P50','P50',1],['eng-Nc','Nc','Nc',0],
];
const ENG_COMPS={
  'eng-comp-lpc':{key:'T24',dir:+1,base:'#0e2234'},
  'eng-comp-hpc':{key:'T30',dir:+1,base:'#0e2438'},
  'eng-comp-comb':{key:'Wf',dir:+1,base:'#220e00'},
  'eng-comp-hpt':{key:'T48',dir:+1,base:'#2a0a00'},
  'eng-comp-lpt':{key:'T50',dir:+1,base:'#1e0800'},
  'eng-comp-noz':{key:'Nc',dir:-1,base:'#2e0e00'},
};
function updateEngineSVG(idx){
  ENG_LABELS.forEach(([id,key,lbl,dec])=>{
    const el=document.getElementById(id);if(!el)return;
    const val=(D.sensors[key]||[])[idx];
    if(val==null){el.textContent=lbl+':--';return;}
    el.textContent=lbl+':'+val.toFixed(dec);
    const dv=fmtDev(val,key);if(dv.color!==C_MUTED)el.setAttribute('fill',dv.color);
  });
  Object.entries(ENG_COMPS).forEach(([id,{key,dir,base}])=>{
    const el=document.getElementById(id);if(!el)return;
    const val=(D.sensors[key]||[])[idx],bv=D.baselines[key];
    if(val==null||!bv){el.setAttribute('fill',base);return;}
    const pct=(val-bv)/Math.abs(bv)*100,deg=(dir>0?pct>0:pct<0)?Math.abs(pct):0;
    el.setAttribute('fill',deg>5?'rgba(255,69,69,0.45)':deg>3?'rgba(255,140,66,0.3)':deg>1.5?'rgba(255,187,51,0.18)':base);
  });
}

/* ── Main update ── */
function updateFrame(idx){
  const cycle=D.cycles[idx],ra=D.rul_actual[idx],rg=D.rul_gru[idx],es=D.ens_std[idx];
  const fmt=v=>v!==null?Math.round(v):'--';
  document.getElementById('kpi-rul-actual').textContent=fmt(ra);
  document.getElementById('kpi-rul-pred').textContent=fmt(rg);
  document.getElementById('kpi-ens-std').textContent=es!==null?'±'+es.toFixed(1):'--';
  const risk=riskInfo(rg);
  const kr=document.getElementById('kpi-risk');kr.textContent=risk.label;kr.style.color=risk.color;
  document.getElementById('kpi-risk-sub').textContent=`GRU: ${fmt(rg)} cyc`;
  ['T30','T50'].forEach(k=>{
    const v=(D.sensors[k]||[])[idx];
    if(v==null)return;
    document.getElementById(`kpi-${k}`).textContent=v.toFixed(1);
    const dv=fmtDev(v,k);
    document.getElementById(`kpi-${k}-dev`).textContent=dv.text;
    document.getElementById(`kpi-${k}-dev`).style.color=dv.color;
  });
  const ph=flightPhase(idx);
  const pe=document.getElementById('kpi-phase');pe.textContent=ph.label;pe.style.color=ph.color;
  document.getElementById('kpi-phase-sub').textContent=ph.sub;
  const fa=dominantFault(idx);
  const fe=document.getElementById('kpi-fault');fe.textContent=fa.key;fe.style.color=fa.color;
  document.getElementById('kpi-fault-sub').textContent=fa.label;
  document.getElementById('cyc-lbl').textContent=`Cycle ${cycle}`;
  const bd=document.getElementById('rul-bdg');bd.textContent=`RUL ${fmt(rg)} cyc`;bd.style.color=risk.color;bd.style.borderColor=risk.color;
  D.sensor_meta.forEach(([key])=>{
    const val=(D.sensors[key]||[])[idx],ve=document.getElementById(`sv-${key}`),de=document.getElementById(`sd-${key}`);
    if(!ve)return;
    if(val==null){ve.textContent='--';de.textContent='--';}
    else{ve.textContent=val.toFixed(2);const dv=fmtDev(val,key);de.textContent=dv.text;de.style.color=dv.color;}
  });
  D.spark_keys.forEach(key=>{
    const val=(D.sensors[key]||[])[idx],ve=document.getElementById(`spk-v-${key}`),de=document.getElementById(`spk-d-${key}`);
    if(!ve)return;
    if(val==null){ve.textContent='--';de.textContent='--';}
    else{ve.textContent=val.toFixed(1);const dv=fmtDev(val,key);de.textContent=dv.text;de.style.color=dv.color;}
  });
  rulChart.options.plugins.vline.xi=idx;rulChart.update('none');
  updateEngineSVG(idx);
}

/* ── Slider & playback ── */
const sl=document.getElementById('cyc-sl');sl.max=N-1;
sl.addEventListener('input',()=>updateFrame(+sl.value));
let playing=false,timer=null;
function startPlay(){playing=true;document.getElementById('btn-play').textContent='⏸ Pause';document.getElementById('btn-play').classList.add('on');tick();}
function stopPlay(){playing=false;clearTimeout(timer);document.getElementById('btn-play').textContent='▶ Play';document.getElementById('btn-play').classList.remove('on');}
function tick(){if(!playing)return;let i=+sl.value;i=(i+1)%N;sl.value=i;updateFrame(i);timer=setTimeout(tick,350);}
document.getElementById('btn-play').addEventListener('click',()=>playing?stopPlay():startPlay());
document.getElementById('btn-reset').addEventListener('click',()=>{stopPlay();sl.value=0;updateFrame(0);});

/* ── Init ── */
document.getElementById('hdr-engine').textContent=D.engine_code;
document.getElementById('hdr-gen').textContent='Generated: '+D.generated;
updateFrame(0);
</script>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[dashboard] Loading data...")
    all_df, meta, test_units = load_data()

    print("[dashboard] Running GRU predictions...")
    gru_df  = predict_gru(all_df, meta)

    print("[dashboard] Running XGBoost predictions...")
    xgb_df  = predict_xgb(all_df, meta)

    print("[dashboard] Running XGBoost-Rolling predictions...")
    xgbr_df = predict_xgb_rolling(all_df, meta)

    print("[dashboard] Building data payload...")
    payload = build_payload(all_df, meta, gru_df, xgb_df, xgbr_df)

    json_str = json.dumps(payload, allow_nan=False)
    html_out = HTML.replace("/*DTDATA*/null", f"/*DTDATA*/{json_str}")

    out_path = REPORTS / "dashboard.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"[dashboard] Written → {out_path}  ({len(html_out)//1024} KB)")


if __name__ == "__main__":
    main()
