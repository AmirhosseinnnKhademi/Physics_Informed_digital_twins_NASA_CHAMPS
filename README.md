# Physics-Informed Digital Twin — NASA N-CMAPSS Turbofan Engine

**Dataset:** NASA N-CMAPSS DS03-012 · **Task:** Remaining Useful Life (RUL) prediction  
**Best model:** GRU-Simple · RMSE 2.63 · R² 0.943 · NASA Score 7.6  
**Pipeline:** DVC · **Tracking:** MLflow / DagsHub · **Visualisation:** self-contained HTML dashboard

---

## What This Is

A full ML pipeline that predicts how many flight cycles remain before a jet engine fails, trained on NASA's high-fidelity turbofan degradation simulator. The trained model drives an interactive **Physics-Informed Digital Twin** dashboard — an animated engine schematic plus physics-based anomaly detection that runs entirely in a browser with no server required.

---

## Dataset

**N-CMAPSS DS03-012** is NASA's continuous-time turbofan engine degradation dataset. Each engine accumulates flight cycles under realistic operating conditions until failure.

| Split | Engine units | Purpose |
|-------|-------------|---------|
| Training / Val | 1–9 (dev set) | Model learning and hyperparameter search |
| Test | 10–15 | Final held-out evaluation |

Each raw HDF5 file contains ~7,800 high-frequency sensor samples per flight cycle. The **prepare** stage aggregates these to **one row per cycle**.

### Signal groups

| Group | Signals | Description |
|-------|---------|-------------|
| `W` | alt, Mach, TRA, T2 | Operating conditions (flight envelope) |
| `X_s` | T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50, Nf, Nc, Wf | 14 physical sensors |
| `T` | fan_eff, fan_flow, LPC_eff, LPC_flow, HPC_eff, HPC_flow, HPT_eff, HPT_flow, LPT_eff, LPT_flow | Ground-truth degradation parameters (hidden at inference) |
| `A` | unit, cycle, Fc, hs | Auxiliary — `hs=1` healthy, `hs=0` degraded |
| `Y` | RUL | Target: remaining cycles to failure |

**DS03 degradation mode:** primary failure in HPT efficiency + flow and LPT efficiency + flow — typical of high-cycle turbine wear (tip clearance growth, blade erosion, seal wear).

---

## Models

Three RUL regression models, each trained on cycle-level features (mean / std / min / max of each sensor per cycle):

### GRU-Simple (Best)
Single-layer GRU with a dense head, trained on sliding windows of cycles.

| Hyperparameter | Value |
|---------------|-------|
| Window size | 30 cycles |
| Hidden units | 32 |
| Dense size | 16 |
| Dropout | 0.27 |
| Loss | Huber (delta = 0.35) |
| Optimiser | Adam lr = 0.004 |

### XGBoost-Rolling
Standard XGBoost augmented with rolling-window slope and delta features (window = 23 cycles) to capture temporal trends.

### XGBoost (baseline)
Single-cycle tabular XGBoost with Optuna hyperparameter search.

---

## Results (test set — engine unit 15)

| Model | RMSE | MAE | R² | NASA Score |
|-------|------|-----|----|-----------|
| XGBoost | 5.06 | 4.05 | 0.787 | 21.04 |
| XGBoost-Rolling | 3.35 | 2.53 | 0.907 | 11.59 |
| **GRU-Simple** | **2.63** | **2.21** | **0.943** | **7.60** |

Lower NASA Score = better. The score penalises late predictions more aggressively than early ones, reflecting the cost asymmetry of unplanned engine failures vs conservative early maintenance.

---

## Physics-Informed Anomaly Detection

Beyond statistical RUL prediction, the dashboard computes **isentropic efficiency residuals** from first-principles thermodynamics at every time step within each cycle:

| Component | Residual formula | Physical meaning |
|-----------|-----------------|-----------------|
| Fan | `T24/T2 - (P21/P2)^0.286` | Fan isentropic efficiency proxy |
| HPC | `T30/T24 - (P40/P24)^0.286` | Compressor efficiency proxy |
| HPT + LPT | `T50/T48 - (P50/P40)^0.286` | Turbine efficiency proxy |
| Combustor | `(T48 - T30) / Wf` | Specific fuel consumption |

Each residual is compared against a **per-step-index baseline** built from `hs=1` (healthy) cycles. This eliminates operating-condition variability and yields anomaly z-scores that are consistent across the full flight envelope.

---

## Digital Twin Dashboard

The primary deliverable is `reports/digital_twin.html` — a fully self-contained, zero-dependency browser dashboard for engine unit 15.

See [digital_twin.md](digital_twin.md) for the full user guide.

```bash
# 1. Generate telemetry JSON (requires trained models)
python generate_dt_data.py

# 2. Build the self-contained HTML
python build_dt_html.py

# 3. Open in any browser — no server needed
start reports/digital_twin.html   # Windows
open reports/digital_twin.html    # macOS
```

---

## Pipeline

```
N-CMAPSS_DS03-012.h5
        |
        v
    prepare                   src/prepare.py
    HDF5 -> cycle-level parquet (1 row/cycle)
    72 aggregate features + RUL cap
        |
        +------------------+------------------+
        v                  v                  v
  train_xgb         train_xgb_rolling   train_gru_simple
  XGBoost           XGBoost +           GRU window model
  baseline          rolling features    (BEST)
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                       evaluate              src/evaluate.py
                       metrics/evaluation.json
                       reports/plots/
                           |
                           v
                  generate_dt_data.py   ->   reports/dt_data.json
                           |
                           v
                  build_dt_html.py      ->   reports/digital_twin.html
```

### Run the pipeline

```bat
rem One-time setup
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

rem Full pipeline
.venv\Scripts\dvc.exe repro

rem View metrics
.venv\Scripts\dvc.exe metrics show

rem Build dashboard (after repro completes)
python generate_dt_data.py
python build_dt_html.py
```

---

## Project Structure

```
.
+-- src/
|   +-- prepare.py            HDF5 -> cycle-level parquet
|   +-- features.py           Rolling-window feature engineering
|   +-- models.py             GRU architecture + Dataset classes
|   +-- train_xgb.py          XGBoost baseline
|   +-- train_xgb_rolling.py  XGBoost with rolling features
|   +-- train_gru_simple.py   GRU-Simple RUL regression
|   +-- evaluate.py           Test evaluation + plots
|   +-- finetune.py           Optional fine-tuning stage
|   +-- mlflow_setup.py       MLflow / DagsHub auth
+-- data/
|   +-- raw/                  N-CMAPSS_DS03-012.h5 (DVC-tracked)
|   +-- processed/            all_cycles.parquet, metadata.json
+-- models/                   Saved weights, scalers, best_model.json
+-- metrics/                  Per-model JSON metrics
+-- reports/
|   +-- plots/                Evaluation figures (5 PNG files)
|   +-- dt_data.json          Unit-15 telemetry (generated)
|   +-- digital_twin.html     Interactive dashboard (generated)
+-- generate_dt_data.py       Extract unit-15 telemetry + GRU predictions
+-- build_dt_html.py          Compile self-contained dashboard HTML
+-- dvc.yaml                  Pipeline DAG
+-- params.yaml               All hyperparameters
+-- requirements.txt
```

---

## MLflow Tracking

All training runs are tracked on DagsHub:  
`https://dagshub.com/AmirhosseinnnKhademi/Physics_Informed_digital_twins_NASA_CHAMPS.mlflow`

Credentials go in `.env` (not committed):
```
MLflow_tracking_uri=https://dagshub.com/...
DAGSHUB_USER=AmirhosseinnnKhademi
DAGSHUB_TOKEN=<your_token>
```

---

## Environment

Python 3.9+ · PyTorch 2.6 + CUDA 12.4 · XGBoost 3.2 · Windows 11 (CPU-only also supported)
