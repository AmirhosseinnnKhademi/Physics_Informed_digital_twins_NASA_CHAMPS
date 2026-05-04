# Digital Twin Dashboard — User Guide

`reports/digital_twin.html` is a self-contained interactive dashboard for engine unit 15 of the NASA N-CMAPSS DS03-012 dataset. It requires no server, no internet connection, and no installed dependencies — open it in any modern browser.

---

## Quick Start

```bash
# Generate telemetry data (one-time, requires trained models)
python generate_dt_data.py

# Build the dashboard HTML
python build_dt_html.py

# Open in browser
start reports/digital_twin.html
```

The two scripts take about 30 seconds combined. The output is a single ~5.5 MB HTML file with all data and logic embedded.

---

## Layout Overview

```
+-------------------------------+-------------------------------+
|  Top bar: cycle selector, playback controls, status strip      |
+-------------------------------+-------------------------------+
|  Tab: DIGITAL TWIN  |  Tab: PHYSICS ANALYSIS                  |
+-------------------------------+-------------------------------+
```

The two tabs are independent views of the same cycle/step.

---

## Top Bar

| Element | Description |
|---------|-------------|
| **Cycle N / 67** | Current cycle number and total. Click Prev / Next to step between cycles. |
| **Jump to cycle** | Dropdown on the far right — jump to any specific cycle directly. |
| **Actual RUL / Predicted RUL** | Small header panel showing ground-truth RUL and GRU-Simple model prediction for the current cycle. |
| **Engine Health Stage gauges** | Three bars (HPT eff / LPT eff / LPT flow) calibrated to this engine's observed degradation range. Green = healthy end, red = worst observed. |
| **HEALTHY / DEGRADING / WARNING / CRITICAL** | Status strip derived from the dataset's `hs` label and the sensor anomaly level at the current step. |
| **PLAY / >> / <<** | Animate through steps within the current cycle. Speed selector: 1× 2× 5× 10× 20×. |
| **Step N/100** | Current normalized time step within the cycle. Each cycle is resampled to 100 evenly-spaced steps. |
| **Alt / Mach / Phase** | Live operating condition at the current step. Phase (GROUND / CLIMB / CRUISE / DESCENT) is detected from altitude and step position. |

---

## Digital Twin Tab

### Engine Schematic

An annotated cross-section of the two-spool turbofan shows:

- **Sensor labels** (T24, T30, T48, T50, P15, P24, Ps30, P40, P50) with **dynamic color** — red = above the healthy baseline at this step, blue = below. Neutral gray = within normal range. This uses the same z-score scale as the Physics Analysis tab.
- **Section heat map** — the fill color of each engine section (LPC, HPC, combustor, HPT, LPT) reflects temperature from the corresponding sensor.
- **Combustor glow** — intensity tracks the Specific Fuel Consumption residual.
- **Component health labels** — HPT and LPT sections show their health percentage (100% = new, decreasing as degradation accumulates).
- **Flow particles** — animated dots illustrate airflow through the core.

### Flight Altitude Profile

A canvas below the schematic shows the altitude trace for the full current cycle (0–100 steps), with a moving aircraft icon at the current step. Color transitions from green (low altitude) through cyan (climb) to yellow (cruise) to amber (descent).

### Sensor Cards (bottom strip)

Seven-column grid showing all 14 sensors split across two rows. Each card shows:
- **Current value** with engineering units
- **Colored bar** — red if the within-cycle z-deviation exceeds ~40%, amber if moderate, cyan if normal

### Right Panel: Instruments

- **N1 Fan Speed** gauge and RPM value
- **N2 Core Speed** gauge and RPM value
- **EGT (T50)** bar — exhaust gas temperature
- **Fuel Flow (Wf)** bar
- **Component Health** — per-component health percentages for all 10 degradation parameters from the dataset's ground-truth `T` array

---

## Physics Analysis Tab

Accessed via the **PHYSICS ANALYSIS** tab button. All charts update whenever you change cycle or step.

### Early Warning Delta Panel

Four cards (Fan / HPC / HPT+LPT / Combustor) showing the **cycle at which each component's cycle-mean residual first crossed the detection threshold**:

| State | Display | Meaning |
|-------|---------|---------|
| Monitoring | `sigma = X.XX` | Current z-score, no crossing yet |
| ALARM | Pulsing red border | Threshold crossed at this exact cycle |
| After alarm | `Cyc N | K cyc ago` with red border | Warning was triggered K cycles ago |

Detection threshold: 2σ above the hs=1 cycle-mean baseline. HPT+LPT uses a sustained 2-cycle crossing to reduce false positives. A positive delta ("+N cyc vs hs=0") means the physics model detected degradation **before** the dataset's own health label changed.

### Component Residual Cards (4 panels)

One panel per component. Each shows:

- **Residual formula** and physical station (e.g., `r = T50/T48 − (P50/P40)^0.286`)
- **Current residual value** (large number)
- **Anomaly badge**: NOMINAL / WATCH / ABNORMAL / CRITICAL based on drift from per-step baseline:
  - < 2σ → NOMINAL (green)
  - 2–3σ → WATCH (orange)
  - 3–4σ → ABNORMAL (orange-red)
  - ≥ 4σ → CRITICAL (red)
- **Step-level residual chart** — the orange line traces the residual across all 100 steps of the current cycle. The green band is the hs=1 healthy baseline ±1σ at each step index. The white dot marks the current step.

### Cycle Trend Charts (4 panels)

One trend per component, showing the **cycle-mean residual across all 67 cycles**:

- Orange line = this engine's cycle-mean residual history
- Green band = hs=1 healthy baseline ±1σ (cycle-level)
- Dashed red band = 2σ warning zone
- Background shading: green for hs=1 cycles, red for hs=0 cycles
- Vertical dashed line = current cycle position
- Red label at top-right = current cycle number

### Flight Phase Breakdown (4 panels)

Each panel shows the component residual broken down by flight phase across all cycles:

| Color | Phase |
|-------|-------|
| Dark gray | Ground (alt < 1,000 ft) |
| Blue | Climb (rising to cruise) |
| Green | Cruise (alt ≥ 39,000 ft) |
| Amber | Descent |

Separating by phase removes the effect of different operating conditions on the residual, making degradation trends more visible within each regime.

### Health State & RUL Timeline

Full-width chart at the bottom of the Physics tab:

- **Cyan line** — actual RUL (ground truth), decreasing from ~60 to 0
- **Dashed orange line** — GRU-Simple model prediction
- **Background** — green for hs=1 healthy cycles, red for hs=0 degraded cycles
- **Vertical dashed line** — current cycle position
- **Dot on actual RUL line** — colored green (healthy) or red (degraded) per the dataset label

---

## Understanding the Physics

### Why isentropic residuals?

Raw sensor values reflect both the current operating condition (altitude, throttle) and engine health. Two engines — one healthy, one degraded — flying at different altitudes will show different sensor readings even if the degradation level is identical.

Isentropic efficiency ratios (temperature ratio divided by pressure ratio raised to (gamma-1)/gamma) cancel out the operating point and expose only the thermodynamic efficiency. A rising HPT+LPT residual means the turbine is converting less of the available enthalpy drop into shaft work — the signature of tip clearance growth and blade erosion.

### Why per-step-index baselines?

Even within a single flight, sensors sweep through a wide range (takeoff → climb → cruise → descent). Comparing step 10 (climb) against step 80 (descent) as if they are equivalent would produce false anomalies. The dashboard builds a separate mean and standard deviation for each of the 100 step positions using only the `hs=1` healthy cycles, then z-scores each step against its own position-matched distribution.

### hs=1 as the reference

The dataset labels cycles 1–24 as `hs=1` (healthy) and cycles 25–67 as `hs=0` (degraded). All baselines in this dashboard — for the residual charts, the cycle trends, the early warning delta, and the sensor label colors — are built exclusively from cycles 1–24. This means the dashboard is asking the question: **"how different is this cycle from when the engine was known to be healthy?"**

---

## Regenerating the Dashboard

If you retrain the models, run:

```bash
python generate_dt_data.py   # re-extracts unit-15 telemetry + new GRU predictions
python build_dt_html.py      # rebuilds the HTML with embedded data
```

To change which test unit is visualised, edit `TEST_UNIT = 15` at the top of `generate_dt_data.py`.

---

## File Reference

| File | Role |
|------|------|
| `generate_dt_data.py` | Extracts unit-15 data from HDF5, runs GRU inference, writes `reports/dt_data.json` |
| `build_dt_html.py` | Reads `dt_data.json`, generates `reports/digital_twin.html` with all JS/CSS/data inline |
| `reports/dt_data.json` | Intermediate telemetry file: 67 cycles × 100 steps × 14 sensors + GRU predictions |
| `reports/digital_twin.html` | Final deliverable — open in any browser |
