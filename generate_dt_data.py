"""Extract two-level telemetry: 67 cycles x 10 normalized time-steps per cycle."""
import sys, os, json, pickle, pathlib
sys.path.insert(0, "src")

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from models import SimpleGRUModel, RULWindowDataset

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS   = 100  # normalized time-steps per cycle
TEST_UNIT = 15

# sensor / column names matching HDF5 order
SENSOR_NAMES = ["T24","T30","T48","T50","P15","P2","P21","P24","Ps30","P40","P50","Nf","Nc","Wf"]
OP_NAMES     = ["alt","Mach","TRA","T2"]
DEGRAD_NAMES = ["fan_eff","fan_flow","LPC_eff","LPC_flow",
                "HPC_eff","HPC_flow","HPT_eff","HPT_flow","LPT_eff","LPT_flow"]

def to_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x

def uniform_indices(n, k=N_STEPS):
    """Pick k evenly-spaced row indices from 0..n-1."""
    return np.linspace(0, n - 1, k, dtype=int)

def main():
    print("Loading HDF5 raw data...")
    with h5py.File("data/raw/N-CMAPSS_DS03-012.h5", "r") as f:
        A_all  = f["A_test"][:]
        W_all  = f["W_test"][:]
        Xs_all = f["X_s_test"][:]
        T_all  = f["T_test"][:]
        Y_all  = f["Y_test"][:]

    # Filter to test unit
    mask = A_all[:, 0] == TEST_UNIT
    A  = A_all[mask]
    W  = W_all[mask]
    Xs = Xs_all[mask]
    T  = T_all[mask]
    Y  = Y_all[mask]

    cycles_nums = sorted(np.unique(A[:, 1]).astype(int))
    print(f"Unit {TEST_UNIT}: {len(cycles_nums)} cycles, {mask.sum()} total samples")

    # Global sensor ranges (across ALL test units) for colour mapping
    sensor_ranges = {}
    for i, s in enumerate(SENSOR_NAMES):
        sensor_ranges[s] = {
            "min": float(Xs_all[:, i].min()),
            "max": float(Xs_all[:, i].max()),
        }
    op_ranges = {}
    for i, s in enumerate(OP_NAMES):
        op_ranges[s] = {
            "min": float(W_all[:, i].min()),
            "max": float(W_all[:, i].max()),
        }
    # Degradation global min (worst observed) per component
    degrad_min = {DEGRAD_NAMES[i]: float(T_all[:, i].min()) for i in range(len(DEGRAD_NAMES))}

    # GRU-Simple predictions per cycle (cycle-level)
    print("Loading GRU-Simple predictions...")
    with open("models/gru_simple_scaler.pkl", "rb") as f:
        gru_pkg = pickle.load(f)

    import pandas as pd
    all_df   = pd.read_parquet("data/processed/all_cycles.parquet")
    test_df  = all_df[all_df["unit"] == TEST_UNIT].sort_values("cycle").reset_index(drop=True)
    gru_fcols = gru_pkg["feature_cols"]
    W_gru     = gru_pkg["window_size"]
    rul_cap   = gru_pkg["rul_cap"]

    gru_model = SimpleGRUModel(input_size=len(gru_fcols), hidden_size=32, dense_size=32, dropout=0.2)
    state = torch.load("models/gru_simple_rul.pt", map_location=DEVICE)
    gru_model.load_state_dict(state)
    gru_model.to(DEVICE).eval()

    scaled = test_df.copy()
    scaled[gru_fcols] = gru_pkg["scaler"].transform(test_df[gru_fcols])
    ds     = RULWindowDataset(scaled, gru_fcols, W_gru)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    preds  = []
    with torch.no_grad():
        for Xb, _ in loader:
            p = gru_model(Xb.to(DEVICE)).cpu().numpy()
            preds.extend(np.clip(p * rul_cap, 0, None))
    pred_by_cycle = {}
    for i in range(W_gru, len(test_df) + 1):
        cyc = int(test_df["cycle"].iloc[i - 1])
        pred_by_cycle[cyc] = float(preds[i - W_gru])

    # Build per-cycle, per-step data
    print("Building two-level telemetry...")
    cycles_data = []
    for cyc_num in cycles_nums:
        cmask = A[:, 1] == cyc_num
        A_c  = A[cmask]
        W_c  = W[cmask]
        Xs_c = Xs[cmask]
        T_c  = T[cmask]
        Y_c  = Y[cmask]
        n    = cmask.sum()

        rul    = int(Y_c[0, 0])
        hs     = int(A_c[0, 3])
        T_cyc  = T_c[0]   # degradation is constant within a cycle

        # Component health: map degradation to 0-100% (100=new, 0=fully degraded)
        comp_health = {}
        for i, name in enumerate(DEGRAD_NAMES):
            d_min = degrad_min[name]
            if d_min < 0:
                # health = 1 - |current degradation| / |max degradation|
                comp_health[name] = float(max(0.0, min(100.0,
                    (1.0 - abs(T_cyc[i]) / abs(d_min)) * 100.0
                )))
            else:
                comp_health[name] = 100.0

        # Uniformly sample N_STEPS from within-cycle time series
        idx = uniform_indices(n, N_STEPS)
        steps = []
        for step_i, raw_i in enumerate(idx):
            row = {}
            row["step"] = step_i
            row["raw_t"] = int(raw_i)   # original row index within cycle
            # operating conditions
            for j, s in enumerate(OP_NAMES):
                row[s] = float(W_c[raw_i, j])
            # sensors
            for j, s in enumerate(SENSOR_NAMES):
                row[s] = float(Xs_c[raw_i, j])
            # deviation from cycle's own mean (shows transient behaviour)
            for j, s in enumerate(SENSOR_NAMES):
                cyc_mean = float(Xs_c[:, j].mean())
                cyc_std  = float(Xs_c[:, j].std()) + 1e-9
                row[s + "_z"] = float((Xs_c[raw_i, j] - cyc_mean) / cyc_std)
            steps.append(row)

        cycles_data.append({
            "cycle":         cyc_num,
            "rul":           rul,
            "rul_pred":      pred_by_cycle.get(cyc_num),
            "hs":            hs,
            "n_raw_samples": int(n),
            "comp_health":   comp_health,
            "steps":         steps,
        })

    output = {
        "meta": {
            "test_unit":     TEST_UNIT,
            "n_cycles":      len(cycles_data),
            "n_steps":       N_STEPS,
            "rul_cap":       int(rul_cap),
            "sensor_names":  SENSOR_NAMES,
            "op_names":      OP_NAMES,
            "degrad_names":  DEGRAD_NAMES,
        },
        "sensor_ranges": sensor_ranges,
        "op_ranges":     op_ranges,
        "degrad_min":    degrad_min,
        "cycles":        cycles_data,
    }

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return super().default(o)

    out = pathlib.Path("reports/dt_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fout:
        json.dump(output, fout, cls=NE)

    sz = out.stat().st_size // 1024
    print(f"Written {out} ({sz} KB) — {len(cycles_data)} cycles x {N_STEPS} steps")


if __name__ == "__main__":
    main()
