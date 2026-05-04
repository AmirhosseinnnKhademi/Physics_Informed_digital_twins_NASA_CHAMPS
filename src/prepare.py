"""Stage 1 - prepare: Loads N-CMAPSS HDF5 (dev + test groups), aggregates
high-frequency sensor rows to cycle level, caps RUL, and assigns an
engine-level train/val/test split.

Split (from roadmap):
  train  → engines 1-11   (model sees ALL cycles of these engines)
  val    → engines 12-13  (used for early-stopping / hyperparameter tuning)
  test   → engines 14-15  (held out; used ONCE for final honest evaluation)

This mirrors the roadmap recommendation: split by engine, never by row.
Engines 1-9 come from the HDF5 dev group; 10-15 from the HDF5 test group.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import time

PROCESSED = Path("data/processed")


def aggregate_to_cycles(A_arr, W_arr, Xs_arr, Y_arr, sensor_cols, op_cols, stats):
    feat_cols = op_cols + sensor_cols
    data = np.concatenate([A_arr, W_arr, Xs_arr, Y_arr], axis=1)
    col_names = ["unit", "cycle", "Fc", "hs"] + feat_cols + ["RUL"]
    df = pd.DataFrame(data, columns=col_names)

    agg_kwargs = {}
    for col in feat_cols:
        for stat in stats:
            agg_kwargs[f"{col}_{stat}"] = (col, stat)
    agg_kwargs["Fc"]  = ("Fc",  "first")
    agg_kwargs["hs"]  = ("hs",  "first")
    agg_kwargs["RUL"] = ("RUL", "first")

    cycle_df = df.groupby(["unit", "cycle"], sort=True).agg(**agg_kwargs).reset_index()
    std_cols = [f"{c}_std" for c in feat_cols]
    cycle_df[std_cols] = cycle_df[std_cols].fillna(0.0)
    return cycle_df


def assign_splits(cycle_df, train_units, val_units, test_units):
    """Label each cycle by its engine's role in the engine-level split."""
    unit_to_split = {}
    for u in train_units: unit_to_split[u] = "train"
    for u in val_units:   unit_to_split[u] = "val"
    for u in test_units:  unit_to_split[u] = "test"

    cycle_df = cycle_df.copy()
    cycle_df["split"] = cycle_df["unit"].map(unit_to_split).fillna("unknown")
    return cycle_df


def main():
    params      = yaml.safe_load(open("params.yaml"))["prepare"]
    stats       = params["stats"]
    train_units = params["train_units"]
    val_units   = params["val_units"]
    test_units  = params["test_units"]
    rul_cap     = params["rul_cap"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    with h5py.File("data/raw/N-CMAPSS_DS03-012.h5", "r") as f:
        sensor_cols = [x.decode() for x in f["X_s_var"][:]]
        op_cols     = [x.decode() for x in f["W_var"][:]]

        frames = []
        for split_name in ["dev", "test"]:
            print(f"\n[prepare] aggregating {split_name} group...")
            t0 = time.time()
            A  = f[f"A_{split_name}"][:]
            W  = f[f"W_{split_name}"][:]
            Xs = f[f"X_s_{split_name}"][:]
            Y  = f[f"Y_{split_name}"][:]
            df = aggregate_to_cycles(A, W, Xs, Y, sensor_cols, op_cols, stats)
            print(f"  {split_name}: {df.shape}  ({time.time()-t0:.1f}s)")
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n[prepare] combined: {all_df.shape}  "
          f"units: {sorted(all_df['unit'].unique().astype(int))}")

    # Cap RUL: prevents large early-life values from dominating the loss
    all_df["RUL"] = all_df["RUL"].clip(upper=rul_cap)
    print(f"[prepare] RUL capped at {rul_cap}  "
          f"(max after cap: {all_df['RUL'].max():.0f})")

    # Engine-level split assignment
    all_df = assign_splits(all_df, train_units, val_units, test_units)
    counts = all_df["split"].value_counts()
    print(f"[prepare] split counts: "
          f"train={counts.get('train',0)} "
          f"val={counts.get('val',0)} "
          f"test={counts.get('test',0)}")
    print(f"  train engines: {sorted(all_df[all_df['split']=='train']['unit'].unique().astype(int))}")
    print(f"  val   engines: {sorted(all_df[all_df['split']=='val'  ]['unit'].unique().astype(int))}")
    print(f"  test  engines: {sorted(all_df[all_df['split']=='test' ]['unit'].unique().astype(int))}")

    out = PROCESSED / "all_cycles.parquet"
    all_df.to_parquet(out, index=False)
    print(f"[prepare] saved -> {out}")

    # Feature columns: cycle + aggregated sensor/op stats
    stat_cols    = [f"{c}_{s}" for c in (op_cols + sensor_cols) for s in stats]
    feature_cols = ["cycle"] + stat_cols   # cycle gives positional context to XGBoost

    meta = {
        "feature_cols":  feature_cols,
        "stat_cols":     stat_cols,
        "sensor_cols":   sensor_cols,
        "op_cols":       op_cols,
        "stats":         stats,
        "n_features":    len(feature_cols),
        "train_units":   train_units,
        "val_units":     val_units,
        "test_units":    test_units,
        "rul_cap":       rul_cap,
    }
    json.dump(meta, open(PROCESSED / "metadata.json", "w"), indent=2)
    print(f"[prepare] metadata saved: {len(feature_cols)} features")
    print("[prepare] Done.")


if __name__ == "__main__":
    main()
