"""Stage 3 - train_xgb: XGBoost RUL regression on cycle-level features.

Engine-level split (roadmap-compliant):
  train  = engines 1-11  — all cycles used for training
  val    = engines 12-13 — used for early stopping only

Scaler fitted on training engines only; same scaler applied to val/test.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import pickle
from pathlib import Path
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
from mlflow_setup import setup_mlflow
import mlflow

load_dotenv()


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def main():
    params = yaml.safe_load(open("params.yaml"))
    p      = params["xgb"]
    meta   = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    all_df   = pd.read_parquet("data/processed/all_cycles.parquet")
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]

    print(f"[train_xgb] train engines: {sorted(train_df['unit'].unique().astype(int))}")
    print(f"[train_xgb] val   engines: {sorted(val_df['unit'].unique().astype(int))}")
    print(f"[train_xgb] train: {len(train_df)} cycles  val: {len(val_df)} cycles")
    print(f"[train_xgb] features: {len(feature_cols)}")

    # Fit scaler on training engines ONLY
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    y_train = train_df["RUL"].values
    y_val   = val_df["RUL"].values

    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_xgb] XGBoost device: {device_str}")

    model = xgb.XGBRegressor(
        n_estimators      = p["n_estimators"],
        max_depth         = p["max_depth"],
        learning_rate     = p["learning_rate"],
        subsample         = p["subsample"],
        colsample_bytree  = p["colsample_bytree"],
        min_child_weight  = p["min_child_weight"],
        gamma             = p["gamma"],
        reg_alpha         = p["reg_alpha"],
        reg_lambda        = p["reg_lambda"],
        tree_method       = "hist",
        device            = device_str,
        eval_metric       = "rmse",
        early_stopping_rounds = p["early_stopping_rounds"],
        verbosity         = 1,
        random_state      = 42,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=50)

    val_preds  = model.predict(X_val)
    val_rmse   = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mae    = float(mean_absolute_error(y_val, val_preds))
    val_r2     = float(r2_score(y_val, val_preds))
    val_nasa   = nasa_score(y_val, val_preds)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))

    print(f"\n[train_xgb] Val RMSE={val_rmse:.3f}  MAE={val_mae:.3f}  "
          f"R2={val_r2:.4f}  NASA={val_nasa:.1f}")

    metrics = {
        "val_rmse":       round(val_rmse, 4),
        "val_mae":        round(val_mae, 4),
        "val_r2":         round(val_r2, 4),
        "val_nasa_score": round(val_nasa, 2),
        "train_rmse":     round(train_rmse, 4),
        "best_iteration": int(model.best_iteration),
    }

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    model.save_model("models/xgb_rul.ubj")
    with open("models/xgb_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    json.dump(metrics, open("metrics/xgb_rul.json", "w"), indent=2)

    setup_mlflow("XGBoost-RUL")
    with mlflow.start_run(run_name="xgb_rul"):
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.log_param("device", device_str)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("models/xgb_rul.ubj")

    print("[train_xgb] Done.")


if __name__ == "__main__":
    main()
