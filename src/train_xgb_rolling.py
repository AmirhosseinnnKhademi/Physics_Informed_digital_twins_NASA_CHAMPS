"""Stage - train_xgb_rolling: XGBoost RUL regression with rolling temporal features.

Extends the base XGBoost model by adding per-engine rolling statistics over the
last W cycles for each sensor:
  {col}_slope : linear trend (is this sensor degrading fast or slow?)
  {col}_delta : value change over W cycles (how much has it shifted?)

These give XGBoost explicit temporal context without requiring a sequence model.
Engine-level split — same as train_xgb.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
from mlflow_setup import setup_mlflow
import mlflow

from features import add_rolling_features

load_dotenv()


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def main():
    params = yaml.safe_load(open("params.yaml"))
    p      = params["xgb_rolling"]
    meta   = json.load(open("data/processed/metadata.json"))
    stat_cols    = meta["stat_cols"]
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    W = p["rolling_window"]

    all_df = pd.read_parquet("data/processed/all_cycles.parquet")

    print(f"[train_xgb_rolling] Computing rolling features (window={W})...")
    enriched = add_rolling_features(all_df, stat_cols, W)

    rolling_cols     = [f"{c}_slope" for c in stat_cols] + [f"{c}_delta" for c in stat_cols]
    all_feature_cols = feature_cols + rolling_cols

    train_df = enriched[enriched["unit"].isin(train_units)]
    val_df   = enriched[enriched["unit"].isin(val_units)]

    print(f"[train_xgb_rolling] train: {len(train_df)} cycles  val: {len(val_df)} cycles")
    print(f"[train_xgb_rolling] features: {len(all_feature_cols)} "
          f"(base {len(feature_cols)} + rolling {len(rolling_cols)})")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_feature_cols])
    X_val   = scaler.transform(val_df[all_feature_cols])
    y_train = train_df["RUL"].values
    y_val   = val_df["RUL"].values

    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_xgb_rolling] device: {device_str}")

    model = xgb.XGBRegressor(
        n_estimators          = p["n_estimators"],
        max_depth             = p["max_depth"],
        learning_rate         = p["learning_rate"],
        subsample             = p["subsample"],
        colsample_bytree      = p["colsample_bytree"],
        min_child_weight      = p["min_child_weight"],
        gamma                 = p["gamma"],
        reg_alpha             = p["reg_alpha"],
        reg_lambda            = p["reg_lambda"],
        tree_method           = "hist",
        device                = device_str,
        eval_metric           = "rmse",
        early_stopping_rounds = p["early_stopping_rounds"],
        verbosity             = 1,
        random_state          = 42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_preds  = model.predict(X_val)
    val_rmse   = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mae    = float(mean_absolute_error(y_val, val_preds))
    val_r2     = float(r2_score(y_val, val_preds))
    val_nasa   = nasa_score(y_val, val_preds)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))

    print(f"\n[train_xgb_rolling] Val RMSE={val_rmse:.3f}  MAE={val_mae:.3f}  "
          f"R2={val_r2:.4f}  NASA={val_nasa:.1f}")

    metrics = {
        "val_rmse":       round(val_rmse, 4),
        "val_mae":        round(val_mae, 4),
        "val_r2":         round(val_r2, 4),
        "val_nasa_score": round(val_nasa, 2),
        "train_rmse":     round(train_rmse, 4),
        "best_iteration": int(model.best_iteration),
        "n_features":     len(all_feature_cols),
        "rolling_window": W,
    }

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    model.save_model("models/xgb_rolling_rul.ubj")
    with open("models/xgb_rolling_scaler.pkl", "wb") as f:
        pickle.dump({
            "scaler":          scaler,
            "feature_cols":    feature_cols,
            "stat_cols":       stat_cols,
            "all_feature_cols": all_feature_cols,
            "rolling_window":  W,
        }, f)
    json.dump(metrics, open("metrics/xgb_rolling_rul.json", "w"), indent=2)

    setup_mlflow("XGBoost-Rolling-RUL")
    with mlflow.start_run(run_name="xgb_rolling_rul"):
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.log_param("device", device_str)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("models/xgb_rolling_rul.ubj")

    print("[train_xgb_rolling] Done.")


if __name__ == "__main__":
    main()
