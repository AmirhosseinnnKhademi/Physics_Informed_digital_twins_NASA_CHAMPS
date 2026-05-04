"""Stage 4 - train_classifier: Health-state (hs) binary classifiers.

Engine-level split (roadmap-compliant):
  train  = engines 1-11  (scaler fitted here; both hs classes present)
  val    = engines 12-13 (both hs classes present — meaningful evaluation)
  test   = engines 14-15 (evaluated in the evaluate stage)
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)
from dotenv import load_dotenv
import mlflow

from models import LSTMClassifierModel, HealthWindowDataset
from mlflow_setup import setup_mlflow

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_classifier] Using device: {DEVICE}")


def eval_cls(y_true, y_proba, thr=0.5):
    y_pred = (y_proba >= thr).astype(int)
    labels = [0, 1]
    try:
        auc = round(float(roc_auc_score(y_true, y_proba)), 4)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",
                                            labels=labels, zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted",
                                            labels=labels, zero_division=0)), 4),
        "roc_auc":     auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_xgb_cls(X_tr, y_tr, X_val, y_val, p, device_str):
    pw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    m = xgb.XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        tree_method="hist", device=device_str, eval_metric="logloss",
        early_stopping_rounds=p["early_stopping_rounds"],
        scale_pos_weight=pw, random_state=42,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=30)
    return m


def train_lstm_cls(scaled_train_df, scaled_val_df, feature_cols, p):
    ws = p["window_size"]
    tr_ds  = HealthWindowDataset(scaled_train_df, feature_cols, ws)
    val_ds = HealthWindowDataset(scaled_val_df,   feature_cols, ws)

    tr_loader  = DataLoader(tr_ds,  batch_size=p["batch_size"], shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=p["batch_size"] * 2, shuffle=False, num_workers=0)

    model = LSTMClassifierModel(
        input_size=len(feature_cols), hidden_size=p["hidden_size"],
        num_layers=p["num_layers"], dropout=p["dropout"],
    ).to(DEVICE)

    criterion  = nn.BCELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=p["lr"])
    best_f1, patience_counter = 0.0, 0

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                preds.append(model(X.to(DEVICE)).cpu().numpy())
                targets.append(y.numpy())
        preds   = np.concatenate(preds)
        targets = np.concatenate(targets)
        f1 = f1_score((targets >= 0.5).astype(int),
                      (preds  >= 0.5).astype(int), average="macro", zero_division=0)
        print(f"  epoch {epoch:3d}/{p['epochs']} | val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1, patience_counter = f1, 0
            torch.save(model.state_dict(), "models/lstm_cls.pt")
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                print(f"[train_classifier] LSTM early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load("models/lstm_cls.pt", map_location=DEVICE))
    return model, preds, targets


def main():
    params = yaml.safe_load(open("params.yaml"))
    p      = params["classifier"]
    meta   = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    all_df   = pd.read_parquet("data/processed/all_cycles.parquet")
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]

    print(f"[train_classifier] train engines: {sorted(train_df['unit'].unique().astype(int))}")
    print(f"[train_classifier] val   engines: {sorted(val_df['unit'].unique().astype(int))}")
    print(f"[train_classifier] train={len(train_df)} val={len(val_df)}")
    print(f"  train hs: {train_df['hs'].value_counts().to_dict()}")
    print(f"  val   hs: {val_df['hs'].value_counts().to_dict()}")

    # Fit scaler on training engines ONLY
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    y_train = train_df["hs"].values.astype(int)
    y_val   = val_df["hs"].values.astype(int)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    print("\n[train_classifier] Training XGBoost classifier...")
    xgb_model = train_xgb_cls(X_train, y_train, X_val, y_val, p, device_str)
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_m     = eval_cls(y_val, xgb_proba)
    print(f"  XGB -> acc={xgb_m['accuracy']}  F1={xgb_m['f1_macro']}  "
          f"AUC={xgb_m['roc_auc']}")
    print(classification_report(y_val, (xgb_proba >= 0.5).astype(int),
                                labels=[0, 1], target_names=["degraded", "healthy"],
                                zero_division=0))
    xgb_model.save_model("models/xgb_cls.ubj")

    # Build scaled DataFrames for LSTM windowed training
    scaled_all = all_df.copy()
    scaled_all[feature_cols] = scaler.transform(all_df[feature_cols])
    scaled_train = scaled_all[scaled_all["unit"].isin(train_units)]
    scaled_val   = scaled_all[scaled_all["unit"].isin(val_units)]

    print("\n[train_classifier] Training LSTM classifier...")
    _, lstm_proba, lstm_true = train_lstm_cls(scaled_train, scaled_val, feature_cols, p)
    lstm_m = eval_cls((lstm_true >= 0.5).astype(int), lstm_proba)
    print(f"  LSTM -> acc={lstm_m['accuracy']}  F1={lstm_m['f1_macro']}  "
          f"AUC={lstm_m['roc_auc']}")

    metrics = {"xgb_classifier": xgb_m, "lstm_classifier": lstm_m}
    json.dump(metrics, open("metrics/classifier.json", "w"), indent=2)
    with open("models/cls_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)

    setup_mlflow("Health-Classifier")
    with mlflow.start_run(run_name="classifier"):
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.log_metrics({
            "xgb_accuracy":  xgb_m["accuracy"],  "xgb_f1_macro":  xgb_m["f1_macro"],
            "xgb_roc_auc":   xgb_m["roc_auc"],
            "lstm_accuracy": lstm_m["accuracy"],  "lstm_f1_macro": lstm_m["f1_macro"],
            "lstm_roc_auc":  lstm_m["roc_auc"],
        })

    print("[train_classifier] Done.")


if __name__ == "__main__":
    main()
