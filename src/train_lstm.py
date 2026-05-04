"""Stage 2 - train_gru: Trains a 1D-CNN + GRU model on the training engines
(1-11) to predict Remaining Useful Life.

Engine-level split (roadmap-compliant):
  train  = engines 1-11  — model learns from ALL cycles of these engines
  val    = engines 12-13 — used for early stopping only, no weight updates
  test   = engines 14-15 — held out, evaluated in the evaluate stage

Scaler is fitted on training engine cycles only, then applied to val/test.
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
import yaml
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import mlflow

from models import CNNGRUModel, RULWindowDataset
from mlflow_setup import setup_mlflow

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_gru] Using device: {DEVICE}")


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += criterion(out, y).item() * len(y)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    return total_loss / len(targets), rmse, mae, preds, targets


def main():
    params = yaml.safe_load(open("params.yaml"))
    p      = params["gru"]
    meta   = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    all_df   = pd.read_parquet("data/processed/all_cycles.parquet")
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]

    print(f"[train_gru] train engines: {sorted(train_df['unit'].unique().astype(int))}")
    print(f"[train_gru] val   engines: {sorted(val_df['unit'].unique().astype(int))}")
    print(f"[train_gru] train cycles: {len(train_df)}  val cycles: {len(val_df)}")
    print(f"[train_gru] features: {len(feature_cols)}  window_size: {p['window_size']}")

    # Fit scaler on training engines ONLY
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    # Scale all engines with the same scaler
    scaled_all = all_df.copy()
    scaled_all[feature_cols] = scaler.transform(all_df[feature_cols])

    scaled_train = scaled_all[scaled_all["unit"].isin(train_units)]
    scaled_val   = scaled_all[scaled_all["unit"].isin(val_units)]

    ws = p["window_size"]
    train_ds = RULWindowDataset(scaled_train, feature_cols, ws)
    val_ds   = RULWindowDataset(scaled_val,   feature_cols, ws)
    print(f"[train_gru] train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=p["batch_size"] * 2,
                              shuffle=False, num_workers=0, pin_memory=pin)

    model = CNNGRUModel(
        input_size   = len(feature_cols),
        cnn_channels = p["cnn_channels"],
        hidden_size  = p["hidden_size"],
        num_layers   = p["num_layers"],
        dropout      = p["dropout"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    setup_mlflow("GRU-RUL")

    best_val_rmse    = float("inf")
    patience_counter = 0
    history          = {"train_loss": [], "val_rmse": [], "val_mae": []}

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    with mlflow.start_run(run_name="gru_rul"):
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("device",     str(DEVICE))

        for epoch in range(1, p["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out  = model(X)
                loss = criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
                optimizer.step()
                epoch_loss += loss.item() * len(y)
            epoch_loss /= len(train_ds)

            _, val_rmse, val_mae, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_rmse)

            history["train_loss"].append(epoch_loss)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_mae)

            mlflow.log_metrics(
                {"train_loss": epoch_loss, "val_rmse": val_rmse, "val_mae": val_mae},
                step=epoch,
            )
            print(f"  epoch {epoch:3d}/{p['epochs']} | "
                  f"train_loss={epoch_loss:.4f} | "
                  f"val_rmse={val_rmse:.3f} | val_mae={val_mae:.3f}")

            if val_rmse < best_val_rmse:
                best_val_rmse    = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), "models/gru_rul.pt")
            else:
                patience_counter += 1
                if patience_counter >= p["patience"]:
                    print(f"[train_gru] Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(
            torch.load("models/gru_rul.pt", map_location=DEVICE))
        _, val_rmse, val_mae, val_preds, val_true = evaluate(
            model, val_loader, criterion)
        val_r2   = float(1 - np.sum((val_preds - val_true) ** 2) /
                         np.sum((val_true - val_true.mean()) ** 2))
        val_nasa = nasa_score(val_true, val_preds)

        metrics = {
            "val_rmse":       round(val_rmse, 4),
            "val_mae":        round(val_mae, 4),
            "val_r2":         round(val_r2, 4),
            "val_nasa_score": round(val_nasa, 2),
            "best_epoch":     epoch - patience_counter,
        }

        with open("models/gru_scaler.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "feature_cols": feature_cols,
                         "window_size": ws}, f)
        json.dump(history, open("metrics/gru_history.json", "w"), indent=2)
        json.dump(metrics, open("metrics/gru_rul.json",     "w"), indent=2)

        mlflow.log_metrics({
            "best_val_rmse":       val_rmse,
            "best_val_mae":        val_mae,
            "best_val_r2":         val_r2,
            "best_val_nasa_score": val_nasa,
        })
        mlflow.log_artifact("models/gru_rul.pt")

    print(f"\n[train_gru] Best val RMSE: {val_rmse:.3f}  "
          f"R2: {val_r2:.4f}  NASA score: {val_nasa:.1f}")
    print("[train_gru] Done.")


if __name__ == "__main__":
    main()
