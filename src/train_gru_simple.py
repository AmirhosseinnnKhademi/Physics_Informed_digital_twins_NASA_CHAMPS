"""Stage - train_gru_simple: lightweight GRU for RUL regression.

Key design choices vs the previous 1D-CNN+GRU:
  - Target normalised to [0,1]: RUL / rul_cap  (stable gradients)
  - Huber loss (robust to early-life windows with large RUL)
  - Tiny model: GRU(32) -> Dense(32) -> Dense(1)
  - Batch size 16 (better generalisation from 12 training engines)
  - Best-weights restore on early stopping
  - Validation by engine, not random windows
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
import mlflow

from models import SimpleGRUModel, RULWindowDataset
from mlflow_setup import setup_mlflow

load_dotenv()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def main():
    import pandas as pd
    params      = yaml.safe_load(open("params.yaml"))
    p           = params["gru_simple"]
    rul_cap     = params["prepare"]["rul_cap"]
    meta        = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    W         = p["window_size"]
    hidden    = p["hidden_size"]
    dense     = p["dense_size"]
    dropout   = p["dropout"]
    lr        = p["lr"]
    bs        = p["batch_size"]
    epochs    = p["epochs"]
    patience  = p["patience"]
    delta     = p["huber_delta"]
    grad_clip = p["grad_clip"]

    all_df   = pd.read_parquet("data/processed/all_cycles.parquet")
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]

    print(f"[train_gru_simple] device: {DEVICE}")
    print(f"[train_gru_simple] train: {len(train_df)} cycles  val: {len(val_df)} cycles")
    print(f"[train_gru_simple] window={W}  hidden={hidden}  bs={bs}  rul_cap={rul_cap}")

    scaler  = StandardScaler()
    scaled_train = train_df.copy()
    scaled_val   = val_df.copy()
    scaled_train[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    scaled_val[feature_cols]   = scaler.transform(val_df[feature_cols])

    train_ds = RULWindowDataset(scaled_train, feature_cols, W)
    val_ds   = RULWindowDataset(scaled_val,   feature_cols, W)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)

    print(f"[train_gru_simple] train windows: {len(train_ds)}  val windows: {len(val_ds)}")

    model     = SimpleGRUModel(len(feature_cols), hidden, dense, dropout).to(DEVICE)
    criterion = nn.HuberLoss(delta=delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse   = float("inf")
    best_state      = copy.deepcopy(model.state_dict())
    patience_ctr    = 0
    train_loss_hist = []
    val_rmse_hist   = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_norm = y / rul_cap
            pred   = model(X)
            loss   = criterion(pred, y_norm)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= len(train_ds)
        train_loss_hist.append(epoch_loss)

        model.eval()
        preds_v, true_v = [], []
        with torch.no_grad():
            for X, y in val_loader:
                p_norm = model(X.to(DEVICE)).cpu().numpy()
                preds_v.extend(p_norm * rul_cap)
                true_v.extend(y.numpy())
        val_rmse = float(np.sqrt(mean_squared_error(true_v, preds_v)))
        val_rmse_hist.append(val_rmse)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  loss={epoch_loss:.5f}  val_rmse={val_rmse:.3f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(model.state_dict())
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[train_gru_simple] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"[train_gru_simple] best val RMSE = {best_val_rmse:.3f}")

    model.eval()
    preds_t = []
    train_loader_full = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for X, y in train_loader_full:
            preds_t.extend(model(X.to(DEVICE)).cpu().numpy() * rul_cap)
    y_train_true = np.array([train_ds.y[i] for i in range(len(train_ds))])
    train_rmse   = float(np.sqrt(mean_squared_error(y_train_true, preds_t)))

    preds_v_arr = np.array(preds_v)
    true_v_arr  = np.array(true_v)
    val_mae     = float(mean_absolute_error(true_v_arr, preds_v_arr))
    val_r2      = float(r2_score(true_v_arr, preds_v_arr))
    val_nasa    = nasa_score(true_v_arr, preds_v_arr)

    print(f"[train_gru_simple] Val RMSE={best_val_rmse:.3f}  MAE={val_mae:.3f}  "
          f"R2={val_r2:.4f}  NASA={val_nasa:.1f}")

    metrics = {
        "val_rmse":       round(best_val_rmse, 4),
        "val_mae":        round(val_mae, 4),
        "val_r2":         round(val_r2, 4),
        "val_nasa_score": round(val_nasa, 2),
        "train_rmse":     round(train_rmse, 4),
        "best_epoch":     len(train_loss_hist),
        "window_size":    W,
        "rul_cap":        rul_cap,
    }

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/gru_simple_rul.pt")
    with open("models/gru_simple_scaler.pkl", "wb") as f:
        pickle.dump({
            "scaler":       scaler,
            "feature_cols": feature_cols,
            "window_size":  W,
            "rul_cap":      rul_cap,
        }, f)
    json.dump(metrics, open("metrics/gru_simple_rul.json", "w"), indent=2)

    history = {"train_loss": train_loss_hist, "val_rmse": val_rmse_hist}
    json.dump(history, open("metrics/gru_simple_history.json", "w"), indent=2)

    setup_mlflow("GRU-Simple-RUL")
    with mlflow.start_run(run_name="gru_simple_rul"):
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.log_param("device", str(DEVICE))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("models/gru_simple_rul.pt")

    print("[train_gru_simple] Done.")


if __name__ == "__main__":
    main()
