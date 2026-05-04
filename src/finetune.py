"""Stage - finetune: Optuna hyperparameter search for all three models.

Runs independent Optuna studies for:
  1. XGBoost (base)           — 50 trials, GPU via tree_method=hist
  2. XGBoost-Rolling          — 30 trials, GPU
  3. GRU-Simple               — 30 trials, GPU, MedianPruner kills weak trials at epoch 20

After each study the winning config is retrained to produce the final model.
models/best_model.json records the overall winner for evaluate.py to highlight.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import copy, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import mlflow

from models import SimpleGRUModel, RULWindowDataset
from features import add_rolling_features
from mlflow_setup import setup_mlflow

load_dotenv()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── XGBoost (base) ────────────────────────────────────────────────────────────

def _xgb_trial(trial, X_tr, y_tr, X_val, y_val, device_str):
    p = dict(
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
        n_estimators              = 3000,
        early_stopping_rounds     = 50,
        tree_method               = "hist",
        device                    = device_str,
        eval_metric               = "rmse",
        verbosity                 = 0,
        random_state              = 42,
    )
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))


def run_xgb_study(all_df, train_units, val_units, feature_cols, n_trials, device_str):
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]
    scaler   = StandardScaler()
    X_tr  = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    y_tr  = train_df["RUL"].values
    y_val = val_df["RUL"].values

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.NopPruner(),
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _xgb_trial(t, X_tr, y_tr, X_val, y_val, device_str),
        n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  XGBoost  best val RMSE={study.best_value:.4f}  params={best}")

    # Retrain with best params
    p = {**best,
         "n_estimators": 3000, "early_stopping_rounds": 50,
         "tree_method": "hist", "device": device_str,
         "eval_metric": "rmse", "verbosity": 1, "random_state": 42}
    ft_scaler = StandardScaler()
    X_tr2  = ft_scaler.fit_transform(train_df[feature_cols])
    X_val2 = ft_scaler.transform(val_df[feature_cols])
    ft_model = xgb.XGBRegressor(**p)
    ft_model.fit(X_tr2, y_tr, eval_set=[(X_val2, y_val)], verbose=100)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, ft_model.predict(X_val2))))
    scaler_pkg = {"scaler": ft_scaler, "feature_cols": feature_cols}
    return ft_model, scaler_pkg, val_rmse, best


# ── XGBoost-Rolling ───────────────────────────────────────────────────────────

def _xgb_rolling_trial(trial, all_df, train_units, val_units,
                       stat_cols, feature_cols, device_str):
    W   = trial.suggest_int("rolling_window", 3, 25)
    p   = dict(
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
        n_estimators              = 3000,
        early_stopping_rounds     = 50,
        tree_method               = "hist",
        device                    = device_str,
        eval_metric               = "rmse",
        verbosity                 = 0,
        random_state              = 42,
    )
    enriched = add_rolling_features(all_df, stat_cols, W)
    rcols    = [f"{c}_slope" for c in stat_cols] + [f"{c}_delta" for c in stat_cols]
    all_fc   = feature_cols + rcols

    tr  = enriched[enriched["unit"].isin(train_units)]
    val = enriched[enriched["unit"].isin(val_units)]
    sc  = StandardScaler()
    X_tr  = sc.fit_transform(tr[all_fc]);  y_tr  = tr["RUL"].values
    X_val = sc.transform(val[all_fc]);     y_val = val["RUL"].values

    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))


def run_xgb_rolling_study(all_df, train_units, val_units, stat_cols,
                           feature_cols, n_trials, device_str):
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.NopPruner(),
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _xgb_rolling_trial(t, all_df, train_units, val_units,
                                     stat_cols, feature_cols, device_str),
        n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    W    = best.pop("rolling_window")
    print(f"  XGBoost-Rolling  best val RMSE={study.best_value:.4f}  W={W}  params={best}")

    # Retrain
    enriched = add_rolling_features(all_df, stat_cols, W)
    rcols    = [f"{c}_slope" for c in stat_cols] + [f"{c}_delta" for c in stat_cols]
    all_fc   = feature_cols + rcols
    tr  = enriched[enriched["unit"].isin(train_units)]
    val = enriched[enriched["unit"].isin(val_units)]
    sc  = StandardScaler()
    X_tr  = sc.fit_transform(tr[all_fc]);  y_tr  = tr["RUL"].values
    X_val = sc.transform(val[all_fc]);     y_val = val["RUL"].values

    p = {**best,
         "n_estimators": 3000, "early_stopping_rounds": 50,
         "tree_method": "hist", "device": device_str,
         "eval_metric": "rmse", "verbosity": 1, "random_state": 42}
    ft_model = xgb.XGBRegressor(**p)
    ft_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, ft_model.predict(X_val))))
    scaler_pkg = {
        "scaler": sc, "feature_cols": feature_cols,
        "stat_cols": stat_cols, "all_feature_cols": all_fc,
        "rolling_window": W,
    }
    return ft_model, scaler_pkg, val_rmse, {**best, "rolling_window": W}


# ── GRU-Simple ────────────────────────────────────────────────────────────────

def _gru_trial(trial, train_df, val_df, feature_cols, rul_cap):
    window  = trial.suggest_categorical("window_size", [10, 15, 20, 30])
    hidden  = trial.suggest_categorical("hidden_size", [16, 32, 64])
    dense   = trial.suggest_categorical("dense_size",  [16, 32, 64])
    dropout = trial.suggest_float("dropout",  0.1, 0.5)
    lr      = trial.suggest_float("lr",       5e-4, 5e-3, log=True)
    batch   = trial.suggest_categorical("batch_size", [8, 16, 32])
    delta   = trial.suggest_float("huber_delta", 0.1, 1.0)

    sc = StandardScaler()
    sc_tr  = train_df.copy(); sc_tr[feature_cols]  = sc.fit_transform(train_df[feature_cols])
    sc_val = val_df.copy();   sc_val[feature_cols] = sc.transform(val_df[feature_cols])

    tr_ds  = RULWindowDataset(sc_tr,  feature_cols, window)
    val_ds = RULWindowDataset(sc_val, feature_cols, window)
    tr_ld  = DataLoader(tr_ds,  batch_size=batch, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=512,   shuffle=False, num_workers=0)

    model = SimpleGRUModel(len(feature_cols), hidden, dense, dropout).to(DEVICE)
    crit  = nn.HuberLoss(delta=delta)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse, patience_ctr = float("inf"), 0
    for epoch in range(150):
        model.train()
        for X, y in tr_ld:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = crit(model(X), y / rul_cap)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for X, y in val_ld:
                preds.extend(model(X.to(DEVICE)).cpu().numpy() * rul_cap)
                true.extend(y.numpy())
        val_rmse = float(np.sqrt(mean_squared_error(true, preds)))

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_rmse < best_rmse:
            best_rmse = val_rmse; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 15:
                break

    return best_rmse


def run_gru_study(all_df, train_units, val_units, feature_cols, rul_cap, n_trials):
    train_df = all_df[all_df["unit"].isin(train_units)]
    val_df   = all_df[all_df["unit"].isin(val_units)]

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study  = optuna.create_study(direction="minimize", pruner=pruner,
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda t: _gru_trial(t, train_df, val_df, feature_cols, rul_cap),
        n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  GRU-Simple  best val RMSE={study.best_value:.4f}  params={best}")

    # Retrain with best params — full training with best-weights restore
    window  = best["window_size"]
    hidden  = best["hidden_size"]
    dense   = best["dense_size"]
    dropout = best["dropout"]
    lr      = best["lr"]
    batch   = best["batch_size"]
    delta   = best["huber_delta"]

    sc = StandardScaler()
    sc_tr  = train_df.copy(); sc_tr[feature_cols]  = sc.fit_transform(train_df[feature_cols])
    sc_val = val_df.copy();   sc_val[feature_cols] = sc.transform(val_df[feature_cols])

    tr_ds  = RULWindowDataset(sc_tr,  feature_cols, window)
    val_ds = RULWindowDataset(sc_val, feature_cols, window)
    tr_ld  = DataLoader(tr_ds,  batch_size=batch, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=512,   shuffle=False, num_workers=0)

    model      = SimpleGRUModel(len(feature_cols), hidden, dense, dropout).to(DEVICE)
    crit       = nn.HuberLoss(delta=delta)
    opt        = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_rmse  = float("inf")
    patience_ctr = 0

    for epoch in range(200):
        model.train()
        for X, y in tr_ld:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = crit(model(X), y / rul_cap)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for X, y in val_ld:
                preds.extend(model(X.to(DEVICE)).cpu().numpy() * rul_cap)
                true.extend(y.numpy())
        val_rmse = float(np.sqrt(mean_squared_error(true, preds)))
        if epoch % 20 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>4}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 20:
                break

    model.load_state_dict(best_state)
    print(f"  GRU-Simple  final best val RMSE={best_rmse:.4f}")

    scaler_pkg = {
        "scaler":       sc,
        "feature_cols": feature_cols,
        "window_size":  window,
        "rul_cap":      rul_cap,
        "hidden_size":  hidden,
        "dense_size":   dense,
        "dropout":      dropout,
    }
    return model, scaler_pkg, best_rmse, best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params       = yaml.safe_load(open("params.yaml"))
    ft_p         = params["finetune"]
    rul_cap      = params["prepare"]["rul_cap"]
    meta         = json.load(open("data/processed/metadata.json"))
    feature_cols = meta["feature_cols"]
    stat_cols    = meta["stat_cols"]
    train_units  = meta["train_units"]
    val_units    = meta["val_units"]

    all_df = pd.read_parquet("data/processed/all_cycles.parquet")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[finetune] device: {device_str}")
    print(f"[finetune] trials — XGB:{ft_p['n_trials_xgb']}  "
          f"XGB-Rolling:{ft_p['n_trials_xgb_rolling']}  "
          f"GRU:{ft_p['n_trials_gru']}")

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    # ── XGBoost ───────────────────────────────────────────────────────────
    print(f"\n[finetune] ── XGBoost ({ft_p['n_trials_xgb']} trials) ──")
    ft_xgb, xgb_pkg, xgb_rmse, xgb_best = run_xgb_study(
        all_df, train_units, val_units, feature_cols,
        ft_p["n_trials_xgb"], device_str)
    ft_xgb.save_model("models/ft_xgb_rul.ubj")
    with open("models/ft_xgb_scaler.pkl", "wb") as f:
        pickle.dump(xgb_pkg, f)

    # ── XGBoost-Rolling ───────────────────────────────────────────────────
    print(f"\n[finetune] ── XGBoost-Rolling ({ft_p['n_trials_xgb_rolling']} trials) ──")
    ft_xgb_roll, xgb_roll_pkg, xgb_roll_rmse, xgb_roll_best = run_xgb_rolling_study(
        all_df, train_units, val_units, stat_cols, feature_cols,
        ft_p["n_trials_xgb_rolling"], device_str)
    ft_xgb_roll.save_model("models/ft_xgb_rolling_rul.ubj")
    with open("models/ft_xgb_rolling_scaler.pkl", "wb") as f:
        pickle.dump(xgb_roll_pkg, f)

    # ── GRU-Simple ────────────────────────────────────────────────────────
    print(f"\n[finetune] ── GRU-Simple ({ft_p['n_trials_gru']} trials, GPU+MedianPruner) ──")
    ft_gru, gru_pkg, gru_rmse, gru_best = run_gru_study(
        all_df, train_units, val_units, feature_cols,
        rul_cap, ft_p["n_trials_gru"])
    torch.save(ft_gru.state_dict(), "models/ft_gru_simple_rul.pt")
    with open("models/ft_gru_simple_scaler.pkl", "wb") as f:
        pickle.dump(gru_pkg, f)

    # ── Select best model ─────────────────────────────────────────────────
    candidates = {
        "XGBoost":        xgb_rmse,
        "XGBoost-Rolling": xgb_roll_rmse,
        "GRU-Simple":     gru_rmse,
    }
    best_name = min(candidates, key=candidates.get)
    best_rmse = candidates[best_name]

    print(f"\n[finetune] ── Results ──")
    for name, rmse_val in candidates.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<20}  val RMSE = {rmse_val:.4f}{marker}")

    best_info = {
        "best_model":    best_name,
        "best_val_rmse": round(best_rmse, 4),
        "all_val_rmse": {k: round(v, 4) for k, v in candidates.items()},
        "best_params": {
            "XGBoost":         xgb_best,
            "XGBoost-Rolling": xgb_roll_best,
            "GRU-Simple":      gru_best,
        },
    }
    json.dump(best_info, open("models/best_model.json", "w"), indent=2)

    metrics = {
        "XGBoost_val_rmse":         round(xgb_rmse, 4),
        "XGBoost_Rolling_val_rmse": round(xgb_roll_rmse, 4),
        "GRU_Simple_val_rmse":      round(gru_rmse, 4),
        "best_model":               best_name,
        "best_val_rmse":            round(best_rmse, 4),
    }
    json.dump(metrics, open("metrics/finetune.json", "w"), indent=2)

    # ── MLflow ────────────────────────────────────────────────────────────
    setup_mlflow("Finetune")
    with mlflow.start_run(run_name="finetune"):
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_param("best_model", best_name)
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_best.items()})
        mlflow.log_params({f"xgb_roll_{k}": v for k, v in xgb_roll_best.items()})
        mlflow.log_params({f"gru_{k}": v for k, v in gru_best.items()})

    print(f"\n[finetune] Done. Best model: {best_name}  (val RMSE={best_rmse:.4f})")


if __name__ == "__main__":
    main()
