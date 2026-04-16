"""
Problem 3 — Hyperparameter Optimisation
========================================
Runs Optuna Bayesian HPO for LightGBM and XGBoost, then retrains the champion
model on the full training set.

Usage (from repo root, with conda env active):
    python scripts/train_tuned_models.py [--trials 30] [--sample 60000]

Artifacts saved to model/:
    tuned_lgb_model.pkl   — champion LightGBM model
    tuned_xgb_model.pkl   — champion XGBoost model
    hpo_results.json      — all trial metrics + best params + default vs tuned comparison
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH  = REPO_ROOT / "data" / "rideshare_kaggle.csv"
MODEL_DIR  = REPO_ROOT / "model"

NUM_FEATS = ["distance", "surge_multiplier", "hour", "is_weekend",
             "temperature", "precipProbability", "windSpeed"]
CAT_FEATS = ["name", "cab_type", "source", "destination"]


def _time_bucket(h):
    if   6 <= h < 10: return "Morning Commute"
    elif 10 <= h < 16: return "Daytime"
    elif 16 <= h < 20: return "Evening Commute"
    elif 20 <= h < 24: return "Nightlife"
    else:              return "Late Night"


def load_and_prepare():
    print("Loading data…")
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["price"], inplace=True)

    wc = df["short_summary"].value_counts()
    df["weather_group"] = df["short_summary"].where(
        df["short_summary"].isin(wc[wc >= 1000].index), "Other"
    )
    df["is_weekend"]  = pd.to_datetime(df["datetime"]).dt.day_name().isin(
        ["Saturday", "Sunday"]).astype(int)
    df["time_period"] = df["hour"].apply(_time_bucket)

    feat_cols = NUM_FEATS + CAT_FEATS
    df = df[feat_cols + ["price", "weather_group", "time_period"]].dropna()

    strat_key = (df["weather_group"] + "_" + df["cab_type"] + "_" + df["time_period"])
    X = df[feat_cols]
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=strat_key
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    prep = joblib.load(MODEL_DIR / "preprocessor.pkl")
    X_tr_enc = prep.transform(X_train)
    X_te_enc = prep.transform(X_test)
    return X_tr_enc, X_te_enc, y_train, y_test, prep


def metrics(y_true, y_pred):
    return dict(
        mae  = float(mean_absolute_error(y_true, y_pred)),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = float(r2_score(y_true, y_pred)),
    )


# ── LightGBM Optuna study ─────────────────────────────────────────────────────

def lgb_objective(trial, X_cv, y_cv, n_splits=3):
    params = dict(
        n_estimators      = trial.suggest_int   ("n_estimators",       500,  3000, step=100),
        learning_rate     = trial.suggest_float  ("learning_rate",      0.01, 0.15, log=True),
        num_leaves        = trial.suggest_int    ("num_leaves",         20,   150),
        min_child_samples = trial.suggest_int    ("min_child_samples",  10,   100),
        subsample         = trial.suggest_float  ("subsample",          0.5,  1.0),
        colsample_bytree  = trial.suggest_float  ("colsample_bytree",   0.5,  1.0),
        reg_alpha         = trial.suggest_float  ("reg_alpha",          1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float  ("reg_lambda",         1e-8, 10.0, log=True),
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []
    for tr_idx, val_idx in kf.split(X_cv):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_cv[tr_idx], y_cv[tr_idx],
            eval_set=[(X_cv[val_idx], y_cv[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        maes.append(mean_absolute_error(y_cv[val_idx], model.predict(X_cv[val_idx])))
    return float(np.mean(maes))


def xgb_objective(trial, X_cv, y_cv, n_splits=3):
    from xgboost import XGBRegressor
    params = dict(
        n_estimators      = trial.suggest_int   ("n_estimators",       500,  3000, step=100),
        learning_rate     = trial.suggest_float  ("learning_rate",      0.01, 0.15, log=True),
        max_depth         = trial.suggest_int    ("max_depth",          3,    10),
        min_child_weight  = trial.suggest_int    ("min_child_weight",   1,    20),
        subsample         = trial.suggest_float  ("subsample",          0.5,  1.0),
        colsample_bytree  = trial.suggest_float  ("colsample_bytree",   0.5,  1.0),
        gamma             = trial.suggest_float  ("gamma",              0,    5.0),
        reg_alpha         = trial.suggest_float  ("reg_alpha",          1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float  ("reg_lambda",         1e-8, 10.0, log=True),
        random_state      = 42,
        n_jobs            = -1,
        tree_method       = "hist",
        eval_metric       = "mae",
        early_stopping_rounds = 50,
    )
    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []
    for tr_idx, val_idx in kf.split(X_cv):
        model = XGBRegressor(**params)
        model.fit(
            X_cv[tr_idx], y_cv[tr_idx],
            eval_set=[(X_cv[val_idx], y_cv[val_idx])],
            verbose=False,
        )
        maes.append(mean_absolute_error(y_cv[val_idx], model.predict(X_cv[val_idx])))
    return float(np.mean(maes))


def main(n_trials: int = 30, sample_size: int = 60_000):
    X_train, X_test, y_train, y_test, prep = load_and_prepare()

    # Use a stratified sub-sample for HPO speed
    n_cv = min(sample_size, len(y_train))
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(y_train), n_cv, replace=False)
    X_cv = X_train[idx]
    y_cv = y_train[idx]

    results = {}

    # ── Default model metrics (loaded from disk) ──────────────────────────────
    print("\n── Default model baselines (from saved artifacts) ──")
    lgb_default = joblib.load(MODEL_DIR / "lgb_model.pkl")
    xgb_default = joblib.load(MODEL_DIR / "xgb_model.pkl")
    results["lgb_default"] = metrics(y_test, lgb_default.predict(X_test))
    results["xgb_default"] = metrics(y_test, xgb_default.predict(X_test))
    for name, m in results.items():
        print(f"  {name}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.5f}")

    # ── LightGBM HPO ─────────────────────────────────────────────────────────
    print(f"\n── LightGBM Optuna HPO ({n_trials} trials, sample={n_cv:,}) ──")
    t0 = time.time()
    lgb_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(
        lambda t: lgb_objective(t, X_cv, y_cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min.  Best MAE (CV): {lgb_study.best_value:.4f}")
    print(f"  Best params: {lgb_study.best_params}")

    # Retrain champion on full train set
    best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42, n_jobs=-1, verbose=-1)
    best_lgb.fit(X_train, y_train)
    results["lgb_tuned"] = metrics(y_test, best_lgb.predict(X_test))
    print(f"  Tuned test: MAE={results['lgb_tuned']['mae']:.4f}  R²={results['lgb_tuned']['r2']:.5f}")
    joblib.dump(best_lgb, MODEL_DIR / "tuned_lgb_model.pkl")

    # Store trial history
    lgb_trials = [
        dict(number=t.number, value=t.value, params=t.params,
             state=str(t.state))
        for t in lgb_study.trials
    ]

    # ── XGBoost HPO ──────────────────────────────────────────────────────────
    print(f"\n── XGBoost Optuna HPO ({n_trials} trials, sample={n_cv:,}) ──")
    t0 = time.time()
    xgb_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(
        lambda t: xgb_objective(t, X_cv, y_cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min.  Best MAE (CV): {xgb_study.best_value:.4f}")

    from xgboost import XGBRegressor
    best_xgb = XGBRegressor(**xgb_study.best_params, random_state=42, n_jobs=-1,
                            tree_method="hist")
    best_xgb.fit(X_train, y_train)
    results["xgb_tuned"] = metrics(y_test, best_xgb.predict(X_test))
    print(f"  Tuned test: MAE={results['xgb_tuned']['mae']:.4f}  R²={results['xgb_tuned']['r2']:.5f}")
    joblib.dump(best_xgb, MODEL_DIR / "tuned_xgb_model.pkl")

    xgb_trials = [
        dict(number=t.number, value=t.value, params=t.params,
             state=str(t.state))
        for t in xgb_study.trials
    ]

    # ── Save results ──────────────────────────────────────────────────────────
    output = dict(
        metrics      = results,
        lgb_best_params = lgb_study.best_params,
        lgb_best_cv_mae = lgb_study.best_value,
        xgb_best_params = xgb_study.best_params,
        xgb_best_cv_mae = xgb_study.best_value,
        lgb_trials   = lgb_trials,
        xgb_trials   = xgb_trials,
        n_trials     = n_trials,
        sample_size  = n_cv,
    )
    with open(MODEL_DIR / "hpo_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n── Improvement summary ──")
    for model in ("lgb", "xgb"):
        d = results[f"{model}_default"]
        t = results[f"{model}_tuned"]
        delta_mae = d["mae"] - t["mae"]
        pct       = delta_mae / d["mae"] * 100
        print(f"  {model.upper()}: MAE {d['mae']:.4f} → {t['mae']:.4f}  "
              f"(Δ={delta_mae:+.4f}, {pct:+.2f}%)")

    print(f"\nArtifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=30)
    parser.add_argument("--sample",  type=int, default=60_000)
    args = parser.parse_args()
    main(n_trials=args.trials, sample_size=args.sample)
