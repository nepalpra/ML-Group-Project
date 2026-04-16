"""
Problem 7 — Extended Model Architectures
==========================================
Trains:
  1. Quantile LightGBM — q10, q50, q90 (prediction intervals)
  2. MLP (sklearn) — neural baseline
  3. Time-ordered validation — train on November, test on December

Usage:
    python scripts/train_advanced_models.py

Artifacts saved to model/:
    lgb_q10.pkl, lgb_q50.pkl, lgb_q90.pkl
    mlp_model.pkl
    advanced_metrics.json    — all model test metrics
    time_split_metrics.json  — Nov→Dec temporal validation results
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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


def load_data():
    print("Loading data…")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df.dropna(subset=["price"], inplace=True)

    wc = df["short_summary"].value_counts()
    df["weather_group"] = df["short_summary"].where(
        df["short_summary"].isin(wc[wc >= 1000].index), "Other"
    )
    df["is_weekend"]  = df["datetime"].dt.day_name().isin(
        ["Saturday", "Sunday"]).astype(int)
    df["time_period"] = df["hour"].apply(_time_bucket)
    df["month"]       = df["datetime"].dt.month   # 11 = Nov, 12 = Dec

    feat_cols = NUM_FEATS + CAT_FEATS
    df = df[feat_cols + ["price", "weather_group", "time_period", "month"]].dropna()
    print(f"  {len(df):,} rows — Nov: {(df.month==11).sum():,}, Dec: {(df.month==12).sum():,}")
    return df


def metrics(y_true, y_pred):
    return dict(
        mae  = float(mean_absolute_error(y_true, y_pred)),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = float(r2_score(y_true, y_pred)),
    )


def quantile_coverage(y_true, y_lo, y_hi):
    """Empirical coverage: fraction of actuals within the predicted interval."""
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def main():
    df   = load_data()
    prep = joblib.load(MODEL_DIR / "preprocessor.pkl")

    feat_cols  = NUM_FEATS + CAT_FEATS
    advanced   = {}

    # ── Random split (matches existing notebook) ──────────────────────────────
    strat_key = (df["weather_group"] + "_" + df["cab_type"] + "_" + df["time_period"])
    X         = df[feat_cols]
    y         = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=strat_key
    )
    X_tr_enc = prep.transform(X_train)
    X_te_enc = prep.transform(X_test)
    print(f"Random split — Train: {len(y_train):,}  Test: {len(y_test):,}")

    # ── 1. Quantile LightGBM (q10 / q50 / q90) ───────────────────────────────
    print("\n── Quantile LightGBM ──")
    quantile_models = {}
    for q in [0.10, 0.50, 0.90]:
        tag = f"q{int(q*100):02d}"
        print(f"  Training {tag}…", end=" ", flush=True)
        t0 = time.time()
        model = lgb.LGBMRegressor(
            objective       = "quantile",
            alpha           = q,
            n_estimators    = 1500,
            learning_rate   = 0.05,
            num_leaves      = 63,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            random_state    = 42,
            n_jobs          = -1,
            verbose         = -1,
        )
        model.fit(
            X_tr_enc, y_train,
            eval_set=[(X_te_enc, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        pred = model.predict(X_te_enc)
        m    = metrics(y_test, pred)
        quantile_models[tag] = (model, pred)
        advanced[f"lgb_{tag}"] = m
        joblib.dump(model, MODEL_DIR / f"lgb_{tag}.pkl")
        print(f"MAE={m['mae']:.4f}  ({time.time()-t0:.0f}s)")

    # Interval coverage (q10 → q90 = 80% prediction interval)
    _, pred_lo = quantile_models["q10"]
    _, pred_hi = quantile_models["q90"]
    cov = quantile_coverage(y_test, pred_lo, pred_hi)
    advanced["quantile_interval_coverage_80pct"] = cov
    print(f"  80% interval empirical coverage: {cov:.3f}  (target: 0.80)")

    # ── 2. MLP Regressor ─────────────────────────────────────────────────────
    print("\n── MLP Regressor (128-64, max_iter=300) ──")
    t0 = time.time()
    # subsample for MLP speed (sklearn MLP is slow on 540k rows)
    rng     = np.random.default_rng(42)
    tr_idx  = rng.choice(len(y_train), min(150_000, len(y_train)), replace=False)
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr_enc[tr_idx], y_train[tr_idx])
    pred_mlp        = mlp.predict(X_te_enc)
    advanced["mlp"] = metrics(y_test, pred_mlp)
    joblib.dump(mlp, MODEL_DIR / "mlp_model.pkl")
    print(f"  MAE={advanced['mlp']['mae']:.4f}  R²={advanced['mlp']['r2']:.5f}"
          f"  ({time.time()-t0:.0f}s)  iterations={mlp.n_iter_}")

    # ── 3. Time-ordered validation (Nov → Dec) ────────────────────────────────
    print("\n── Time-ordered validation (train=Nov, test=Dec) ──")
    X_nov = df.loc[df.month == 11, feat_cols]
    y_nov = df.loc[df.month == 11, "price"].values
    X_dec = df.loc[df.month == 12, feat_cols]
    y_dec = df.loc[df.month == 12, "price"].values
    print(f"  Nov (train): {len(y_nov):,}  Dec (test): {len(y_dec):,}")

    X_nov_enc = prep.transform(X_nov)
    X_dec_enc = prep.transform(X_dec)

    time_split = {}

    # LGB default (load existing)
    lgb_def = joblib.load(MODEL_DIR / "lgb_model.pkl")
    time_split["lgb_default_on_dec"]  = metrics(y_dec,  lgb_def.predict(X_dec_enc))

    # Retrain LGB on Nov only, evaluate on Dec
    lgb_nov = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    # Use 10% of Nov as validation for early stopping
    split_idx = int(0.9 * len(y_nov))
    lgb_nov.fit(
        X_nov_enc[:split_idx], y_nov[:split_idx],
        eval_set=[(X_nov_enc[split_idx:], y_nov[split_idx:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    time_split["lgb_nov_trained_on_dec"] = metrics(y_dec, lgb_nov.predict(X_dec_enc))
    time_split["lgb_nov_trained_on_nov"] = metrics(y_nov[split_idx:],
                                                    lgb_nov.predict(X_nov_enc[split_idx:]))

    # Platform breakdown on Dec
    dec_df = df[df.month == 12].copy()
    for platform in ["Uber", "Lyft"]:
        mask = dec_df["cab_type"] == platform
        if mask.sum() < 100:
            continue
        X_p = prep.transform(dec_df.loc[mask, feat_cols])
        y_p = dec_df.loc[mask, "price"].values
        time_split[f"lgb_default_dec_{platform.lower()}"] = metrics(y_p, lgb_def.predict(X_p))

    for name, m in time_split.items():
        print(f"  {name}: MAE={m['mae']:.4f}  R²={m['r2']:.5f}")

    # ── Save all metrics ──────────────────────────────────────────────────────
    with open(MODEL_DIR / "advanced_metrics.json", "w") as f:
        json.dump(advanced, f, indent=2)

    with open(MODEL_DIR / "time_split_metrics.json", "w") as f:
        json.dump(time_split, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}")
    print("\n── Advanced model summary ──")
    for name, m in advanced.items():
        if isinstance(m, dict):
            print(f"  {name}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.5f}")


if __name__ == "__main__":
    main()
