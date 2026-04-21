"""
Problem 8 — Demand & Supply Modelling
=======================================
Trains a LightGBM surge *classifier*: P(surge_multiplier > 1.0 | context).
No surge_multiplier in the feature set — that would be leakage.
Also saves pre-aggregated demand tables used by the Demand & Supply app page.

Usage:
    python scripts/train_surge_model.py

Artifacts saved to model/:
    surge_classifier.pkl   — LGB binary classifier (predicts P(surge>1))
    surge_regressor.pkl    — LGB regressor (predicts surge magnitude | surge>1)
    surge_metrics.json     — classifier + regressor evaluation metrics
    demand_agg.pkl         — pre-aggregated demand tables (Pandas dict)
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH  = REPO_ROOT / "data" / "rideshare_kaggle.csv"
MODEL_DIR  = REPO_ROOT / "model"

# Surge model feature set — NO surge_multiplier (that's what we're predicting)
NUM_FEATS = ["distance", "hour", "is_weekend",
             "temperature", "precipProbability", "windSpeed"]
CAT_FEATS = ["cab_type", "source", "destination"]


def _time_bucket(h):
    if   6 <= h < 10: return "Morning Commute"
    elif 10 <= h < 16: return "Daytime"
    elif 16 <= h < 20: return "Evening Commute"
    elif 20 <= h < 24: return "Nightlife"
    else:              return "Late Night"


def load_and_prepare():
    print("Loading data…")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df.dropna(subset=["price", "surge_multiplier"], inplace=True)

    df["is_weekend"]    = df["datetime"].dt.day_name().isin(
        ["Saturday", "Sunday"]).astype(int)
    df["day_of_week"]   = df["datetime"].dt.day_name()
    df["time_period"]   = df["hour"].apply(_time_bucket)
    df["surge_active"]  = (df["surge_multiplier"] > 1.0).astype(int)

    wc = df["short_summary"].value_counts()
    df["weather_group"] = df["short_summary"].where(
        df["short_summary"].isin(wc[wc >= 1000].index), "Other"
    )
    df["month"] = df["datetime"].dt.month

    print(f"  Surge rate: {df['surge_active'].mean():.3f} "
          f"({df['surge_active'].sum():,} / {len(df):,} rides)")
    return df


def build_preprocessor(X_train: pd.DataFrame):
    """Build and fit a ColumnTransformer for the surge feature set."""
    prep = ColumnTransformer([
        ("num", "passthrough",                                 NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
    ])
    prep.fit(X_train)
    return prep


def main():
    df = load_and_prepare()

    feat_cols = NUM_FEATS + CAT_FEATS

    # Stratified split (matching nb split style)
    strat = (df["weather_group"] + "_" + df["cab_type"] + "_" + df["time_period"])
    X     = df[feat_cols]
    y_cls = df["surge_active"].values
    y_reg = df.loc[df.surge_active == 1, "surge_multiplier"].values

    X_tr, X_te, y_cls_tr, y_cls_te = train_test_split(
        X, y_cls, test_size=0.15, random_state=42, stratify=strat
    )

    prep = build_preprocessor(X_tr)
    X_tr_enc = prep.transform(X_tr)
    X_te_enc = prep.transform(X_te)

    # ── 1. Binary surge classifier ────────────────────────────────────────────
    print("\n── Surge Classifier (P(surge > 1.0)) ──")
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators     = 1000,
        learning_rate    = 0.05,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        class_weight     = "balanced",
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )
    clf.fit(
        X_tr_enc, y_cls_tr,
        eval_set  = [(X_te_enc, y_cls_te)],
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    proba      = clf.predict_proba(X_te_enc)[:, 1]
    pred_cls   = (proba >= 0.5).astype(int)

    clf_metrics = dict(
        accuracy         = float(accuracy_score(y_cls_te, pred_cls)),
        roc_auc          = float(roc_auc_score(y_cls_te, proba)),
        avg_precision    = float(average_precision_score(y_cls_te, proba)),
        surge_rate_train = float(y_cls_tr.mean()),
        surge_rate_test  = float(y_cls_te.mean()),
    )
    print(f"  Accuracy={clf_metrics['accuracy']:.4f}  "
          f"ROC-AUC={clf_metrics['roc_auc']:.4f}  "
          f"AP={clf_metrics['avg_precision']:.4f}  "
          f"({time.time()-t0:.0f}s)")

    # Calibration curve data for the app
    frac_pos, mean_pred = calibration_curve(y_cls_te, proba, n_bins=10)
    clf_metrics["calibration"] = dict(
        fraction_of_positives = frac_pos.tolist(),
        mean_predicted_value  = mean_pred.tolist(),
    )
    joblib.dump(clf, MODEL_DIR / "surge_classifier.pkl")
    joblib.dump(prep, MODEL_DIR / "surge_preprocessor.pkl")

    # ── 2. Surge magnitude regressor (conditioned on surge > 1) ───────────────
    print("\n── Surge Magnitude Regressor (surge_multiplier | surge > 1) ──")
    # Re-filter to only surging rows
    surge_mask_tr = y_cls_tr == 1
    surge_mask_te = y_cls_te == 1
    X_surge_tr    = X_tr_enc[surge_mask_tr]
    y_surge_tr    = X_tr.iloc[surge_mask_tr].index  # wrong — use original
    # Correct approach: use original df
    # Re-encode from df directly for surging rides
    df_surge = df[df.surge_active == 1][feat_cols + ["surge_multiplier"]].dropna()
    X_s = df_surge[feat_cols]
    y_s = df_surge["surge_multiplier"].values

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_s, y_s, test_size=0.15, random_state=42
    )
    Xs_tr_enc = prep.transform(Xs_tr)
    Xs_te_enc = prep.transform(Xs_te)

    t0 = time.time()
    reg = lgb.LGBMRegressor(
        n_estimators     = 800,
        learning_rate    = 0.05,
        num_leaves       = 31,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )
    reg.fit(
        Xs_tr_enc, ys_tr,
        eval_set  = [(Xs_te_enc, ys_te)],
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    reg_pred = reg.predict(Xs_te_enc)
    reg_metrics = dict(
        mae = float(mean_absolute_error(ys_te, reg_pred)),
        r2  = float(r2_score(ys_te, reg_pred)),
        mean_surge_true = float(ys_te.mean()),
        mean_surge_pred = float(reg_pred.mean()),
    )
    print(f"  MAE={reg_metrics['mae']:.4f}  R²={reg_metrics['r2']:.4f}  ({time.time()-t0:.0f}s)")
    joblib.dump(reg, MODEL_DIR / "surge_regressor.pkl")

    # ── 3. Demand aggregation tables ──────────────────────────────────────────
    print("\n── Pre-aggregating demand tables ──")
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Hour × day_of_week surge probability
    surge_heatmap = (
        df.groupby(["day_of_week", "hour"])["surge_active"]
        .agg(surge_prob="mean", count="count")
        .reset_index()
    )
    surge_heatmap["day_of_week"] = pd.Categorical(
        surge_heatmap["day_of_week"], categories=dow_order, ordered=True
    )

    # Ride volume: hour × source
    vol_heat = df.groupby(["hour", "source"]).size().reset_index(name="count")

    # Best-time-to-travel: median price ± IQR by hour, route, platform
    best_time = (
        df.groupby(["source", "destination", "cab_type", "hour"])["price"]
        .agg(median="median",
             q25=lambda x: float(np.percentile(x, 25)),
             q75=lambda x: float(np.percentile(x, 75)),
             count="count")
        .reset_index()
    )

    # Hourly surge stats
    hourly_surge = (
        df.groupby("hour")
        .agg(surge_prob=("surge_active", "mean"),
             mean_surge=("surge_multiplier", "mean"),
             ride_count=("surge_active", "count"))
        .reset_index()
    )

    demand_agg = dict(
        surge_heatmap  = surge_heatmap,
        vol_heat       = vol_heat,
        best_time      = best_time,
        hourly_surge   = hourly_surge,
    )
    joblib.dump(demand_agg, MODEL_DIR / "demand_agg.pkl")
    print(f"  demand_agg saved ({len(best_time):,} route×platform×hour records)")

    # ── Save metrics ──────────────────────────────────────────────────────────
    surge_metrics = dict(
        classifier = clf_metrics,
        regressor  = reg_metrics,
    )
    with open(MODEL_DIR / "surge_metrics.json", "w") as f:
        json.dump(surge_metrics, f, indent=2)

    print(f"\nAll artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
