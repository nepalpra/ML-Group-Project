"""
Problem 6 — Advanced SHAP Analysis
=====================================
Computes:
  1. Conditional SHAP — separate mean |SHAP| profiles for Uber vs Lyft rides.
  2. Within-tier SHAP — feature importance for individual service tiers.
  3. SHAP stability — rank variance across 5 random seeds (500 rows each).
  4. SHAP interaction values — top feature-pair interactions (500 rows, ~10 min).

Usage:
    python scripts/compute_advanced_shap.py [--skip-interactions]

Artifacts saved to model/:
    shap_conditional.pkl    — dict: {Uber: mean_abs_shap, Lyft: mean_abs_shap}
    shap_within_tier.pkl    — dict: {tier_name: mean_abs_shap}
    shap_stability.pkl      — DataFrame: feature × seed × mean_abs_shap
    shap_interactions.npy   — (500, 45, 45) interaction matrix (if not skipped)
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"


def load_artifacts():
    lgb_model  = joblib.load(MODEL_DIR / "lgb_model.pkl")
    lgb_sv     = np.load(MODEL_DIR / "lgb_shap_vals.npy")      # (5000, 45)
    X_shap     = np.load(MODEL_DIR / "X_shap.npy")              # (5000, 45)
    feat_names = np.load(MODEL_DIR / "feature_names.npy",
                         allow_pickle=True).tolist()
    print(f"SHAP sample: {lgb_sv.shape[0]} rows × {lgb_sv.shape[1]} features")
    return lgb_model, lgb_sv, X_shap, feat_names


def get_platform_mask(X_shap: np.ndarray, feat_names: list):
    """Return boolean masks for Uber and Lyft rows in the SHAP sample."""
    uber_col = feat_names.index("cab_type_Uber")
    lyft_col = feat_names.index("cab_type_Lyft")
    uber_mask = X_shap[:, uber_col] > 0.5
    lyft_mask = X_shap[:, lyft_col] > 0.5
    return uber_mask, lyft_mask


def get_tier_masks(X_shap: np.ndarray, feat_names: list):
    """Return dict of {tier_name: boolean_mask} for name_* features."""
    masks = {}
    for i, fn in enumerate(feat_names):
        if fn.startswith("name_"):
            tier = fn[len("name_"):]
            masks[tier] = X_shap[:, i] > 0.5
    return masks


def mean_abs_shap(sv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature over masked rows. Returns array of shape (n_feats,)."""
    if mask.sum() == 0:
        return np.zeros(sv.shape[1])
    return np.abs(sv[mask]).mean(axis=0)


def main(skip_interactions: bool = False):
    lgb_model, lgb_sv, X_shap, feat_names = load_artifacts()

    # ── 1. Conditional SHAP ───────────────────────────────────────────────────
    print("\n── Conditional SHAP (Uber vs Lyft) ──")
    uber_mask, lyft_mask = get_platform_mask(X_shap, feat_names)
    print(f"  Uber rows: {uber_mask.sum()}  Lyft rows: {lyft_mask.sum()}")

    conditional = {
        "Uber": mean_abs_shap(lgb_sv, uber_mask).tolist(),
        "Lyft": mean_abs_shap(lgb_sv, lyft_mask).tolist(),
        "All" : mean_abs_shap(lgb_sv, np.ones(len(lgb_sv), dtype=bool)).tolist(),
        "feature_names": feat_names,
    }
    joblib.dump(conditional, MODEL_DIR / "shap_conditional.pkl")
    print("  Saved shap_conditional.pkl")

    # ── 2. Within-tier SHAP ───────────────────────────────────────────────────
    print("\n── Within-Tier SHAP ──")
    tier_masks = get_tier_masks(X_shap, feat_names)
    within_tier = {"feature_names": feat_names}
    for tier, mask in tier_masks.items():
        n = mask.sum()
        if n < 20:
            continue
        within_tier[tier] = mean_abs_shap(lgb_sv, mask).tolist()
        print(f"  {tier}: {n} rows")
    joblib.dump(within_tier, MODEL_DIR / "shap_within_tier.pkl")
    print("  Saved shap_within_tier.pkl")

    # ── 3. SHAP Stability ────────────────────────────────────────────────────
    print("\n── SHAP Stability (5 seeds × 500 rows) ──")
    n_stability = 500
    n_seeds     = 5
    rng         = np.random.default_rng(0)
    stability_records = []
    for seed in range(n_seeds):
        idx  = rng.choice(len(lgb_sv), n_stability, replace=False)
        sv_s = lgb_sv[idx]
        m_abs = np.abs(sv_s).mean(axis=0)
        ranks = pd.Series(m_abs, index=feat_names).rank(ascending=False).astype(int)
        for feat, mean_val, rank in zip(feat_names, m_abs, ranks):
            stability_records.append(dict(
                seed=seed, feature=feat,
                mean_abs_shap=float(mean_val), rank=int(rank),
            ))

    stability_df = pd.DataFrame(stability_records)
    # Rank stability: std of rank across seeds per feature
    rank_stability = (
        stability_df.groupby("feature")
        .agg(
            mean_rank=("rank", "mean"),
            std_rank =("rank", "std"),
            mean_abs =("mean_abs_shap", "mean"),
        )
        .reset_index()
        .sort_values("mean_rank")
    )
    joblib.dump({
        "stability_df": stability_df,
        "rank_stability": rank_stability,
        "feature_names": feat_names,
    }, MODEL_DIR / "shap_stability.pkl")
    print(f"  Saved shap_stability.pkl  ({len(stability_df)} records)")
    print("  Top-5 most stable features (lowest rank std):")
    for _, row in rank_stability.head(5).iterrows():
        print(f"    {row.feature}: mean_rank={row.mean_rank:.1f}  "
              f"rank_std={row.std_rank:.2f}")

    # ── 4. SHAP Interaction Values ────────────────────────────────────────────
    if not skip_interactions:
        print("\n── SHAP Interaction Values (500 rows — may take ~10 min) ──")
        t0    = time.time()
        n_int = 500
        rng2  = np.random.default_rng(99)
        idx2  = rng2.choice(len(X_shap), n_int, replace=False)
        X_int = X_shap[idx2]

        explainer = shap.TreeExplainer(lgb_model)
        interact  = explainer.shap_interaction_values(X_int)   # (n, 45, 45)
        np.save(MODEL_DIR / "shap_interactions.npy", interact)
        print(f"  Saved shap_interactions.npy  ({interact.shape})  "
              f"elapsed: {(time.time()-t0)/60:.1f} min")

        # Summary: mean |interaction| per pair (off-diagonal)
        mean_interact = np.abs(interact).mean(axis=0)
        np.fill_diagonal(mean_interact, 0)  # ignore self-interactions
        pairs = []
        n_feats = len(feat_names)
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                pairs.append(dict(
                    feat_i=feat_names[i], feat_j=feat_names[j],
                    interaction=float(mean_interact[i, j]),
                ))
        pairs_df = pd.DataFrame(pairs).sort_values("interaction", ascending=False)
        print("  Top-5 interaction pairs:")
        for _, row in pairs_df.head(5).iterrows():
            print(f"    {row.feat_i} × {row.feat_j}: {row.interaction:.4f}")
    else:
        print("\n── Skipping interaction values (--skip-interactions) ──")

    print(f"\nAll SHAP artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-interactions", action="store_true",
                        help="Skip slow SHAP interaction value computation")
    args = parser.parse_args()
    main(skip_interactions=args.skip_interactions)
