"""
Statistical compute utilities for the rigour analysis pages.
All heavy functions are @st.cache_data so they only run once per session.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR  = REPO_ROOT / "model"

# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _bootstrap_stat(arr: np.ndarray, stat_fn, n_boot: int, rng: np.random.Generator):
    """Draw n_boot bootstrap samples and apply stat_fn to each."""
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.array([stat_fn(arr[i]) for i in idx])


@st.cache_data(show_spinner="Computing bootstrap confidence intervals…")
def compute_platform_bootstrap(_df: pd.DataFrame, n_boot: int = 800):
    """
    For every route, bootstrap 95% CIs on median price for Uber and Lyft.
    Returns a DataFrame with columns:
      route, uber_median, uber_lo, uber_hi, lyft_median, lyft_lo, lyft_hi,
      diff_median, diff_lo, diff_hi, significant
    'significant' = True when the 95% CI on (lyft - uber) excludes 0.
    """
    rng = np.random.default_rng(42)
    records = []
    for route, grp in _df.groupby("route"):
        uber = grp.loc[grp.cab_type == "Uber", "price"].values
        lyft = grp.loc[grp.cab_type == "Lyft", "price"].values
        if len(uber) < 30 or len(lyft) < 30:
            continue

        u_boot = _bootstrap_stat(uber, np.median, n_boot, rng)
        l_boot = _bootstrap_stat(lyft, np.median, n_boot, rng)
        d_boot = l_boot - u_boot

        records.append(dict(
            route        = route,
            uber_median  = float(np.median(uber)),
            uber_lo      = float(np.percentile(u_boot, 2.5)),
            uber_hi      = float(np.percentile(u_boot, 97.5)),
            lyft_median  = float(np.median(lyft)),
            lyft_lo      = float(np.percentile(l_boot, 2.5)),
            lyft_hi      = float(np.percentile(l_boot, 97.5)),
            diff_median  = float(np.median(d_boot)),
            diff_lo      = float(np.percentile(d_boot, 2.5)),
            diff_hi      = float(np.percentile(d_boot, 97.5)),
            n_uber       = len(uber),
            n_lyft       = len(lyft),
        ))

    df_ci = pd.DataFrame(records)
    df_ci["significant"] = ~(
        (df_ci["diff_lo"] <= 0) & (df_ci["diff_hi"] >= 0)
    )
    df_ci["cheaper"] = df_ci["diff_median"].apply(
        lambda d: "Lyft" if d > 0 else "Uber"
    )
    df_ci.sort_values("diff_median", inplace=True)
    return df_ci


# ── Hypothesis tests ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running Welch t-tests (Bonferroni corrected)…")
def compute_route_significance(_df: pd.DataFrame):
    """
    Welch two-sample t-test for each route: H0 = Uber and Lyft prices same.
    Returns DataFrame with p-values and Bonferroni-corrected significance flags.
    """
    records = []
    for route, grp in _df.groupby("route"):
        uber = grp.loc[grp.cab_type == "Uber", "price"].values
        lyft = grp.loc[grp.cab_type == "Lyft", "price"].values
        if len(uber) < 30 or len(lyft) < 30:
            continue
        t, p = scipy_stats.ttest_ind(uber, lyft, equal_var=False)
        records.append(dict(
            route       = route,
            uber_median = float(np.median(uber)),
            lyft_median = float(np.median(lyft)),
            difference  = float(np.median(lyft) - np.median(uber)),
            t_stat      = float(t),
            p_raw       = float(p),
            n_uber      = len(uber),
            n_lyft      = len(lyft),
        ))

    df_tests = pd.DataFrame(records)
    n_tests  = len(df_tests)
    # Bonferroni correction
    df_tests["p_bonferroni"]  = (df_tests["p_raw"] * n_tests).clip(upper=1.0)
    df_tests["sig_bonferroni"]= df_tests["p_bonferroni"] < 0.05
    # BH (Benjamini-Hochberg) FDR correction for reference
    df_sorted = df_tests.sort_values("p_raw").reset_index(drop=True)
    m = len(df_sorted)
    df_sorted["bh_threshold"]  = (df_sorted.index + 1) / m * 0.05
    df_sorted["sig_bh"]        = df_sorted["p_raw"] <= df_sorted["bh_threshold"]
    # propagate BH decision (once False, all subsequent are False)
    last_true = df_sorted["sig_bh"].values.nonzero()[0]
    if len(last_true):
        df_sorted.loc[: last_true[-1], "sig_bh"] = True
    df_tests = df_tests.merge(df_sorted[["route", "bh_threshold", "sig_bh"]], on="route", how="left")
    df_tests.sort_values("p_bonferroni", inplace=True)
    return df_tests, n_tests


# ── Weather (confounder-adjusted) ─────────────────────────────────────────────

@st.cache_data(show_spinner="Adjusting weather effect for time-of-day and service tier…")
def compute_weather_adjusted(_df: pd.DataFrame):
    """
    Computes:
      1. Raw average price per weather group (confounded).
      2. Average price residual per weather group after regressing out
         time_period + service tier (removes time-of-day and tier effects).
    Returns (raw_df, adjusted_df, overall_mean).
    """
    df_m = _df[["price", "weather_group", "time_period", "name"]].dropna().copy()
    overall_mean = float(df_m["price"].mean())

    # --- Raw ---
    raw = (
        df_m.groupby("weather_group")["price"]
        .agg(mean_price="mean", count="count", std="std")
        .reset_index()
    )
    raw["se"] = raw["std"] / np.sqrt(raw["count"])
    raw["ci_lo"] = raw["mean_price"] - 1.96 * raw["se"]
    raw["ci_hi"] = raw["mean_price"] + 1.96 * raw["se"]
    raw["deviation"] = raw["mean_price"] - overall_mean

    # --- Adjusted ---
    # Encode confounders (time_period + name) as dummies
    tp_dummies   = pd.get_dummies(df_m["time_period"], prefix="tp").astype(float)
    tier_dummies = pd.get_dummies(df_m["name"],        prefix="tr").astype(float)
    X_conf       = pd.concat([tp_dummies, tier_dummies], axis=1).values
    y            = df_m["price"].values

    resid = y - LinearRegression().fit(X_conf, y).predict(X_conf)
    df_m  = df_m.copy()
    df_m["residual"] = resid

    adj = (
        df_m.groupby("weather_group")["residual"]
        .agg(mean_resid="mean", count="count", std="std")
        .reset_index()
    )
    adj["se"]     = adj["std"] / np.sqrt(adj["count"])
    adj["ci_lo"]  = adj["mean_resid"] - 1.96 * adj["se"]
    adj["ci_hi"]  = adj["mean_resid"] + 1.96 * adj["se"]

    return raw, adj, overall_mean


# ── KMeans validation ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running KMeans for k = 2 … 10 (silhouette + inertia)…")
def compute_kmeans_validation(_df: pd.DataFrame, k_max: int = 10):
    """
    Fits KMeans for k in [2, k_max] on the same feature set as compute_q6.
    Returns DataFrame with columns: k, inertia, silhouette.
    Uses a 10 000-row stratified sample for speed.
    """
    feats   = ["hour", "surge_multiplier", "price", "distance",
               "temperature", "precipProbability", "windSpeed", "is_weekend"]
    df_c    = _df[feats].dropna()
    sample  = df_c.sample(min(10_000, len(df_c)), random_state=42)
    X       = StandardScaler().fit_transform(sample)

    rows = []
    for k in range(2, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
        lbl = km.fit_predict(X)
        sil = silhouette_score(X, lbl, sample_size=min(5000, len(X)), random_state=42)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=float(sil)))

    return pd.DataFrame(rows)


# ── Normality tests ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running normality tests per tier…")
def compute_normality_tests(_df: pd.DataFrame):
    """
    D'Agostino-Pearson normality test for price distribution of each service tier.
    Returns DataFrame with columns: name, cab_type, n, stat, p_value, normal.
    """
    records = []
    for name, grp in _df.groupby("name"):
        prices = grp["price"].dropna().values
        if len(prices) < 20:
            continue
        sample = prices if len(prices) <= 5000 else np.random.default_rng(42).choice(prices, 5000, replace=False)
        stat, p = scipy_stats.normaltest(sample)
        cab = grp["cab_type"].iloc[0]
        records.append(dict(
            name     = name,
            cab_type = cab,
            n        = len(prices),
            stat     = float(stat),
            p_value  = float(p),
            normal   = bool(p > 0.05),
        ))
    return pd.DataFrame(records).sort_values("p_value", ascending=False)


# ── Surge patterns from data ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing surge probability patterns…")
def compute_surge_patterns(_df: pd.DataFrame):
    """
    Computes:
      1. P(surge > 1.0) by hour × day_of_week heatmap.
      2. Mean surge_multiplier by hour.
      3. Surge rate by weather group.
    """
    df_s = _df[["hour", "day_of_week", "surge_multiplier", "weather_group",
                "source", "destination", "route"]].dropna().copy()
    df_s["surge_active"] = (df_s["surge_multiplier"] > 1.0).astype(int)

    # Hour × day_of_week heatmap
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = (
        df_s.groupby(["day_of_week", "hour"])["surge_active"]
        .mean()
        .reset_index()
        .rename(columns={"surge_active": "surge_prob"})
    )
    heatmap["day_of_week"] = pd.Categorical(heatmap["day_of_week"], categories=dow_order, ordered=True)
    heatmap.sort_values(["day_of_week", "hour"], inplace=True)

    # Hourly average surge
    hourly = (
        df_s.groupby("hour")
        .agg(surge_prob=("surge_active", "mean"),
             mean_surge=("surge_multiplier", "mean"),
             count=("surge_active", "count"))
        .reset_index()
    )

    # Weather group surge rate
    weather_surge = (
        df_s.groupby("weather_group")
        .agg(surge_prob=("surge_active", "mean"),
             mean_surge=("surge_multiplier", "mean"),
             count=("surge_active", "count"))
        .reset_index()
        .sort_values("surge_prob", ascending=False)
    )

    # Route surge rate (top routes)
    route_surge = (
        df_s.groupby("route")
        .agg(surge_prob=("surge_active", "mean"),
             mean_surge=("surge_multiplier", "mean"),
             count=("surge_active", "count"))
        .reset_index()
        .sort_values("surge_prob", ascending=False)
    )

    return heatmap, hourly, weather_surge, route_surge


# ── Best-time-to-travel from historical data ──────────────────────────────────

@st.cache_data(show_spinner="Computing hourly price profiles…")
def compute_best_time(_df: pd.DataFrame, source: str, destination: str, cab_type: str):
    """
    For a given route + platform, returns hourly median price + IQR from data.
    """
    mask = (
        (_df["source"]      == source) &
        (_df["destination"] == destination) &
        (_df["cab_type"]    == cab_type)
    )
    grp = _df[mask].copy()
    if grp.empty:
        return pd.DataFrame()

    hourly = (
        grp.groupby("hour")["price"]
        .agg(
            median="median",
            q25   =lambda x: x.quantile(0.25),
            q75   =lambda x: x.quantile(0.75),
            mean  ="mean",
            count ="count",
        )
        .reset_index()
    )
    return hourly


# ── Demand volume aggregation ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_demand_patterns(_df: pd.DataFrame):
    """
    Returns:
      1. hour × source volume heatmap
      2. hourly total ride volume
    """
    vol_heat = (
        _df.groupby(["hour", "source"])
        .size()
        .reset_index(name="count")
    )
    vol_hourly = (
        _df.groupby("hour")
        .size()
        .reset_index(name="count")
    )
    vol_dow = (
        _df.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    vol_dow["day_of_week"] = pd.Categorical(vol_dow["day_of_week"], categories=dow_order, ordered=True)
    vol_dow.sort_values(["day_of_week", "hour"], inplace=True)

    return vol_heat, vol_hourly, vol_dow


# ── Route price-per-mile bootstrap ───────────────────────────────────────────

@st.cache_data(show_spinner="Computing route $/mile confidence intervals…")
def compute_route_ppm_bootstrap(_df: pd.DataFrame, n_boot: int = 500, top_n: int = 20):
    """
    Bootstrap 95% CIs on mean price_per_mile for every route.
    Returns top_n routes by mean $/mile with CI columns + significance flag
    (sig_above_mean = True when CI lower bound exceeds the dataset-wide mean $/mile).
    """
    rng     = np.random.default_rng(42)
    overall = float(_df["price_per_mile"].mean())
    records = []
    for route, grp in _df.groupby("route"):
        ppm = grp["price_per_mile"].dropna().values
        if len(ppm) < 20:
            continue
        boot = np.array([
            np.mean(rng.choice(ppm, len(ppm), replace=True))
            for _ in range(n_boot)
        ])
        records.append(dict(
            route    = route,
            mean_ppm = float(ppm.mean()),
            ci_lo    = float(np.percentile(boot, 2.5)),
            ci_hi    = float(np.percentile(boot, 97.5)),
            n        = len(ppm),
        ))
    df_boot = (
        pd.DataFrame(records)
        .sort_values("mean_ppm", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df_boot["sig_above_mean"] = df_boot["ci_lo"] > overall
    df_boot["overall_mean"]   = overall
    return df_boot


# ── Artifact availability helpers ────────────────────────────────────────────

def surge_model_available() -> bool:
    return (MODEL_DIR / "surge_classifier.pkl").exists()

def advanced_models_available() -> bool:
    return (MODEL_DIR / "lgb_q50.pkl").exists()

def hpo_results_available() -> bool:
    return (MODEL_DIR / "hpo_results.json").exists()

def advanced_shap_available() -> bool:
    return (MODEL_DIR / "shap_conditional.pkl").exists()

def time_split_available() -> bool:
    return (MODEL_DIR / "time_split_metrics.json").exists()
