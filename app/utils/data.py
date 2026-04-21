from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "rideshare_kaggle.csv"

NEIGHBORHOODS = [
    "Back Bay", "Beacon Hill", "Boston University", "Fenway",
    "Financial District", "Haymarket Square", "North End",
    "North Station", "Northeastern University", "South Station",
    "Theatre District", "West End",
]

TIME_ORDER = ["Morning Commute", "Daytime", "Evening Commute", "Nightlife", "Late Night"]

# ── Design system ─────────────────────────────────────────────────────────────
# Platform identity — one color per platform, used everywhere
COLORS = {"Uber": "#0F172A", "Lyft": "#E91E8C"}

# Pricing regimes — analytical series palette, semantically mapped
REGIME_COLORS = {
    "Standard":        "#10B981",   # emerald  — baseline / normal
    "Morning Commute": "#F59E0B",   # amber    — warm / time pressure
    "Evening Commute": "#EF4444",   # red      — urgency
    "Surge Heavy":     "#6366F1",   # indigo   — surge (distinct from Lyft pink)
    "Nightlife":       "#8B5CF6",   # violet   — night / cool
    "Bad Weather":     "#06B6D4",   # cyan     — atmospheric
}

# Analytical series palette — for multi-series charts with no platform meaning
SERIES_COLORS = ["#6366F1", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4"]

# Statistical significance colors — semantic, never cross-purposed
SIG_COLORS = {
    "positive":  "#059669",   # emerald-600 — significant, positive
    "negative":  "#DC2626",   # red-600     — significant, negative
    "neutral":   "#2563EB",   # blue-600    — significant, no direction
    "not_sig":   "#94A3B8",   # slate-400   — not significant (muted)
    "reference": "#64748B",   # slate-500   — reference lines / baselines
}


def apply_chart_theme(fig, height: int = None, margin: dict = None):
    """
    Apply the global chart design theme to any Plotly figure.
    Call after building the figure, before st.plotly_chart().
    """
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif",
                  size=12, color="#0F172A"),
        margin=margin or dict(t=40, b=50, l=60, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#E2E8F0",
            font_color="#0F172A",
        ),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(
        showgrid=True, gridcolor="#F1F5F9",
        zeroline=False, tickfont=dict(size=11),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#F1F5F9",
        zeroline=False, tickfont=dict(size=11),
    )
    return fig

SERVICE_MAP = {
    "Shared": "Lyft", "Lyft": "Lyft", "Lyft XL": "Lyft",
    "Lux": "Lyft", "Lux Black": "Lyft", "Lux Black XL": "Lyft",
    "UberPool": "Uber", "UberX": "Uber", "UberXL": "Uber",
    "Black": "Uber", "Black SUV": "Uber", "WAV": "Uber",
}


def _time_bucket(h):
    if   6 <= h < 10: return "Morning Commute"
    elif 10 <= h < 16: return "Daytime"
    elif 16 <= h < 20: return "Evening Commute"
    elif 20 <= h < 24: return "Nightlife"
    else:              return "Late Night"


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["price"], inplace=True)

    df["route"]         = df["source"] + " → " + df["destination"]
    df["price_per_mile"]= df["price"] / df["distance"]
    df["datetime"]      = pd.to_datetime(df["datetime"])
    df["day_of_week"]   = df["datetime"].dt.day_name()
    df["is_weekend"]    = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["time_period"]   = df["hour"].apply(_time_bucket)

    wc = df["short_summary"].value_counts()
    df["weather_group"] = df["short_summary"].where(
        df["short_summary"].isin(wc[wc >= 1000].index), "Other"
    )
    return df


# ── Q1 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_q1(_df: pd.DataFrame):
    agg = (
        _df.groupby(["route", "cab_type"])
        .agg(median_price=("price", "median"), median_ppm=("price_per_mile", "median"),
             ride_count=("price", "count"))
        .reset_index()
    )
    uber = agg[agg.cab_type == "Uber"].set_index("route")[["median_price","median_ppm","ride_count"]].add_prefix("uber_")
    lyft = agg[agg.cab_type == "Lyft"].set_index("route")[["median_price","median_ppm","ride_count"]].add_prefix("lyft_")
    pivot = uber.join(lyft).reset_index()
    pivot["price_diff"] = pivot["lyft_median_price"] - pivot["uber_median_price"]
    pivot["cheaper"]    = pivot["price_diff"].apply(lambda x: "Uber" if x > 0 else "Lyft")
    pivot["pct_diff"]   = (pivot["price_diff"].abs() / pivot[["uber_median_price","lyft_median_price"]].min(axis=1) * 100).round(1)
    pivot[["src","dst"]]= pivot["route"].str.split(" → ", expand=True)
    pivot.sort_values("price_diff", inplace=True)

    time_agg = (
        _df.groupby(["cab_type", "time_period"])
        .agg(median_price=("price","median"), ride_count=("price","count"))
        .reset_index()
    )
    return pivot, time_agg


# ── Q2 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_q2(_df: pd.DataFrame):
    tier_order = _df.groupby("name")["price"].median().sort_values().index.tolist()
    iqr = (
        _df.groupby(["name", "cab_type"])["price"]
        .describe(percentiles=[0.25, 0.75])
        .reset_index()
        .rename(columns={"25%": "q1", "75%": "q3", "50%": "median"})
    )
    iqr["iqr"] = iqr["q3"] - iqr["q1"]
    ppm = (
        _df.groupby("name")["price_per_mile"]
        .median()
        .reindex(tier_order)
        .reset_index()
        .rename(columns={"price_per_mile": "median_ppm"})
    )
    return tier_order, iqr, ppm


# ── Q3 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing R² decomposition…")
def compute_q3(_df: pd.DataFrame):
    df_m = _df[["price","distance","surge_multiplier","name","cab_type"]].dropna().copy()
    nd   = pd.get_dummies(df_m["name"],     prefix="prod").astype(float)
    pd_  = pd.get_dummies(df_m["cab_type"], prefix="plt").astype(float)
    df_m = pd.concat([df_m, nd, pd_], axis=1)
    y    = df_m["price"].values.astype(float)
    nc   = [c for c in df_m.columns if c.startswith("prod_")]
    pc   = [c for c in df_m.columns if c.startswith("plt_")]

    def lone_r2(cols):
        X = df_m[cols].values.astype(float)
        return r2_score(y, LinearRegression().fit(X, y).predict(X))

    individual = {
        "Distance":         lone_r2(["distance"]),
        "Surge multiplier": lone_r2(["surge_multiplier"]),
        "Product type":     lone_r2(nc),
        "Platform":         lone_r2(pc),
    }
    steps = [
        ("Distance",           ["distance"]),
        ("+ Product type",     ["distance"] + nc),
        ("+ Surge multiplier", ["distance"] + nc + ["surge_multiplier"]),
        ("+ Platform",         ["distance"] + nc + ["surge_multiplier"] + pc),
    ]
    rows, prev = [], 0.0
    for label, cols in steps:
        r2 = r2_score(y, LinearRegression().fit(df_m[cols].values.astype(float), y)
                      .predict(df_m[cols].values.astype(float)))
        rows.append({"label": label, "r2": round(r2, 4), "gain": round(r2 - prev, 4)})
        prev = r2
    return individual, pd.DataFrame(rows)


# ── Q4 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_q4(_df: pd.DataFrame):
    route_ppm = (
        _df.groupby(["source","destination"])
        .agg(avg_price=("price","mean"), avg_ppm=("price_per_mile","mean"),
             ride_count=("price","count"))
        .reset_index()
    )
    route_ppm["route"] = route_ppm["source"] + " → " + route_ppm["destination"]
    overpriced = route_ppm.sort_values("avg_ppm", ascending=False).head(15)
    return route_ppm, overpriced


# ── Q5 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_q5(_df: pd.DataFrame):
    weather_agg = (
        _df.groupby(["weather_group","cab_type"])
        .agg(avg_price=("price","mean"), ride_count=("price","count"))
        .reset_index()
    )
    sample = _df.sample(min(15000, len(_df)), random_state=42)
    return weather_agg, sample


# ── Q6 ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running KMeans clustering…")
def compute_q6(_df: pd.DataFrame):
    feats = ["hour","surge_multiplier","price","distance",
             "temperature","precipProbability","windSpeed","is_weekend"]
    df_c  = _df[feats + ["cab_type","day_of_week"]].dropna().copy()
    sample= df_c.sample(min(60000, len(df_c)), random_state=42)

    scaler = StandardScaler()
    X      = scaler.fit_transform(sample[feats])
    km     = KMeans(n_clusters=6, random_state=42, n_init=10)
    sample = sample.copy()
    sample["cluster"] = km.fit_predict(X)

    profile = sample.groupby("cluster")[feats].mean()

    # Rank-based unique label assignment: pick the cluster that best represents
    # each regime in priority order, guaranteeing all 6 labels are used.
    remaining = list(profile.index)
    c2l = {}

    idx = profile.loc[remaining, "surge_multiplier"].idxmax()
    c2l[idx] = "Surge Heavy"; remaining.remove(idx)

    idx = profile.loc[remaining, "hour"].idxmax()
    c2l[idx] = "Nightlife"; remaining.remove(idx)

    idx = profile.loc[remaining, "hour"].idxmin()
    c2l[idx] = "Morning Commute"; remaining.remove(idx)

    idx = profile.loc[remaining, "hour"].idxmax()
    c2l[idx] = "Evening Commute"; remaining.remove(idx)

    idx = profile.loc[remaining, "precipProbability"].idxmax()
    c2l[idx] = "Bad Weather"; remaining.remove(idx)

    c2l[remaining[0]] = "Standard"
    sample["regime"]     = sample["cluster"].map(c2l)

    norm = profile[feats].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    norm.index = norm.index.map(c2l)

    hour_dist = sample.groupby(["hour","regime"]).size().reset_index(name="count")
    stats     = (sample.groupby("regime")
                 .agg(count=("price","count"), avg_price=("price","mean"))
                 .reset_index().sort_values("avg_price", ascending=False))
    return norm, hour_dist, stats


# ── Predictor helper ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_route_distance(_df: pd.DataFrame) -> pd.DataFrame:
    return (
        _df.groupby(["source","destination"])["distance"]
        .median()
        .reset_index()
        .rename(columns={"distance": "median_distance"})
    )
