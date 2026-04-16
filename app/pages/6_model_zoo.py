"""
Page 6 — Model Zoo
===================
Extended model comparison: quantile regression for prediction intervals,
MLP baseline, time-ordered validation, and per-platform performance breakdown.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import load_data, NEIGHBORHOODS, SERVICE_MAP
from utils.model import (
    models_available, load_models, linear_models_available, load_linear_models,
    NUM_FEATURES, CAT_FEATURES,
)
from utils.stats import advanced_models_available, time_split_available

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"

st.set_page_config(page_title="Model Zoo", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:12px 16px;
}
[data-testid="stAlert"][data-type="info"] {
    background:#F0F9FF; border-left:4px solid #0284C7; border-radius:0 6px 6px 0;
}
details summary p { font-size:13px !important; color:#64748B !important; }
hr { border-color:#E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:22px; font-weight:700; color:#0F172A; margin-bottom:2px;">Model Zoo</div>'
    '<div style="font-size:13px; color:#64748B; margin-bottom:8px;">'
    'Quantile prediction intervals, MLP baseline, Nov→Dec temporal validation, per-tier breakdown.</div>',
    unsafe_allow_html=True,
)

# ── TL;DR banner ──────────────────────────────────────────────────────────────
_base_path = Path(__file__).resolve().parents[2] / "model" / "model_metrics.json"
_tsplit_path = Path(__file__).resolve().parents[2] / "model" / "time_split_metrics.json"
if _base_path.exists():
    import json as _json
    _bm = _json.loads(_base_path.read_text())
    _lgb_mae = _bm.get("LightGBM", {}).get("mae", None)
    _lgb_r2  = _bm.get("LightGBM", {}).get("r2", None)
    _tldr_parts = []
    if _lgb_mae: _tldr_parts.append(f"LightGBM MAE: <b>${_lgb_mae:.2f}</b>")
    if _lgb_r2:  _tldr_parts.append(f"R²: <b>{_lgb_r2:.4f}</b>")
    if _tsplit_path.exists():
        _ts = _json.loads(_tsplit_path.read_text())
        _nov_dec = _ts.get("lgb_nov_trained_on_dec", {}).get("mae", None)
        if _nov_dec: _tldr_parts.append(f"Nov→Dec MAE: <b>${_nov_dec:.2f}</b>")
    if _tldr_parts:
        st.markdown(
            '<div style="background:#F0F9FF; border:1px solid #BAE6FD; border-radius:8px; '
            'padding:12px 20px; margin-bottom:12px; font-size:14px; color:#0F172A;">'
            + " &nbsp;·&nbsp; ".join(_tldr_parts) + "</div>",
            unsafe_allow_html=True,
        )

_PLOT_BG = dict(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
_AXIS    = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

if not models_available():
    st.error("Base model artifacts not found. Run `02_modeling.ipynb` first.")
    st.stop()

import joblib
lgb, xgb, prep, feat_names = load_models()

# ── Load all available metrics ────────────────────────────────────────────────
@st.cache_data
def _load_metrics():
    results = {}
    base_path = MODEL_DIR / "model_metrics.json"
    if base_path.exists():
        with open(base_path) as f:
            results.update(json.load(f))
    adv_path = MODEL_DIR / "advanced_metrics.json"
    if adv_path.exists():
        with open(adv_path) as f:
            results.update(json.load(f))
    return results

@st.cache_data
def _load_time_split():
    p = MODEL_DIR / "time_split_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

all_metrics   = _load_metrics()
time_metrics  = _load_time_split()

tabs = st.tabs([
    "All Models Overview",
    "Prediction Intervals",
    "Time-Ordered Validation",
    "Per-Platform Performance",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — All Models Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("All trained models — test set metrics")
    st.markdown(
        "Comparison of all models from `02_modeling.ipynb` plus new architectures "
        "from `train_advanced_models.py`. Each metric is on the same 15% held-out test set."
    )

    MODEL_LABELS = {
        "LightGBM":    "LightGBM (default)",
        "XGBoost":     "XGBoost (default)",
        "Ridge":       "Ridge (L2)",
        "Linear":      "OLS Linear",
        "lgb_q50":     "LGB Quantile (q50)",
        "lgb_q10":     "LGB Quantile (q10)",
        "lgb_q90":     "LGB Quantile (q90)",
        "mlp":         "MLP (128-64)",
        "lgb_tuned":   "LightGBM (Optuna-tuned)",
        "xgb_tuned":   "XGBoost (Optuna-tuned)",
    }
    MODEL_COLORS = {
        "LightGBM (default)":       "#4C78A8",
        "XGBoost (default)":        "#F58518",
        "Ridge (L2)":               "#72B7B2",
        "OLS Linear":               "#54A24B",
        "LGB Quantile (q50)":       "#B279A2",
        "LGB Quantile (q10)":       "#D8B5D8",
        "LGB Quantile (q90)":       "#9E4F9C",
        "MLP (128-64)":             "#EECA3B",
        "LightGBM (Optuna-tuned)":  "#1a6ab0",
        "XGBoost (Optuna-tuned)":   "#c45e00",
    }

    rows = []
    for key, label in MODEL_LABELS.items():
        if key in all_metrics and isinstance(all_metrics[key], dict):
            m = all_metrics[key]
            rows.append(dict(
                model=label,
                mae  =m.get("mae",  np.nan),
                rmse =m.get("rmse", np.nan),
                r2   =m.get("r2",   np.nan),
            ))
    if not rows:
        st.warning("No model metrics found. Run `02_modeling.ipynb` and optionally "
                   "`scripts/train_advanced_models.py` and `scripts/train_tuned_models.py`.")
    else:
        metrics_df = pd.DataFrame(rows).sort_values("mae")

        c1, c2 = st.columns(2)
        with c1:
            fig_mae = go.Figure(go.Bar(
                y=metrics_df["model"],
                x=metrics_df["mae"],
                orientation="h",
                marker_color=[MODEL_COLORS.get(m, "#888") for m in metrics_df["model"]],
                hovertemplate="<b>%{y}</b><br>MAE=$%{x:.4f}<extra></extra>",
            ))
            fig_mae.update_layout(
                height=420, title="Mean Absolute Error ($) — lower is better",
                margin=dict(t=40, b=40, l=170, r=20),
                xaxis_title="MAE ($)", **_PLOT_BG, xaxis=_AXIS,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with c2:
            fig_r2 = go.Figure(go.Bar(
                y=metrics_df["model"],
                x=metrics_df["r2"],
                orientation="h",
                marker_color=[MODEL_COLORS.get(m, "#888") for m in metrics_df["model"]],
                hovertemplate="<b>%{y}</b><br>R²=%{x:.5f}<extra></extra>",
            ))
            fig_r2.update_layout(
                height=420, title="R² — higher is better",
                margin=dict(t=40, b=40, l=170, r=20),
                xaxis_title="R²", xaxis=dict(range=[0.8, 1.0], showgrid=True, gridcolor="#F1F5F9"),
                **_PLOT_BG,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        # Styled table
        st.dataframe(
            metrics_df.rename(columns={"model":"Model","mae":"MAE ($)","rmse":"RMSE ($)","r2":"R²"})
            .style
            .highlight_min(subset=["MAE ($)","RMSE ($)"], color="#c6efce")
            .highlight_max(subset=["R²"], color="#c6efce")
            .format({"MAE ($)":"${:.4f}","RMSE ($)":"${:.4f}","R²":"{:.5f}"}),
            use_container_width=True,
        )

        # Coverage stat if available
        if "quantile_interval_coverage_80pct" in all_metrics:
            cov = all_metrics["quantile_interval_coverage_80pct"]
            st.metric(
                "80% quantile interval empirical coverage",
                f"{cov:.3f}",
                delta=f"{cov-0.80:+.3f} vs target 0.80",
                delta_color="normal" if abs(cov - 0.80) < 0.05 else "inverse",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Prediction Intervals
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Quantile regression — honest prediction intervals")
    st.markdown(
        "Instead of a single point prediction, quantile LGB estimates the full "
        "conditional distribution: q10 (lower bound), q50 (median), q90 (upper bound). "
        "This gives an **80% prediction interval** — 80% of actual prices should fall "
        "within it if the model is well-calibrated."
    )

    q_available = advanced_models_available()
    if q_available:
        @st.cache_resource
        def _load_quantile_models():
            import joblib
            return {
                "q10": joblib.load(MODEL_DIR / "lgb_q10.pkl"),
                "q50": joblib.load(MODEL_DIR / "lgb_q50.pkl"),
                "q90": joblib.load(MODEL_DIR / "lgb_q90.pkl"),
            }
        q_models = _load_quantile_models()

        df = load_data()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            src  = st.selectbox("From", sorted(NEIGHBORHOODS), key="qr_src")
            dst  = st.selectbox("To",   sorted([n for n in NEIGHBORHOODS if n != src]), key="qr_dst")
            hour = st.slider("Hour", 0, 23, 8, key="qr_hour")
            platform = st.radio("Platform", ["Uber", "Lyft"], horizontal=True, key="qr_plat")
            surge = st.slider("Surge multiplier", 1.0, 3.0, 1.0, step=0.25, key="qr_surge")

        with col_b:
            dist_series = df.loc[
                (df.source == src) & (df.destination == dst), "distance"
            ]
            dist = float(dist_series.median()) if not dist_series.empty else 2.0
            is_wknd = 0

            rows_pred = []
            tiers = [k for k, v in SERVICE_MAP.items() if v == platform]
            for tier in tiers:
                row = pd.DataFrame([dict(
                    distance=dist, surge_multiplier=surge, hour=hour,
                    is_weekend=is_wknd, temperature=45, precipProbability=0.1,
                    windSpeed=10, name=tier, cab_type=platform,
                    source=src, destination=dst,
                )])
                X_enc = prep.transform(row[NUM_FEATURES + CAT_FEATURES])
                q10_p = float(q_models["q10"].predict(X_enc)[0])
                q50_p = float(q_models["q50"].predict(X_enc)[0])
                q90_p = float(q_models["q90"].predict(X_enc)[0])
                rows_pred.append(dict(tier=tier, q10=q10_p, q50=q50_p, q90=q90_p,
                                      interval=q90_p - q10_p))

            pred_df = pd.DataFrame(rows_pred).sort_values("q50")

            fig_qi = go.Figure()
            for _, row in pred_df.iterrows():
                fig_qi.add_trace(go.Scatter(
                    x=[row.q10, row.q90], y=[row.tier, row.tier],
                    mode="lines", line=dict(color="#B279A2", width=10),
                    opacity=0.3, showlegend=False, hoverinfo="skip",
                ))
                fig_qi.add_trace(go.Scatter(
                    x=[row.q50], y=[row.tier],
                    mode="markers",
                    marker=dict(color="#7B2D8B", size=12, symbol="diamond"),
                    name=row.tier,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row.tier}</b><br>"
                        f"q10=${row.q10:.2f}  q50=${row.q50:.2f}  q90=${row.q90:.2f}<br>"
                        f"80% interval width: ${row.interval:.2f}"
                        "<extra></extra>"
                    ),
                ))
            fig_qi.update_layout(
                height=360, margin=dict(t=30, b=40, l=120, r=20),
                title=f"{platform} — {src} → {dst} (hour={hour:02d}:00, surge={surge}x)",
                xaxis_title="Predicted price ($)",
                **_PLOT_BG, xaxis=_AXIS,
            )
            st.plotly_chart(fig_qi, use_container_width=True)
            st.caption("◆ = q50 median prediction  |  colored band = 80% prediction interval (q10 → q90)")

            # Coverage on historical data
            hist = df.loc[
                (df.source == src) & (df.destination == dst) & (df.cab_type == platform)
            ]
            if not hist.empty:
                X_hist = prep.transform(hist[NUM_FEATURES + CAT_FEATURES])
                lo = q_models["q10"].predict(X_hist)
                hi = q_models["q90"].predict(X_hist)
                coverage = float(np.mean((hist["price"].values >= lo) & (hist["price"].values <= hi)))
                st.metric(
                    f"Actual 80% interval coverage — {platform} {src} → {dst}",
                    f"{coverage:.3f}",
                    delta=f"{coverage - 0.80:+.3f} vs target 0.80",
                    delta_color="normal" if abs(coverage - 0.80) < 0.07 else "inverse",
                )
    else:
        st.warning(
            "Quantile models not found. Run:\n"
            "```\npython scripts/train_advanced_models.py\n```\n"
            "Then reload this page."
        )
        # Show conceptual diagram
        fig_concept = go.Figure()
        tiers_c = ["UberPool","UberX","UberXL","Black","Black SUV"]
        medians  = [8, 14, 18, 30, 45]
        widths   = [3,  4,   5,   8,  12]
        for t, m, w in zip(tiers_c, medians, widths):
            fig_concept.add_trace(go.Scatter(
                x=[m-w, m+w], y=[t, t], mode="lines",
                line=dict(color="#B279A2", width=8), opacity=0.3,
                showlegend=False, hoverinfo="skip",
            ))
            fig_concept.add_trace(go.Scatter(
                x=[m], y=[t], mode="markers",
                marker=dict(color="#7B2D8B", size=10, symbol="diamond"),
                showlegend=False,
                hovertemplate=f"<b>{t}</b><br>Median≈${m}<br>80% PI≈±${w}<extra></extra>",
            ))
        fig_concept.update_layout(
            height=300, title="Conceptual: prediction interval by tier (illustrative)",
            xaxis_title="Price ($)", **_PLOT_BG, xaxis=_AXIS,
            margin=dict(t=40, b=40, l=100, r=20),
        )
        st.plotly_chart(fig_concept, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Time-Ordered Validation
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Temporal generalization: train on November → test on December")
    st.markdown(
        "The standard 85/15 random split **leaks future data into training** "
        "when the dataset has a time dimension. Temporal validation (train=Nov, test=Dec) "
        "tests whether the model generalizes to unseen time periods — a stricter and "
        "more realistic evaluation."
    )

    if time_metrics:
        rows_t = []
        label_map = {
            "lgb_default_on_dec"       : ("LGB default (Nov+Dec train → Dec test)", "baseline"),
            "lgb_nov_trained_on_nov"   : ("LGB Nov-only (eval on Nov validation)", "nov_val"),
            "lgb_nov_trained_on_dec"   : ("LGB Nov-only (eval on Dec — temporal)", "temporal"),
            "lgb_default_dec_uber"     : ("LGB default — Uber rides only (Dec)", "platform"),
            "lgb_default_dec_lyft"     : ("LGB default — Lyft rides only (Dec)", "platform"),
        }
        for key, (label, group) in label_map.items():
            if key in time_metrics and isinstance(time_metrics[key], dict):
                m = time_metrics[key]
                rows_t.append(dict(label=label, group=group, **m))

        if rows_t:
            t_df = pd.DataFrame(rows_t)
            GROUP_COLORS = {
                "baseline": "#4C78A8",
                "nov_val":  "#54A24B",
                "temporal": "#E45756",
                "platform": "#F58518",
            }

            fig_t = make_subplots(rows=1, cols=3,
                                  subplot_titles=["MAE ($)", "RMSE ($)", "R²"])
            for metric, col_idx in [("mae", 1), ("rmse", 2), ("r2", 3)]:
                if metric in t_df.columns:
                    fig_t.add_trace(go.Bar(
                        y=t_df["label"],
                        x=t_df[metric],
                        orientation="h",
                        marker_color=[GROUP_COLORS.get(g, "#888") for g in t_df["group"]],
                        showlegend=False,
                        hovertemplate="<b>%{y}</b><br>Value=%{x:.4f}<extra></extra>",
                    ), row=1, col=col_idx)

            fig_t.update_layout(
                height=360, margin=dict(t=50, b=40, l=330, r=20),
                **_PLOT_BG,
            )
            fig_t.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_t, use_container_width=True)

            # Legend
            for grp, color in GROUP_COLORS.items():
                if any(t_df["group"] == grp):
                    st.markdown(f'<span style="color:{color}">■</span> {grp.replace("_"," ").title()}',
                                unsafe_allow_html=True)

            # Key comparison
            if "lgb_default_on_dec" in time_metrics and "lgb_nov_trained_on_dec" in time_metrics:
                m_rand     = time_metrics["lgb_default_on_dec"]
                m_temporal = time_metrics["lgb_nov_trained_on_dec"]
                mae_drop   = m_temporal["mae"] - m_rand["mae"]
                r2_drop    = m_rand["r2"] - m_temporal["r2"]
                st.info(
                    f"**Key finding:** Training on Nov only (temporal split) vs random split: "
                    f"MAE increases by ${mae_drop:+.4f} and R² drops by {r2_drop:.4f}. "
                    + ("The model generalizes well across months — temporal leakage in "
                       "the random split is minimal."
                       if mae_drop < 0.1
                       else "This degradation reveals that the random split inflates "
                            "performance by leaking Dec patterns into training.")
                )
    else:
        st.warning(
            "Time-split metrics not found. Run:\n"
            "```\npython scripts/train_advanced_models.py\n```"
        )
        # Show conceptual explanation
        st.markdown("""
        #### Why random split can be misleading

        With a random 85/15 split on time-ordered data:
        - Training set contains rides from **both November and December**
        - Test set also contains rides from **both months**
        - The model sees December patterns during training → overly optimistic R²

        With a temporal split (train=Nov, test=Dec):
        - Model must generalize from Nov to Dec without seeing Dec during training
        - Any R² degradation reveals temporal leakage in the random split
        - More realistic estimate of production performance
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Per-Platform Performance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Does the model perform equally on Uber and Lyft rides?")
    st.markdown(
        "Aggregate R² = 0.97 hides whether accuracy differs by platform, "
        "tier, or route. A model that predicts Lyft perfectly but fails on "
        "premium Uber tiers still reports high overall R²."
    )

    df = load_data()

    @st.cache_data(show_spinner="Computing per-segment model performance…")
    def _per_segment_metrics(_df):
        records = []
        for platform in ["Uber", "Lyft"]:
            for tier, grp in _df.groupby("name"):
                if grp["cab_type"].iloc[0] != platform:
                    continue
                if len(grp) < 50:
                    continue
                X_enc = prep.transform(grp[NUM_FEATURES + CAT_FEATURES])
                y_true = grp["price"].values
                y_pred = lgb.predict(X_enc)
                mae  = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
                r2   = float(1 - np.sum((y_true - y_pred)**2) /
                             np.sum((y_true - y_true.mean())**2))
                records.append(dict(
                    platform=platform, tier=tier, n=len(grp),
                    mae=mae, rmse=rmse, r2=r2,
                ))
        return pd.DataFrame(records).sort_values(["platform","mae"])

    seg_df = _per_segment_metrics(df)

    col_a, col_b = st.columns(2)
    for col, platform in [(col_a, "Uber"), (col_b, "Lyft")]:
        sub = seg_df[seg_df.platform == platform].sort_values("mae", ascending=False)
        color = "#0F172A" if platform == "Uber" else "#E91E8C"
        with col:
            st.markdown(f"**{platform}**")
            fig_seg = go.Figure(go.Bar(
                y=sub["tier"],
                x=sub["mae"],
                orientation="h",
                marker_color=color,
                hovertemplate="<b>%{y}</b><br>MAE=$%{x:.3f}<extra></extra>",
            ))
            fig_seg.update_layout(
                height=320, margin=dict(t=30, b=40, l=110, r=20),
                xaxis_title="MAE ($)", **_PLOT_BG, xaxis=_AXIS,
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    st.dataframe(
        seg_df[["platform","tier","n","mae","rmse","r2"]]
        .rename(columns={"platform":"Platform","tier":"Tier","n":"N",
                         "mae":"MAE ($)","rmse":"RMSE ($)","r2":"R²"})
        .style
        .highlight_min(subset=["MAE ($)"], color="#c6efce")
        .highlight_max(subset=["R²"],       color="#c6efce")
        .highlight_max(subset=["MAE ($)"], color="#ffd7d7")
        .format({"MAE ($)":"${:.3f}","RMSE ($)":"${:.3f}","R²":"{:.4f}"}),
        use_container_width=True,
    )

    # Uber vs Lyft aggregate
    plat_agg = seg_df.groupby("platform")[["mae","rmse","r2"]].mean().reset_index()
    for _, row in plat_agg.iterrows():
        st.metric(
            f"{row.platform} — avg MAE (across tiers)",
            f"${row.mae:.4f}",
            delta=None,
        )

    worst = seg_df.loc[seg_df["mae"].idxmax()]
    best  = seg_df.loc[seg_df["mae"].idxmin()]
    st.info(
        f"**Key finding:** Worst prediction accuracy is for "
        f"**{worst.platform} {worst.tier}** (MAE=${worst.mae:.3f}). "
        f"Best is **{best.platform} {best.tier}** (MAE=${best.mae:.3f}). "
        "Premium tiers typically have higher absolute MAE due to wider price ranges, "
        "even if their relative error is similar."
    )
