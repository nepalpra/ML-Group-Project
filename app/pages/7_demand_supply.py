"""
Page 7 — Demand & Supply Intelligence
=======================================
Surge probability atlas, surge classifier performance, demand patterns,
and best-time-to-travel with historical price profiles.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import load_data, NEIGHBORHOODS, COLORS, apply_chart_theme
from utils.stats import (
    compute_surge_patterns,
    compute_best_time,
    compute_demand_patterns,
    surge_model_available,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"

st.set_page_config(page_title="Demand & Supply", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:8px; padding:12px 16px;
}
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:20px; font-weight:700; color:#0F172A; margin-bottom:2px;">'
    'Demand & Supply Intelligence</div>'
    '<div style="font-size:13px; color:#64748B; margin-bottom:16px;">'
    'Surge probability patterns, demand heatmaps, and best-time-to-travel — '
    'the actionable layer beneath raw price prediction.</div>',
    unsafe_allow_html=True,
)

_PLOT_BG = dict(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
_AXIS    = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

df = load_data()

# ── TL;DR banner ──────────────────────────────────────────────────────────────
with st.spinner("Computing surge patterns and demand aggregates…"):
    _surge_hm, _hourly_sg, _weather_sg, _route_sg = compute_surge_patterns(df)
_peak = _surge_hm.loc[_surge_hm["surge_prob"].idxmax()]
_peak_label = f"{_peak['day_of_week']} {int(_peak['hour']):02d}:00"
_peak_prob  = _peak["surge_prob"]
_high_surge_pct = float((_surge_hm["surge_prob"] > 0.3).mean())
_tldr_parts = [
    f"Peak surge: <b>{_peak_label}</b> (P={_peak_prob:.0%})",
    f"<b>{_high_surge_pct:.0%}</b> of hour×day slots have P(surge)&gt;30%",
    f"Weather surge rates in: <b>{len(_weather_sg)}</b> conditions",
]
st.markdown(
    '<div style="background:#F0F9FF; border:1px solid #BAE6FD; border-radius:8px; '
    'padding:12px 20px; margin-bottom:12px; font-size:14px; color:#0F172A;">'
    + " &nbsp;·&nbsp; ".join(_tldr_parts) + "</div>",
    unsafe_allow_html=True,
)

# Pre-compute (all cached) — reuse values already computed for TL;DR banner
surge_heatmap, hourly_surge, weather_surge, route_surge = (
    _surge_hm, _hourly_sg, _weather_sg, _route_sg
)
vol_heat, vol_hourly, vol_dow = compute_demand_patterns(df)

tabs = st.tabs([
    "Surge Probability Atlas",
    "Surge Model Performance",
    "Demand Patterns",
    "Best Time to Travel",
])

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Surge Probability Atlas
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("When and where does surge happen?")
    st.markdown(
        "This is what a real surge *predictor* needs to answer — not price given surge, "
        "but **P(surge > 1.0)** given time, day, and route. "
        "Derived directly from historical data patterns."
    )

    col_a, col_b = st.columns(2)

    # Hour × day-of-week surge probability heatmap
    with col_a:
        st.markdown("#### P(surge > 1.0) by hour × day of week")
        heat_pivot = surge_heatmap.pivot(
            index="day_of_week", columns="hour", values="surge_prob"
        ).reindex(DOW_ORDER)

        fig_sh = go.Figure(go.Heatmap(
            z=heat_pivot.values.tolist(),
            x=list(range(24)),
            y=DOW_ORDER,
            colorscale="YlOrRd",
            zmin=0, zmax=heat_pivot.values.max(),
            colorbar=dict(title="P(surge)", tickformat=".0%"),
            hovertemplate="<b>%{y} %{x}:00</b><br>P(surge)=%{z:.1%}<extra></extra>",
        ))
        fig_sh.update_layout(
            height=360, margin=dict(t=30, b=50, l=100, r=20),
            xaxis_title="Hour of day",
            xaxis=dict(dtick=2),
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    with col_b:
        st.markdown("#### Hourly average surge probability")
        fig_hs = go.Figure()
        fig_hs.add_trace(go.Bar(
            x=hourly_surge["hour"],
            y=hourly_surge["surge_prob"],
            marker_color="#E91E8C",
            name="P(surge)",
            hovertemplate="<b>%{x}:00</b><br>P(surge)=%{y:.1%}<extra></extra>",
        ))
        fig_hs.add_trace(go.Scatter(
            x=hourly_surge["hour"],
            y=hourly_surge["mean_surge"],
            mode="lines+markers",
            line=dict(color="#0F172A", width=2),
            name="Mean surge multiplier",
            yaxis="y2",
            hovertemplate="<b>%{x}:00</b><br>Mean surge=%{y:.3f}x<extra></extra>",
        ))
        fig_hs.update_layout(
            height=360, margin=dict(t=30, b=50, l=60, r=60),
            xaxis=dict(title="Hour", dtick=2, showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(title="P(surge > 1.0)", tickformat=".0%", showgrid=True, gridcolor="#F1F5F9"),
            yaxis2=dict(title="Mean surge multiplier", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            **_PLOT_BG,
        )
        st.plotly_chart(fig_hs, use_container_width=True)

    # Peak surge hour/day
    peak_row = surge_heatmap.loc[surge_heatmap["surge_prob"].idxmax()]
    st.info(
        f"**Peak surge:** {peak_row['day_of_week']} at "
        f"{int(peak_row['hour']):02d}:00 — "
        f"P(surge) = {peak_row['surge_prob']:.1%}"
    )

    # Weather × surge
    st.markdown("#### Surge probability by weather condition")
    fig_ws = go.Figure(go.Bar(
        x=weather_surge["weather_group"],
        y=weather_surge["surge_prob"],
        marker_color="#F58518",
        error_y=dict(
            type="data",
            array=(weather_surge["mean_surge"] - 1).values.tolist(),
            visible=True, color="#888",
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "P(surge)=%{y:.1%}<extra></extra>"
        ),
    ))
    fig_ws.update_layout(
        height=320, margin=dict(t=30, b=80, l=60, r=20),
        yaxis=dict(title="P(surge > 1.0)", tickformat=".0%", showgrid=True, gridcolor="#F1F5F9"),
        xaxis_title=None, **_PLOT_BG,
    )
    st.plotly_chart(fig_ws, use_container_width=True)

    # Route surge table
    with st.expander("Route-level surge probability (top 20)"):
        st.dataframe(
            route_surge.head(20)
            .rename(columns={"route":"Route","surge_prob":"P(surge)",
                              "mean_surge":"Mean multiplier","count":"N rides"})
            .style.format({"P(surge)":"{:.1%}","Mean multiplier":"{:.3f}"}),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Surge Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Surge classifier — how well can we predict surge in advance?")
    st.markdown(
        "A surge *predictor* uses only pre-trip context: route, time, day, weather. "
        "No surge_multiplier in the feature set — that's the target. "
        "Run `python scripts/train_surge_model.py` to train this model."
    )

    surge_metrics_path = MODEL_DIR / "surge_metrics.json"
    if surge_metrics_path.exists():
        with open(surge_metrics_path) as f:
            sm = json.load(f)

        clf_m = sm.get("classifier", {})
        reg_m = sm.get("regressor",  {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC",       f"{clf_m.get('roc_auc', 0):.4f}",
                  help="Area under ROC curve — 0.5 = random, 1.0 = perfect")
        c2.metric("Avg Precision", f"{clf_m.get('avg_precision', 0):.4f}",
                  help="Average precision score — PR-AUC, better for imbalanced classes")
        c3.metric("Accuracy",      f"{clf_m.get('accuracy', 0):.4f}")
        c4.metric("Surge rate",    f"{clf_m.get('surge_rate_test', 0):.3f}",
                  help="Fraction of rides with surge > 1.0 in test set")

        # Calibration curve
        calib = clf_m.get("calibration", {})
        if calib:
            frac_pos   = calib["fraction_of_positives"]
            mean_pred  = calib["mean_predicted_value"]
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines", line=dict(color="#aaa", dash="dash"),
                name="Perfect calibration",
            ))
            fig_cal.add_trace(go.Scatter(
                x=mean_pred, y=frac_pos,
                mode="lines+markers",
                line=dict(color="#E91E8C", width=2),
                marker=dict(size=8),
                name="Surge classifier",
                hovertemplate="Mean predicted=%{x:.3f}<br>Fraction positive=%{y:.3f}<extra></extra>",
            ))
            fig_cal.update_layout(
                height=360, margin=dict(t=40, b=60, l=60, r=20),
                title="Calibration curve — P(surge) predictions",
                xaxis=dict(title="Mean predicted P(surge)", range=[0,1], **_AXIS),
                yaxis=dict(title="Fraction of rides with surge > 1.0", range=[0,1], **_AXIS),
                **_PLOT_BG,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption(
                "A well-calibrated classifier's curve follows the diagonal. "
                "Points above = over-confident, below = under-confident."
            )

        # Regressor metrics
        if reg_m:
            st.markdown("#### Surge magnitude regressor (given surge > 1.0, predict multiplier)")
            r1, r2, r3 = st.columns(3)
            r1.metric("MAE (surge magnitude)", f"{reg_m.get('mae',0):.4f}")
            r2.metric("R²",                    f"{reg_m.get('r2',0):.4f}")
            r3.metric("Mean actual surge",      f"{reg_m.get('mean_surge_true',0):.4f}x")

        auc = clf_m.get('roc_auc', 0)
        st.info(
            f"**Key finding:** The surge classifier achieves ROC-AUC = {auc:.4f}. "
            + (f"This is substantially above 0.5 (random), confirming that surge "
               "is partially predictable from pre-trip context (time, day, weather, route)."
               if auc > 0.65
               else "ROC-AUC is close to random (0.5), suggesting surge is largely "
                    "unpredictable from pre-trip context alone — it depends on real-time "
                    "supply/demand signals not captured in this dataset.")
        )
    else:
        st.warning(
            "Surge model not trained yet. Run:\n"
            "```\npython scripts/train_surge_model.py\n```\n\n"
            "**Why this matters:** The existing predictor page requires the user to input "
            "`surge_multiplier` — a value they don't know before opening the app. "
            "This classifier estimates surge probability from context alone."
        )

        # Show what the model would use as features
        st.markdown("#### Feature set for surge prediction (no leakage)")
        feature_table = pd.DataFrame([
            {"Feature": "hour",              "Type": "numeric",    "Why": "Rush hours correlate with higher demand"},
            {"Feature": "is_weekend",        "Type": "binary",     "Why": "Weekend patterns differ (nightlife surge)"},
            {"Feature": "temperature",       "Type": "numeric",    "Why": "Cold weather → fewer drivers, more riders"},
            {"Feature": "precipProbability", "Type": "numeric",    "Why": "Rain → spike in ride requests"},
            {"Feature": "windSpeed",         "Type": "numeric",    "Why": "Proxy for adverse conditions"},
            {"Feature": "source",            "Type": "categorical","Why": "High-traffic areas surge more"},
            {"Feature": "destination",       "Type": "categorical","Why": "Destination demand patterns"},
            {"Feature": "cab_type",          "Type": "categorical","Why": "Platform-specific surge policies"},
            {"Feature": "surge_multiplier",  "Type": "TARGET",     "Why": "What we're predicting — NOT a feature"},
        ])
        st.dataframe(feature_table.style.apply(
            lambda row: ["background-color: #ffd7d7" if row["Type"] == "TARGET" else "" for _ in row],
            axis=1,
        ), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Demand Patterns
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Where and when are people riding?")
    st.markdown(
        "Understanding demand volume helps explain surge: high-demand windows "
        "(rush hours, Friday evenings) drive surge when driver supply is constrained."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Ride volume by hour (overall)")
        fig_vh = go.Figure(go.Bar(
            x=vol_hourly["hour"],
            y=vol_hourly["count"],
            marker_color="#4C78A8",
            hovertemplate="<b>%{x}:00</b><br>Rides=%{y:,}<extra></extra>",
        ))
        fig_vh.update_layout(
            height=320, margin=dict(t=30, b=50, l=60, r=20),
            xaxis=dict(title="Hour", dtick=2, **_AXIS),
            yaxis=dict(title="Ride count", **_AXIS),
            **_PLOT_BG,
        )
        st.plotly_chart(fig_vh, use_container_width=True)

    with col_b:
        st.markdown("#### Ride volume: day of week × hour")
        dow_pivot = vol_dow.pivot(index="day_of_week", columns="hour",
                                   values="count").reindex(DOW_ORDER).fillna(0)
        fig_vdow = go.Figure(go.Heatmap(
            z=dow_pivot.values.tolist(),
            x=list(range(24)),
            y=DOW_ORDER,
            colorscale="Blues",
            colorbar=dict(title="Rides"),
            hovertemplate="<b>%{y} %{x}:00</b><br>Rides=%{z:,.0f}<extra></extra>",
        ))
        fig_vdow.update_layout(
            height=320, margin=dict(t=30, b=50, l=100, r=20),
            xaxis=dict(title="Hour", dtick=2),
        )
        st.plotly_chart(fig_vdow, use_container_width=True)

    # Volume by source neighborhood
    st.markdown("#### Ride volume by pickup neighborhood")
    nbhd_vol = (
        df.groupby("source")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig_nbhd = go.Figure(go.Bar(
        y=nbhd_vol["source"], x=nbhd_vol["count"],
        orientation="h", marker_color="#4C78A8",
        hovertemplate="<b>%{y}</b><br>Rides=%{x:,}<extra></extra>",
    ))
    fig_nbhd.update_layout(
        height=360, margin=dict(t=30, b=40, l=190, r=20),
        xaxis_title="Total rides", **_PLOT_BG, xaxis=_AXIS,
    )
    st.plotly_chart(fig_nbhd, use_container_width=True)

    # Volume by hour × source (heatmap)
    with st.expander("Pickup volume heatmap: hour × neighborhood"):
        vol_pivot = vol_heat.pivot(index="source", columns="hour",
                                    values="count").fillna(0)
        fig_vpvt = go.Figure(go.Heatmap(
            z=vol_pivot.values.tolist(),
            x=list(range(24)),
            y=vol_pivot.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="Rides"),
            hovertemplate="<b>%{y} %{x}:00</b><br>Rides=%{z:,.0f}<extra></extra>",
        ))
        fig_vpvt.update_layout(
            height=380, margin=dict(t=30, b=50, l=190, r=20),
            xaxis=dict(title="Hour", dtick=2),
        )
        st.plotly_chart(fig_vpvt, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Best Time to Travel
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("When should you book to minimise cost?")
    st.markdown(
        "For a given route and platform, this shows the **historical median price ± IQR** "
        "by hour of day — the closest thing to a 'cheapest time to travel' recommendation "
        "without real-time data. The IQR band reflects price volatility, not just the median."
    )

    c_in, c_out = st.columns([1, 2])
    with c_in:
        src_bt   = st.selectbox("From", sorted(NEIGHBORHOODS), key="bt_src")
        dst_bt   = st.selectbox("To",   sorted([n for n in NEIGHBORHOODS if n != src_bt]),
                                key="bt_dst")
        plat_bt  = st.radio("Platform", ["Uber", "Lyft"], horizontal=True, key="bt_plat")
        show_all = st.checkbox("Overlay both platforms", value=False)

    with c_out:
        fig_bt = go.Figure()

        platforms_to_show = ["Uber", "Lyft"] if show_all else [plat_bt]
        for plat in platforms_to_show:
            hourly_bt = compute_best_time(df, src_bt, dst_bt, plat)
            if hourly_bt.empty:
                st.warning(f"No rides found for {plat} on this route.")
                continue

            color = COLORS.get(plat, "#4C78A8")
            # IQR band
            fig_bt.add_trace(go.Scatter(
                x=pd.concat([hourly_bt["hour"], hourly_bt["hour"].iloc[::-1]]),
                y=pd.concat([hourly_bt["q75"], hourly_bt["q25"].iloc[::-1]]),
                fill="toself", fillcolor=color,
                line=dict(color="rgba(0,0,0,0)"),
                opacity=0.15, showlegend=False, hoverinfo="skip",
                name=f"{plat} IQR",
            ))
            # Median line
            fig_bt.add_trace(go.Scatter(
                x=hourly_bt["hour"],
                y=hourly_bt["median"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7),
                name=f"{plat} median",
                hovertemplate=(
                    f"<b>{plat} %{{x}}:00</b><br>"
                    "Median=$%{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig_bt.update_layout(
            height=400, margin=dict(t=40, b=60, l=60, r=20),
            title=f"{src_bt} → {dst_bt}",
            xaxis=dict(title="Hour of day", dtick=2, **_AXIS),
            yaxis=dict(title="Price ($)", **_AXIS),
            legend=dict(orientation="h", y=-0.2),
            **_PLOT_BG,
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        st.caption("Shaded band = IQR (25th–75th percentile price range). Line = median price.")

    # Best hour call-out
    hourly_for_rec = compute_best_time(df, src_bt, dst_bt, plat_bt)
    if not hourly_for_rec.empty:
        best_hour_row  = hourly_for_rec.loc[hourly_for_rec["median"].idxmin()]
        worst_hour_row = hourly_for_rec.loc[hourly_for_rec["median"].idxmax()]
        saving         = float(worst_hour_row["median"] - best_hour_row["median"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Cheapest hour",
                  f"{int(best_hour_row['hour']):02d}:00",
                  delta=f"${best_hour_row['median']:.2f} median")
        c2.metric("Most expensive hour",
                  f"{int(worst_hour_row['hour']):02d}:00",
                  delta=f"${worst_hour_row['median']:.2f} median",
                  delta_color="inverse")
        c3.metric("Potential saving",
                  f"${saving:.2f}",
                  help="Median price at worst hour minus median price at best hour")

        # Savings over 30 trips
        savings_month = saving * 30
        st.success(
            f"**For a commuter on this route:** Booking at "
            f"{int(best_hour_row['hour']):02d}:00 instead of "
            f"{int(worst_hour_row['hour']):02d}:00 saves ${saving:.2f} per trip. "
            f"Over 30 trips/month: **${savings_month:.0f} saved**."
        )

    # Platform comparison for this route
    st.markdown("#### Platform price comparison — all hours")
    uber_bt = compute_best_time(df, src_bt, dst_bt, "Uber")
    lyft_bt = compute_best_time(df, src_bt, dst_bt, "Lyft")
    if not uber_bt.empty and not lyft_bt.empty:
        merged_bt = uber_bt.merge(lyft_bt, on="hour", suffixes=("_uber", "_lyft"))
        merged_bt["diff"] = merged_bt["median_lyft"] - merged_bt["median_uber"]
        cheaper_hours_uber = int((merged_bt["diff"] > 0).sum())
        cheaper_hours_lyft = int((merged_bt["diff"] < 0).sum())
        st.info(
            f"On **{src_bt} → {dst_bt}**: Uber is cheaper in "
            f"{cheaper_hours_uber}/24 hours, Lyft is cheaper in "
            f"{cheaper_hours_lyft}/24 hours."
        )
