import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import load_data, get_route_distance, NEIGHBORHOODS, SERVICE_MAP, COLORS, apply_chart_theme
from utils.model import (
    models_available, load_models,
    linear_models_available, load_linear_models,
    predict_tiers, build_input_row,
    NUM_FEATURES, CAT_FEATURES, MODEL_DIR,
)

st.set_page_config(page_title="Price Predictor", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:8px; padding:12px 16px;
}
[data-testid="stAlert"][data-type="info"] {
    background:#F0F9FF; border-left:4px solid #0284C7;
    border-radius:0 6px 6px 0;
}
[data-testid="stAlert"][data-type="success"] {
    background:#F0FDF4; border-left:4px solid #16A34A;
    border-radius:0 6px 6px 0;
}
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

if not models_available():
    st.error("Model files not found. Run `02_modeling.ipynb` and save artifacts first.")
    st.stop()

df               = load_data()
lgb, xgb, prep, feat_names = load_models()
route_dist_df    = get_route_distance(df)

_lin_available = linear_models_available()
if _lin_available:
    _lr, _ridge = load_linear_models()

TIER_ORDER_LYF = ["Shared","Lyft","Lyft XL","Lux","Lux Black","Lux Black XL"]
TIER_ORDER_UBR = ["UberPool","UberX","UberXL","Black","Black SUV","WAV"]
ALL_TIERS      = TIER_ORDER_LYF + TIER_ORDER_UBR

# ── Sidebar — all inputs ─────────────────────────────────��─────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-weight:700; font-size:14px; color:#0F172A;
                border-bottom:1px solid #E2E8F0; padding-bottom:10px; margin-bottom:14px;">
        Ride Conditions
    </div>
    """, unsafe_allow_html=True)

    src = st.selectbox("From", sorted(NEIGHBORHOODS), key="pred_src")
    dst = st.selectbox("To",   sorted([n for n in NEIGHBORHOODS if n != src]), key="pred_dst")

    st.markdown("<div style='margin-top:8px; font-size:11px; font-weight:600; "
                "color:#94A3B8; letter-spacing:0.08em;'>TIMING</div>", unsafe_allow_html=True)
    hour       = st.slider("Hour of day", 0, 23, 8, format="%d:00")
    is_weekend = st.toggle("Weekend", value=False)
    surge      = st.slider("Surge multiplier", 1.0, 3.0, 1.0, step=0.25)

    st.markdown("<div style='margin-top:8px; font-size:11px; font-weight:600; "
                "color:#94A3B8; letter-spacing:0.08em;'>WEATHER</div>", unsafe_allow_html=True)
    temp   = st.slider("Temperature (°F)",          20, 70, 45)
    precip = st.slider("Precipitation probability", 0.0, 1.0, 0.1, step=0.05)
    wind   = st.slider("Wind speed (mph)",           0, 40, 10)

    st.divider()
    wait_1h = st.checkbox("Show +1 hour comparison", value=True)
    _model_opts = ["LightGBM", "XGBoost"] + (["Linear", "Ridge"] if _lin_available else [])
    model_choice = st.radio("Model", _model_opts, horizontal=True)

# ── Main area — full-width results ─────────────────────────────────────────────
dist_row = route_dist_df[
    (route_dist_df["source"] == src) & (route_dist_df["destination"] == dst)
]
route_distance = float(dist_row["median_distance"].iloc[0]) if len(dist_row) > 0 else 1.5

if model_choice == "LightGBM":
    chosen_model = lgb
elif model_choice == "XGBoost":
    chosen_model = xgb
elif model_choice == "Linear":
    chosen_model = _lr
else:
    chosen_model = _ridge


def get_predictions(hour_val):
    rows = []
    for name, cab in SERVICE_MAP.items():
        rows.append(dict(
            distance=route_distance, surge_multiplier=surge,
            hour=hour_val, is_weekend=int(is_weekend),
            temperature=temp, precipProbability=precip, windSpeed=wind,
            name=name, cab_type=cab, source=src, destination=dst,
        ))
    df_p = pd.DataFrame(rows)
    X    = prep.transform(df_p[NUM_FEATURES + CAT_FEATURES])
    df_p["predicted_price"] = chosen_model.predict(X)
    return df_p


pred_now  = get_predictions(hour).set_index("name")
pred_next = get_predictions((hour + 1) % 24).set_index("name")

# Store for Explainer page
st.session_state.update({
    "last_src": src, "last_dst": dst, "last_hour": hour,
    "last_weekend": is_weekend, "last_temp": temp, "last_precip": precip,
    "last_wind": wind, "last_surge": surge,
    "last_distance": route_distance, "last_model": model_choice,
})

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-size:20px; font-weight:700; color:#0F172A; margin-bottom:2px;">'
    f'What-If Price Engine</div>'
    f'<div style="font-size:13px; color:#64748B; margin-bottom:16px;">'
    f'{src} → {dst} &nbsp;·&nbsp; {hour:02d}:00 &nbsp;·&nbsp; '
    f'{"Weekend" if is_weekend else "Weekday"} &nbsp;·&nbsp; '
    f'{surge}× surge &nbsp;·&nbsp; {temp}°F &nbsp;·&nbsp; {precip:.0%} precip &nbsp;·&nbsp; '
    f'{route_distance:.2f} mi &nbsp;·&nbsp; {model_choice}</div>',
    unsafe_allow_html=True,
)

# ── Winner card ────────────────────────────────────────────────────────────────
cheapest       = pred_now["predicted_price"].idxmin()
cheapest_price = float(pred_now["predicted_price"].min())
cheapest_cab   = SERVICE_MAP.get(cheapest, "")

if wait_1h:
    best_wait    = (pred_next["predicted_price"] - pred_now["predicted_price"]).idxmin()
    best_saving  = float((pred_now["predicted_price"] - pred_next["predicted_price"]).max())
    wait_price   = float(pred_next.loc[best_wait, "predicted_price"])
    wait_cab     = SERVICE_MAP.get(best_wait, "")
    wait_color   = COLORS.get(wait_cab, "#64748B")

    if best_saving > 0:
        right_block = (
            f'<div style="text-align:right;">'
            f'<div style="font-size:11px; font-weight:600; color:#D97706;'
            f' text-transform:uppercase; letter-spacing:0.05em;">Best if you wait 1 hour</div>'
            f'<div style="font-size:22px; font-weight:700; color:{wait_color}; margin-top:4px;">'
            f'{best_wait} &mdash; ${wait_price:.2f}</div>'
            f'<div style="font-size:13px; color:#059669; margin-top:2px; font-weight:600;">'
            f'&#9660; ${best_saving:.2f} cheaper than now</div>'
            f'</div>'
        )
    else:
        right_block = (
            '<div style="text-align:right;">'
            '<div style="font-size:11px; font-weight:600; color:#DC2626;'
            ' text-transform:uppercase; letter-spacing:0.05em;">Prices rising next hour</div>'
            '<div style="font-size:14px; color:#64748B; margin-top:4px;">'
            'Book now &mdash; waiting costs more</div>'
            '</div>'
        )
else:
    right_block = ""

left_color = COLORS.get(cheapest_cab, "#0F172A")
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 1px solid #86EFAC;
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div>
        <div style="font-size:11px; font-weight:600; color:#16A34A;
                    text-transform:uppercase; letter-spacing:0.05em;">
            Cheapest right now
        </div>
        <div style="font-size:28px; font-weight:700; color:{left_color}; margin-top:4px;">
            {cheapest} &mdash; ${cheapest_price:.2f}
        </div>
        <div style="font-size:13px; color:#64748B; margin-top:2px;">
            {src} → {dst} &nbsp;·&nbsp; {hour:02d}:00
        </div>
    </div>
    {right_block}
</div>
""", unsafe_allow_html=True)

# ── Full-width comparison chart ────────────────────────────────────────────────
fig = go.Figure()
for tier in ALL_TIERS:
    if tier not in pred_now.index:
        continue
    cab   = SERVICE_MAP[tier]
    price = pred_now.loc[tier, "predicted_price"]
    color = COLORS[cab]
    fig.add_trace(go.Bar(
        x=[price], y=[tier], orientation="h",
        marker_color=color, showlegend=False,
        text=[f"${price:.2f}"], textposition="outside",
        hovertemplate=f"<b>{tier}</b> ({cab})<br>Predicted: ${price:.2f}<extra></extra>",
    ))

apply_chart_theme(fig, height=420, margin=dict(t=20, b=20, l=120, r=90))
fig.update_layout(
    xaxis_title="Predicted Price ($)",
    yaxis=dict(categoryorder="total ascending"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Model accuracy context ─────────────────────────────────────────────────────
_metrics_path = MODEL_DIR / "model_metrics.json"
_mae = None
if _metrics_path.exists():
    _met = json.loads(_metrics_path.read_text())
    _mae = float(_met[model_choice]["mae"]) if model_choice in _met else None

_cheapest_hist = df[
    (df["source"] == src) & (df["destination"] == dst) &
    (df["name"] == cheapest)
]["price"]

_acc1, _acc2, _acc3 = st.columns(3)
if _mae is not None:
    _acc1.metric(
        f"{model_choice} MAE (test set)", f"± ${_mae:.2f}",
        help="Mean absolute error on 15% held-out test set across all tiers/routes.",
    )
    _acc2.metric(
        "Prediction range (± MAE)",
        f"${max(0, cheapest_price - _mae):.2f} – ${cheapest_price + _mae:.2f}",
        help=f"Cheapest predicted tier: {cheapest}",
    )
if len(_cheapest_hist) >= 10:
    _q25 = float(_cheapest_hist.quantile(0.25))
    _q75 = float(_cheapest_hist.quantile(0.75))
    _acc3.metric(
        "Historical IQR (cheapest tier)",
        f"${_q25:.2f} – ${_q75:.2f}",
        help=f"25th–75th percentile of observed prices for {cheapest} on {src} → {dst} "
             f"(n={len(_cheapest_hist):,} rides).",
    )

# ── Results table (collapsed by default) ──────────────────────────────────────
with st.expander("See full price table"):
    result_rows = []
    for tier in ALL_TIERS:
        if tier not in pred_now.index:
            continue
        p_now  = pred_now.loc[tier, "predicted_price"]
        p_next = pred_next.loc[tier, "predicted_price"]
        delta  = p_next - p_now
        result_rows.append({
            "Service":    tier,
            "Platform":   SERVICE_MAP[tier],
            "Now":        f"${p_now:.2f}",
            "+1 hr":      f"${p_next:.2f}",
            "Delta":      f"{'▲' if delta > 0 else '▼'} ${abs(delta):.2f}",
            "_delta_num": delta,
        })
    res_df = pd.DataFrame(result_rows).sort_values("_delta_num").drop(columns="_delta_num")
    if not wait_1h:
        res_df = res_df.drop(columns=["+1 hr","Delta"])
    st.dataframe(res_df, use_container_width=True, hide_index=True)

# ── Historical context ─────────────────────────────────────────────────────────
with st.expander("Historical prices for this route"):
    hist = df[(df["source"] == src) & (df["destination"] == dst)]
    if len(hist) == 0:
        st.write("No historical data for this route.")
    else:
        hist_agg = (hist.groupby(["name","cab_type"])["price"]
                    .agg(["median","mean","count"])
                    .reset_index()
                    .rename(columns={"median":"Median","mean":"Mean","count":"Rides"})
                    .sort_values("Median"))
        fig_h = go.Figure(go.Bar(
            x=hist_agg["Median"], y=hist_agg["name"],
            orientation="h",
            marker_color=[COLORS[c] for c in hist_agg["cab_type"]],
            text=[f"${v:.2f}" for v in hist_agg["Median"]],
            textposition="outside",
        ))
        apply_chart_theme(fig_h, height=350, margin=dict(t=30, b=40, l=120, r=80))
        fig_h.update_layout(
            xaxis_title="Historical median price ($)",
            title=f"Historical median price: {src} → {dst}",
        )
        st.plotly_chart(fig_h, use_container_width=True)
