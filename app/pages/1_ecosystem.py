import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import (
    load_data, compute_q1, compute_q2, compute_q3,
    compute_q4, compute_q5, compute_q6,
    COLORS, REGIME_COLORS, SERIES_COLORS, SIG_COLORS,
    TIME_ORDER, NEIGHBORHOODS, apply_chart_theme,
)
from utils.stats import (
    compute_route_significance, compute_normality_tests,
    compute_weather_adjusted, compute_kmeans_validation,
    compute_route_ppm_bootstrap,
)

st.set_page_config(page_title="Ecosystem", layout="wide")

# ── Global CSS (scoped to this page) ──────────────────────────────────────────
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
details summary p { font-size:13px !important; color:#64748B !important; }
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:4px;">
    <span style="font-size:22px; font-weight:700; color:#0F172A;">Ecosystem Analysis</span>
    <span style="font-size:13px; color:#64748B; margin-left:12px;">
        Boston rideshare pricing landscape · six analytical questions
    </span>
</div>
""", unsafe_allow_html=True)

df = load_data()

# ── Persistent KPI header (always visible above tabs) ─────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.metric("Rides", f"{len(df):,}")
h2.metric("Avg price", f"${df['price'].mean():.2f}")
h3.metric("Routes", f"{df['route'].nunique()}")
h4.metric("Date range", "Nov – Dec 2018")

st.divider()

tabs = st.tabs([
    "① Platform",
    "② Tiers",
    "③ Drivers",
    "④ Routes",
    "⑤ Weather",
    "⑥ Regimes",
])

# ─────────────────────────────────────────────────────────────────────────────
# ① PLATFORM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    pivot, time_agg = compute_q1(df)
    _uber_cheap = int((pivot["cheaper"] == "Uber").sum())
    _lyft_cheap = int((pivot["cheaper"] == "Lyft").sum())
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'Uber is cheaper on <span style="color:{COLORS["Uber"]}">{_uber_cheap}</span> of '
        f'{len(pivot)} routes; Lyft on <span style="color:{COLORS["Lyft"]}">{_lyft_cheap}</span>'
        f'</p>',
        unsafe_allow_html=True,
    )

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        src_f = st.selectbox("Filter by source (optional)", ["All"] + NEIGHBORHOODS, key="q1_src")
    with fc2:
        time_f = st.selectbox("Filter by time of day", ["All"] + TIME_ORDER, key="q1_time")

    view = pivot.copy()
    if src_f != "All":
        view = view[view["src"] == src_f]

    # Bar chart — price diff per route
    fig = px.bar(
        view, x="price_diff", y="route", color="cheaper",
        color_discrete_map=COLORS, orientation="h",
        hover_data={"uber_median_price":":.2f","lyft_median_price":":.2f","pct_diff":":.1f"},
        labels={"price_diff":"Lyft − Uber ($)","route":"Route","cheaper":"Cheaper platform",
                "uber_median_price":"Uber median ($)","lyft_median_price":"Lyft median ($)",
                "pct_diff":"% cheaper"},
        title="Price difference per route (Lyft − Uber) · positive = Uber cheaper",
    )
    fig.add_vline(x=0, line_width=1.5, line_color="grey")
    fig.update_layout(height=max(500, len(view)*18),
                      yaxis={"categoryorder":"total ascending"},
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      margin=dict(l=220,r=40,t=50,b=40))
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

    # ── Statistical significance ───────────────────────────────────────────────
    _sig_df, _n_tests = compute_route_significance(df)
    _view_sig = (_sig_df if src_f == "All"
                 else _sig_df[_sig_df["route"].str.startswith(src_f + " →")])
    _n_sig    = int(_view_sig["sig_bonferroni"].sum())
    _n_tot    = len(_view_sig)
    _uber_sig = int(((_view_sig["difference"] < 0) & _view_sig["sig_bonferroni"]).sum())
    _lyft_sig = int(((_view_sig["difference"] > 0) & _view_sig["sig_bonferroni"]).sum())

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _sc1.metric("Routes tested",             str(_n_tot))
    _sc2.metric("Significant (Bonferroni)",  f"{_n_sig} / {_n_tot}",
                help=f"α = 0.05 / {_n_tests} tests = {0.05/_n_tests:.4f} per test")
    _sc3.metric("Uber cheaper (sig.)",       str(_uber_sig))
    _sc4.metric("Lyft cheaper (sig.)",       str(_lyft_sig))

    with st.expander("Welch t-test results — all routes (Bonferroni corrected)"):
        _disp = _view_sig[["route","uber_median","lyft_median","difference",
                            "p_bonferroni","sig_bonferroni"]].copy()
        _disp.columns = ["Route","Uber Median ($)","Lyft Median ($)",
                         "Δ Lyft−Uber ($)","p (Bonferroni)","Significant"]
        for _c in ["Uber Median ($)","Lyft Median ($)","Δ Lyft−Uber ($)"]:
            _disp[_c] = _disp[_c].round(2)
        _disp["p (Bonferroni)"] = _disp["p (Bonferroni)"].apply(lambda x: f"{x:.4f}")
        st.dataframe(_disp.reset_index(drop=True), use_container_width=True, height=300)

    # Heatmap
    st.markdown("**Source × Destination price difference heatmap**")
    heat = pivot.pivot(index="src", columns="dst", values="price_diff")
    fig2 = px.imshow(heat, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     text_auto=".2f", aspect="auto",
                     labels={"color":"Lyft−Uber ($)"},
                     title="Blue = Uber cheaper · Red = Lyft cheaper")
    fig2.update_layout(height=520, margin=dict(t=50,b=40))
    fig2.update_xaxes(tickangle=30)
    st.plotly_chart(fig2, use_container_width=True)

    # Time-of-day breakdown
    st.markdown("**Median price by platform and time of day**")
    t_view = time_agg.copy()
    if time_f != "All":
        t_view = t_view[t_view["time_period"] == time_f]
    fig3 = px.bar(t_view, x="time_period", y="median_price", color="cab_type",
                  barmode="group", color_discrete_map=COLORS, text_auto=".2f",
                  category_orders={"time_period": TIME_ORDER},
                  labels={"median_price":"Median Price ($)","time_period":"Time of Day","cab_type":"Platform"})
    fig3.update_layout(height=380, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                       margin=dict(t=20,b=40,l=40,r=20))
    fig3.update_traces(textposition="outside")
    fig3.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ② SERVICE TIERS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    tier_order, iqr_df, ppm_df = compute_q2(df)
    _top_tier = iqr_df.sort_values("median", ascending=False).iloc[0]
    _bot_tier = iqr_df.sort_values("median").iloc[0]
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'<span style="color:{COLORS["Lyft"]}">{_top_tier["name"]}</span> is the most expensive tier '
        f'(median ${_top_tier["median"]:.2f}); '
        f'<span style="color:{COLORS["Uber"]}">{_bot_tier["name"]}</span> the cheapest '
        f'(${_bot_tier["median"]:.2f})</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Price distribution by service tier**")
        fig = go.Figure()
        for tier in tier_order:
            sub  = df[df["name"] == tier]
            cab  = sub["cab_type"].iloc[0]
            col  = COLORS[cab]
            fig.add_trace(go.Box(
                y=sub["price"], name=tier,
                marker_color=col, opacity=0.8, boxmean=True,
                hovertemplate=f"<b>{tier}</b><br>Price: %{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(height=480, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                          margin=dict(t=20,b=60,l=40,r=20),
                          xaxis_tickangle=30, yaxis_title="Price ($)")
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Median price per mile by service tier**")
        fig2 = px.bar(ppm_df, x="median_ppm", y="name", orientation="h",
                      color="median_ppm", color_continuous_scale="Blues",
                      labels={"median_ppm":"Median $/mile","name":"Service"},
                      text_auto=".2f")
        fig2.update_layout(height=480, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20,b=40,l=120,r=60),
                           coloraxis_showscale=False)
        fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**IQR summary table**")
    display_cols = ["name","cab_type","count","mean","median","q1","q3","iqr"]
    st.dataframe(
        iqr_df[[c for c in display_cols if c in iqr_df.columns]]
        .sort_values("median", ascending=False)
        .style.format({c: "${:.2f}" for c in ["mean","median","q1","q3","iqr"]
                       if c in iqr_df.columns}),
        use_container_width=True,
    )

    # ── Normality justification ────────────────────────────────────────────────
    with st.expander("Why median / IQR? — D'Agostino normality tests per tier"):
        _norm = compute_normality_tests(df)
        _nn   = int((~_norm["normal"]).sum())
        st.caption(
            f"{_nn}/{len(_norm)} tiers fail normality (p < 0.05) → "
            "median and IQR are the appropriate summary statistics, not mean ± SD."
        )
        st.dataframe(
            _norm[["name","n","p_value","normal"]]
            .rename(columns={"name":"Tier","n":"N rides",
                             "p_value":"p-value (D'Agostino)","normal":"Normal?"})
            .style.format({"p-value (D'Agostino)":"{:.2e}"}),
            use_container_width=True, height=300,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ③ PRICE DRIVERS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    individual, cum_df = compute_q3(df)
    _r2_total = cum_df["r2"].iloc[-1]
    _r2_dist  = cum_df.iloc[0]["r2"]
    _r2_tier  = cum_df.iloc[1]["gain"]
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'Product type explains <span style="color:#6366F1">{_r2_tier*100:.0f}%</span> of variance; '
        f'distance <span style="color:#6366F1">{_r2_dist*100:.0f}%</span>; '
        f'all four factors together <span style="color:#059669">{_r2_total*100:.1f}%</span></p>',
        unsafe_allow_html=True,
    )

    PALETTE = SERIES_COLORS[:4]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Standalone R² — each feature alone**")
        labels = list(individual.keys())
        vals   = [round(v, 4) for v in individual.values()]
        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h", marker_color=PALETTE,
            text=[f"{v*100:.1f}%" for v in vals], textposition="outside",
        ))
        fig.update_layout(height=320, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                          margin=dict(t=20,b=40,l=160,r=80),
                          xaxis_title="R²")
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9",
                         range=[0, max(vals)*1.3])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Incremental R² — adding features in sequence**")
        fig2 = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"]*(len(cum_df)-1),
            x=cum_df["label"], y=cum_df["gain"],
            text=[f"+{v*100:.1f}%" if i > 0 else f"{v*100:.1f}%"
                  for i, v in enumerate(cum_df["gain"])],
            textposition="outside",
            connector={"line":{"color":"#888","dash":"dot"}},
            increasing={"marker":{"color":"#4C78A8"}},
            totals={"marker":{"color":"#54A24B"}},
        ))
        fig2.update_layout(height=320, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20,b=80,l=40,r=40),
                           yaxis_title="Incremental R²",
                           yaxis_tickformat=".0%")
        fig2.update_xaxes(tickangle=15)
        fig2.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    final_r2 = cum_df["r2"].iloc[-1]
    st.info(
        f"All four factors combined explain **{final_r2*100:.1f}%** of price variance. "
        f"The remaining **{(1-final_r2)*100:.1f}%** is captured by non-linear interactions "
        f"— which is why LightGBM outperforms linear models."
    )

    # ── Incremental F-tests ────────────────────────────────────────────────────
    with st.expander("Are the R² gains statistically significant? — Incremental F-tests"):
        from scipy.stats import f as _f_dist
        _n_obs  = int(df[["price","distance","surge_multiplier","name","cab_type"]].dropna().shape[0])
        # (label, r2_new, r2_old, k_new_features, k_total_features)
        _fsteps = [
            ("Distance",           cum_df.iloc[0]["r2"], 0.0,                   1,  1),
            ("+ Product type",     cum_df.iloc[1]["r2"], cum_df.iloc[0]["r2"], 11, 12),
            ("+ Surge multiplier", cum_df.iloc[2]["r2"], cum_df.iloc[1]["r2"],  1, 13),
            ("+ Platform",         cum_df.iloc[3]["r2"], cum_df.iloc[2]["r2"],  1, 14),
        ]
        _frows = []
        for _lbl, _r2n, _r2o, _kn, _kt in _fsteps:
            _delta_r2 = _r2n - _r2o
            _denom    = (1 - _r2n) / (_n_obs - _kt - 1)
            _F        = (_delta_r2 / _kn) / _denom if _denom > 0 else float("inf")
            _p        = float(_f_dist.sf(_F, _kn, _n_obs - _kt - 1))
            _frows.append({"Feature added": _lbl, "ΔR²": f"{_delta_r2:.4f}",
                           "F-stat": f"{_F:,.0f}", "p-value": f"{_p:.2e}"})
        st.caption(
            f"n = {_n_obs:,}. With this sample size all gains are significant; "
            "F-stat magnitude reflects practical importance."
        )
        st.dataframe(pd.DataFrame(_frows), use_container_width=True, hide_index=True)

    # 3D scatter (sampled)
    with st.expander("3D — Price as a function of distance, surge and product type"):
        from sklearn.linear_model import LinearRegression as LR
        sample3d = df.sample(5000, random_state=42)
        X_r = df[["distance","surge_multiplier"]].values
        y_r = df["price"].values
        lr  = LR().fit(X_r, y_r)
        d_v = np.linspace(df["distance"].quantile(0.01), df["distance"].quantile(0.99), 40)
        s_v = np.linspace(df["surge_multiplier"].min(), df["surge_multiplier"].max(), 20)
        D, S = np.meshgrid(d_v, s_v)
        Z    = lr.predict(np.c_[D.ravel(), S.ravel()]).reshape(D.shape)
        PROD_COLORS = {
            "UberPool":"#b5c8e2","UberX":"#6baed6","UberXL":"#2171b5",
            "Black":"#08306b","Black SUV":"#041d40","WAV":"#737373",
            "Shared":"#f9b8d4","Lyft":"#e878ac","Lyft XL":"#E91E8C",
            "Lux":"#b5006e","Lux Black":"#7a004b","Lux Black XL":"#3d0025",
        }
        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(x=D, y=S, z=Z, opacity=0.35,
                                   colorscale=[[0,"#c6dbef"],[1,"#08306b"]],
                                   showscale=False, name="Regression plane"))
        for prod, color in PROD_COLORS.items():
            sub = sample3d[sample3d["name"] == prod]
            if sub.empty: continue
            fig3d.add_trace(go.Scatter3d(
                x=sub["distance"], y=sub["surge_multiplier"], z=sub["price"],
                mode="markers", name=prod,
                marker=dict(size=2, color=color, opacity=0.45),
            ))
        coef_d, coef_s = lr.coef_
        fig3d.update_layout(
            height=650,
            title=f"Price = {lr.intercept_:.2f} + {coef_d:.2f}×distance + {coef_s:.2f}×surge",
            scene=dict(
                xaxis_title="Distance (mi)", yaxis_title="Surge multiplier", zaxis_title="Price ($)",
                camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8)),
            ),
            legend=dict(title="Product type"),
        )
        st.plotly_chart(fig3d, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ④ ROUTES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    route_ppm, overpriced = compute_q4(df)
    _ppm_max = overpriced.iloc[0]
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'<span style="color:#EF4444">{_ppm_max["route"]}</span> is the most expensive route at '
        f'${_ppm_max["avg_ppm"]:.2f}/mile — routes with CI above the mean are '
        f'statistically overpriced</p>',
        unsafe_allow_html=True,
    )

    metric = st.radio("Show metric", ["avg_price","avg_ppm"], horizontal=True,
                      format_func=lambda x: "Avg Price ($)" if x == "avg_price" else "Avg $/mile",
                      key="q4_metric")

    heat_ppm = route_ppm.pivot(index="source", columns="destination", values=metric)
    fig = px.imshow(
        heat_ppm, color_continuous_scale="YlOrRd", text_auto=".2f", aspect="auto",
        labels={"color": "Avg Price ($)" if metric == "avg_price" else "Avg $/mile"},
        title=f"Route heatmap — {'Average price' if metric == 'avg_price' else 'Price per mile'}",
    )
    fig.update_layout(height=540, margin=dict(t=50,b=40))
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 20 most expensive routes ($/mile) — bootstrap 95% CI**")
    _boot_df = compute_route_ppm_bootstrap(df, n_boot=500, top_n=20)
    _overall_mean = float(_boot_df["overall_mean"].iloc[0])

    _fig2 = go.Figure()
    for _, _row in _boot_df.iterrows():
        _color = "#E45756" if _row["sig_above_mean"] else "#AAAAAA"
        _fig2.add_trace(go.Bar(
            x=[_row["mean_ppm"]],
            y=[_row["route"]],
            orientation="h",
            marker_color=_color,
            error_x=dict(
                type="data",
                array=[_row["ci_hi"] - _row["mean_ppm"]],
                arrayminus=[_row["mean_ppm"] - _row["ci_lo"]],
                color="#555", thickness=1.5, width=4,
            ),
            text=[f"${_row['mean_ppm']:.2f}"],
            textposition="outside",
            showlegend=False,
            hovertemplate=(
                f"<b>{_row['route']}</b><br>"
                f"Mean $/mile: ${_row['mean_ppm']:.2f}<br>"
                f"95% CI: [${_row['ci_lo']:.2f}, ${_row['ci_hi']:.2f}]<br>"
                f"n = {int(_row['n'])}<extra></extra>"
            ),
        ))
    _fig2.add_vline(x=_overall_mean, line_dash="dash", line_color="#F58518", line_width=1.5,
                    annotation_text=f"Overall mean ${_overall_mean:.2f}/mi",
                    annotation_position="top right")
    _fig2.update_layout(
        height=520,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=50, l=220, r=100),
        xaxis_title="Mean $/mile",
        yaxis={"categoryorder": "total ascending"},
    )
    _fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(_fig2, use_container_width=True)
    _n_sig_ppm = int(_boot_df["sig_above_mean"].sum())
    st.caption(
        f"Red bars: {_n_sig_ppm} routes whose 95% CI lower bound exceeds the dataset-wide "
        f"mean (${_overall_mean:.2f}/mi) — statistically overpriced. "
        "Grey bars: above average but CI overlaps the mean."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ WEATHER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    weather_agg, sample = compute_q5(df)
    _wmax = weather_agg.groupby("weather_group")["avg_price"].mean().idxmax()
    _wmin = weather_agg.groupby("weather_group")["avg_price"].mean().idxmin()
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'<span style="color:#06B6D4">{_wmax}</span> conditions correlate with the highest prices; '
        f'<span style="color:#10B981">{_wmin}</span> with the lowest — '
        f'but time-of-day confounding explains much of the gap</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Avg price by weather condition & platform**")
        fig = px.bar(weather_agg, x="avg_price", y="weather_group", color="cab_type",
                     barmode="group", color_discrete_map=COLORS, orientation="h",
                     text_auto=".2f",
                     labels={"avg_price":"Avg Price ($)","weather_group":"Condition","cab_type":"Platform"})
        fig.update_layout(height=420, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                          margin=dict(t=20,b=40,l=140,r=80))
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Temperature vs price (sampled 15k rides)**")
        fig2 = px.scatter(sample, x="temperature", y="price", color="cab_type",
                          color_discrete_map=COLORS, opacity=0.2, trendline="lowess",
                          labels={"temperature":"Temperature (°F)","price":"Price ($)","cab_type":"Platform"})
        fig2.update_traces(marker=dict(size=3))
        fig2.update_layout(height=420, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20,b=40,l=40,r=20))
        fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        fig2.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Precipitation probability vs price**")
    fig3 = px.scatter(sample, x="precipProbability", y="price", color="cab_type",
                      color_discrete_map=COLORS, opacity=0.15, trendline="lowess",
                      labels={"precipProbability":"Precipitation Probability","price":"Price ($)","cab_type":"Platform"})
    fig3.update_traces(marker=dict(size=3))
    fig3.update_layout(height=380, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                       margin=dict(t=20,b=40,l=40,r=20))
    fig3.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig3.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Confounder-adjusted weather effect ────────────────────────────────────
    with st.expander("Is the weather effect real? — Partial regression (adjusted for time-of-day & tier)"):
        _raw, _adj, _omean = compute_weather_adjusted(df)
        # Merge for side-by-side
        _wr = _raw[["weather_group","mean_price","ci_lo","ci_hi"]].copy()
        _wa = _adj[["weather_group","mean_resid","ci_lo","ci_hi"]].copy()
        _wr.columns = ["weather_group","val","lo","hi"]
        _wr["type"] = "Raw avg price"
        _wa.columns = ["weather_group","val","lo","hi"]
        _wa["type"] = "Adjusted residual"

        _wfig = make_subplots(rows=1, cols=2,
                              subplot_titles=["Raw avg price", "After removing time-of-day & tier"])
        for _wrow in _wr.sort_values("val", ascending=False).itertuples():
            _wfig.add_trace(go.Bar(
                name=_wrow.weather_group, x=[_wrow.weather_group], y=[_wrow.val],
                error_y=dict(type="data",
                             array=[_wrow.hi - _wrow.val],
                             arrayminus=[_wrow.val - _wrow.lo],
                             color="#555", thickness=1.5),
                marker_color="#4C78A8", showlegend=False,
            ), row=1, col=1)
        for _wrow in _wa.sort_values("val", ascending=False).itertuples():
            _wfig.add_trace(go.Bar(
                name=_wrow.weather_group, x=[_wrow.weather_group], y=[_wrow.val],
                error_y=dict(type="data",
                             array=[_wrow.hi - _wrow.val],
                             arrayminus=[_wrow.val - _wrow.lo],
                             color="#555", thickness=1.5),
                marker_color="#E91E8C", showlegend=False,
            ), row=1, col=2)
        _wfig.add_hline(y=0, row=1, col=2, line_dash="dash", line_color="grey", line_width=1)
        _wfig.update_layout(height=380, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                            margin=dict(t=50, b=60, l=60, r=20))
        _wfig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(_wfig, use_container_width=True)

        # Compute shrinkage
        _raw_range = float(_wr["val"].max() - _wr["val"].min())
        _adj_range = float(_wa["val"].max() - _wa["val"].min())
        _shrink    = (1 - _adj_range / _raw_range) * 100 if _raw_range > 0 else 0
        st.caption(
            f"Partial regression removes time-of-day + service tier effects. "
            f"The weather spread shrinks from ${_raw_range:.2f} to ${_adj_range:.2f} "
            f"({_shrink:.0f}% explained by confounders). "
            "Residual differences reflect weather's independent effect."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ PRICING REGIMES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    with st.spinner("Running clustering…"):
        norm, hour_dist, stats = compute_q6(df)
    _top_regime  = stats.iloc[0]
    _std_regime  = stats[stats["regime"] == "Standard"]
    _std_price   = float(_std_regime["avg_price"].values[0]) if not _std_regime.empty else 0
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'Six pricing regimes identified — '
        f'<span style="color:#6366F1">Surge Heavy</span> averages '
        f'${_top_regime["avg_price"]:.2f}/ride vs '
        f'${_std_price:.2f} for Standard conditions</p>',
        unsafe_allow_html=True,
    )
    st.caption("KMeans k=6 on: hour, surge, price, distance, temperature, precipitation, wind, weekend")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("**Regime profiles — normalised feature means**")
        fig = px.imshow(norm, color_continuous_scale="RdBu_r", text_auto=".2f",
                        aspect="auto", zmin=0, zmax=1,
                        labels={"color":"Normalised mean"},
                        title="Higher value (red) = defining characteristic of that regime")
        fig.update_layout(height=380, margin=dict(t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Avg price & ride count by regime**")
        fig2 = go.Figure()
        for _, row in stats.iterrows():
            fig2.add_trace(go.Bar(
                x=[row["avg_price"]], y=[row["regime"]],
                orientation="h",
                marker_color=REGIME_COLORS.get(row["regime"], "#888"),
                text=[f"${row['avg_price']:.2f} · {row['count']//1000}k rides"],
                textposition="outside", showlegend=False,
            ))
        fig2.update_layout(height=380, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           margin=dict(t=20,b=40,l=140,r=120),
                           xaxis_title="Avg Price ($)")
        fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Bubble chart ──────────────────────────────────────────────────────────
    st.markdown("**Pricing regimes — when they occur, what they cost, how common they are**")

    avg_hour = (
        hour_dist.groupby("regime")[["hour", "count"]]
        .apply(lambda x: (x["hour"] * x["count"]).sum() / x["count"].sum())
        .reset_index(name="avg_hour")
    )
    bubble_df = stats.merge(avg_hour, on="regime")
    bubble_df["label"] = bubble_df.apply(
        lambda r: f"<b>{r['regime']}</b><br>Avg price: ${r['avg_price']:.2f}<br>"
                  f"Peak hour: {r['avg_hour']:.0f}:00<br>Rides: {r['count']:,}",
        axis=1,
    )

    fig_bubble = go.Figure()
    for _, row in bubble_df.iterrows():
        fig_bubble.add_trace(go.Scatter(
            x=[row["avg_hour"]],
            y=[row["avg_price"]],
            mode="markers+text",
            name=row["regime"],
            text=[row["regime"]],
            textposition="top center",
            hovertext=[row["label"]],
            hoverinfo="text",
            marker=dict(
                size=row["count"] / bubble_df["count"].max() * 80 + 20,
                color=REGIME_COLORS.get(row["regime"], "#888"),
                opacity=0.82,
                line=dict(width=1.5, color="white"),
            ),
            showlegend=False,
        ))

    fig_bubble.update_layout(
        height=440,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=30, b=60, l=60, r=40),
        xaxis=dict(
            title="Peak hour of day (avg)", tickmode="linear", dtick=2,
            showgrid=True, gridcolor="#F1F5F9", range=[-0.5, 23.5],
        ),
        yaxis=dict(
            title="Avg price ($)",
            showgrid=True, gridcolor="#F1F5F9",
        ),
    )
    fig_bubble.add_annotation(
        text="Bubble size = ride volume",
        xref="paper", yref="paper", x=1, y=-0.12,
        showarrow=False, font=dict(size=11, color="#888"),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("**Hour-of-day ride volume by regime**")
    fig3 = go.Figure()
    for regime in hour_dist["regime"].unique():
        sub = hour_dist[hour_dist["regime"] == regime]
        fig3.add_trace(go.Scatter(
            x=sub["hour"], y=sub["count"], mode="lines", name=regime,
            line=dict(color=REGIME_COLORS.get(regime, "#888"), width=2.5),
        ))
    fig3.update_layout(height=360, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                       margin=dict(t=20,b=40,l=40,r=20),
                       xaxis_title="Hour of day (0 = midnight)",
                       yaxis_title="Ride count",
                       legend_title="Regime")
    fig3.update_xaxes(showgrid=True, gridcolor="#F1F5F9", dtick=2)
    fig3.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig3, use_container_width=True)

    # ── KMeans cluster-count validation ───────────────────────────────────────
    with st.expander("Is k=6 a good choice? — KMeans elbow & silhouette validation"):
        _km_df = compute_kmeans_validation(df, k_max=10)
        _km_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Inertia (elbow method)", "Silhouette score"])
        _km_fig.add_trace(go.Scatter(
            x=_km_df["k"], y=_km_df["inertia"],
            mode="lines+markers", name="Inertia",
            line=dict(color="#4C78A8", width=2), marker=dict(size=7),
        ), row=1, col=1)
        _km_fig.add_trace(go.Scatter(
            x=_km_df["k"], y=_km_df["silhouette"],
            mode="lines+markers", name="Silhouette",
            line=dict(color="#E91E8C", width=2), marker=dict(size=7),
        ), row=1, col=2)
        # Highlight k=6
        _row6 = _km_df[_km_df["k"] == 6]
        if not _row6.empty:
            _km_fig.add_vline(x=6, line_dash="dash", line_color="#F58518",
                              line_width=1.5, row=1, col=1)
            _km_fig.add_vline(x=6, line_dash="dash", line_color="#F58518",
                              line_width=1.5, row=1, col=2)
        _km_fig.update_layout(height=360, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                              margin=dict(t=50, b=50, l=60, r=20), showlegend=False)
        _km_fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", dtick=1)
        _km_fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(_km_fig, use_container_width=True)

        _best_k = int(_km_df.loc[_km_df["silhouette"].idxmax(), "k"])
        _sil6   = float(_km_df.loc[_km_df["k"] == 6, "silhouette"].values[0]) if 6 in _km_df["k"].values else None
        _sil_best = float(_km_df["silhouette"].max())
        _sil_note = (
            f"k=6 silhouette = {_sil6:.3f} vs best k={_best_k} silhouette = {_sil_best:.3f}. "
            if _sil6 is not None else ""
        )
        st.caption(
            f"10k stratified sample; features: hour, surge, price, distance, temperature, "
            f"precipitation, wind, weekend. {_sil_note}"
            "Orange line = k=6 used in analysis."
        )
