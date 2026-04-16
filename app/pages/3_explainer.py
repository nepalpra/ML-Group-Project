import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data import COLORS, SERIES_COLORS, apply_chart_theme
from utils.model import (
    models_available, shap_available,
    load_models, load_explainers, load_shap_artifacts,
    linear_models_available, load_linear_models, compute_linear_shap,
    build_input_row, NUM_FEATURES, CAT_FEATURES,
)

st.set_page_config(page_title="SHAP Explainer", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:8px; padding:12px 16px;
}
[data-testid="stAlert"][data-type="info"] {
    background:#F0F9FF; border-left:4px solid #0284C7; border-radius:0 6px 6px 0;
}
details summary p { font-size:13px !important; color:#64748B !important; }
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:22px; font-weight:700; color:#0F172A; margin-bottom:2px;">Model Explainer</div>'
    '<div style="font-size:13px; color:#64748B; margin-bottom:16px;">SHAP-powered explanations for LightGBM and XGBoost predictions.</div>',
    unsafe_allow_html=True,
)

if not models_available():
    st.error("Model files not found. Run `02_modeling.ipynb` and save artifacts first.")
    st.stop()

if not shap_available():
    st.error("SHAP artifacts not found. Run the save-artifacts cell in the notebook first.")
    st.stop()

lgb, xgb, prep, feat_names = load_models()
lgb_sv, xgb_sv, X_shap, y_shap, lgb_pred, xgb_pred, lgb_base, xgb_base = load_shap_artifacts()
N_SHAP = len(y_shap)

N_TOP = st.sidebar.slider("Features to show", 10, 30, 15)
_lin_avail = linear_models_available()
_model_opts = ["LightGBM", "XGBoost"] + (["Linear", "Ridge"] if _lin_avail else [])
model_sel = st.sidebar.radio("Primary model", _model_opts)

_is_linear = model_sel in ("Linear", "Ridge")

if _is_linear:
    _lin_sv, _lin_base, _lin_pred = compute_linear_shap(model_sel)
    sv_main   = _lin_sv
    base_main = _lin_base
    pred_main = _lin_pred
else:
    sv_main   = lgb_sv   if model_sel == "LightGBM" else xgb_sv
    base_main = lgb_base if model_sel == "LightGBM" else xgb_base
    pred_main = lgb_pred if model_sel == "LightGBM" else xgb_pred


def top_idx(sv, n):
    return np.argsort(np.abs(sv).mean(axis=0))[::-1][:n]


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "What matters most?",
    "How does it vary?",
    "Deep dive",
    "For this ride",
    "How accurate?",
])

# ── What matters most? ────────────────────────────────────────────────────────
with tabs[0]:
    # Compute top feature names for the headline
    _lgb_top_feat = feat_names[np.argsort(np.abs(lgb_sv).mean(axis=0))[::-1][0]]
    _lgb_2nd_feat = feat_names[np.argsort(np.abs(lgb_sv).mean(axis=0))[::-1][1]]
    st.markdown(
        f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
        f'<code style="background:#F1F5F9; padding:2px 6px; border-radius:4px;">{_lgb_top_feat}</code> and '
        f'<code style="background:#F1F5F9; padding:2px 6px; border-radius:4px;">{_lgb_2nd_feat}</code> '
        f'dominate — service tier explains most price variance by construction</p>',
        unsafe_allow_html=True,
    )
    st.caption("Higher bar = feature has larger average impact on price prediction.")

    # LightGBM — full-width primary
    _lgb_top  = top_idx(lgb_sv, N_TOP)
    _lgb_vals = np.abs(lgb_sv).mean(axis=0)[_lgb_top][::-1]
    _lgb_feats= [feat_names[i] for i in _lgb_top][::-1]
    _fig_lgb  = go.Figure(go.Bar(
        x=_lgb_vals, y=_lgb_feats, orientation="h",
        marker_color=COLORS["Uber"],
        text=[f"{v:.3f}" for v in _lgb_vals], textposition="outside",
        hovertemplate="%{y}<br>mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    apply_chart_theme(_fig_lgb, height=max(380, N_TOP * 26), margin=dict(t=40,b=40,l=160,r=80))
    _fig_lgb.update_layout(title="<b>LightGBM</b> — primary model", xaxis_title="mean |SHAP value|")
    _fig_lgb.update_xaxes(range=[0, _lgb_vals.max() * 1.3])
    st.plotly_chart(_fig_lgb, use_container_width=True)

    # XGBoost + linear models — secondary row
    _gi_secondary = [(xgb_sv, "XGBoost", SERIES_COLORS[0])]
    if _lin_avail:
        _lr_sv, _, _ = compute_linear_shap("Linear")
        _rd_sv, _, _ = compute_linear_shap("Ridge")
        _gi_secondary += [(_lr_sv, "Linear", SERIES_COLORS[1]), (_rd_sv, "Ridge", SERIES_COLORS[2])]

    _gi_cols = st.columns(len(_gi_secondary))
    for col, (sv, label, color) in zip(_gi_cols, _gi_secondary):
        top   = top_idx(sv, N_TOP)
        vals  = np.abs(sv).mean(axis=0)[top][::-1]
        feats = [feat_names[i] for i in top][::-1]
        fig   = go.Figure(go.Bar(
            x=vals, y=feats, orientation="h",
            marker_color=color, opacity=0.85,
            hovertemplate="%{y}<br>mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        apply_chart_theme(fig, height=max(320, N_TOP * 22), margin=dict(t=40,b=40,l=160,r=60))
        fig.update_layout(title=f"<b>{label}</b>", xaxis_title="mean |SHAP value|")
        fig.update_xaxes(range=[0, vals.max() * 1.3])
        col.plotly_chart(fig, use_container_width=True)

    # ── SHAP rank stability ────────────────────────────────────────────────────
    with st.expander("Are these rankings stable? — Rank stability across bootstrap samples"):
        st.caption(
            "5 bootstrap samples of 500 rows drawn from the SHAP test set. "
            "Mean rank and rank std quantify how consistently each feature is ranked."
        )
        _rng_stab = np.random.default_rng(0)
        _n_stab, _n_boot_stab = 500, 5
        _ranks_all = []
        for _ in range(_n_boot_stab):
            _idx = _rng_stab.choice(len(lgb_sv), _n_stab, replace=True)
            _imp = np.abs(lgb_sv[_idx]).mean(axis=0)
            _ranks_all.append(pd.Series(_imp, index=feat_names).rank(ascending=False))
        _stab_df = pd.DataFrame(_ranks_all).T
        _stab_df["mean_rank"] = _stab_df.mean(axis=1)
        _stab_df["rank_std"]  = _stab_df.std(axis=1)
        _stab_top = _stab_df.nsmallest(N_TOP, "mean_rank")[["mean_rank","rank_std"]].reset_index()
        _stab_top.columns = ["Feature", "Mean rank", "Rank std"]
        _stab_top["Mean rank"] = _stab_top["Mean rank"].round(1)
        _stab_top["Rank std"]  = _stab_top["Rank std"].round(2)
        _stab_top["Stable?"]   = _stab_top["Rank std"] < 2.0
        st.dataframe(_stab_top, use_container_width=True, hide_index=True)
        _n_stable = int(_stab_top["Stable?"].sum())
        st.caption(
            f"{_n_stable}/{len(_stab_top)} top features have rank std < 2.0 "
            "(consistently ranked across bootstrap samples — robust to sampling variation)."
        )


# ── Beeswarm ─────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader(f"SHAP Beeswarm — {model_sel}")
    st.caption(
        "Each dot = one prediction. "
        "X-axis = SHAP value (impact on price). "
        "Color: :red[red] = high feature value, :blue[blue] = low feature value."
    )

    sv    = lgb_sv if model_sel == "LightGBM" else xgb_sv
    top   = top_idx(sv, N_TOP)
    rng_j = np.random.default_rng(0)

    fig = go.Figure()
    for rank, feat_i in enumerate(top[::-1]):
        vals    = sv[:, feat_i]
        fv      = X_shap[:, feat_i]
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        jitter  = rng_j.uniform(-0.35, 0.35, len(vals))

        fig.add_trace(go.Scatter(
            x=vals, y=rank + jitter,
            mode="markers",
            marker=dict(size=4, color=fv_norm, colorscale="RdBu_r",
                        cmin=0, cmax=1, opacity=0.55,
                        showscale=(rank == N_TOP - 1),
                        colorbar=dict(title="Feature<br>value",
                                      tickvals=[0,1], ticktext=["Low","High"],
                                      len=0.45, y=0.5) if rank == N_TOP - 1 else None),
            showlegend=False,
            hovertemplate=(f"<b>{feat_names[feat_i]}</b><br>"
                           "SHAP: %{x:.3f}<br>Feature: %{text}<extra></extra>"),
            text=[f"{v:.3f}" for v in fv],
        ))

    fig.add_vline(x=0, line_width=1.2, line_color="grey", line_dash="dash")
    fig.update_layout(
        height=max(500, N_TOP * 32),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20,b=60,l=180,r=110),
        xaxis_title="SHAP value  (← lowers price  |  raises price →)",
        yaxis=dict(tickvals=list(range(N_TOP)),
                   ticktext=[feat_names[i] for i in top[::-1]],
                   showgrid=False),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Dependence ────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("SHAP Dependence Plots")
    st.caption("How a feature's SHAP value changes with its magnitude. Color = interacting feature.")

    cont_opts = ["distance","surge_multiplier","hour","temperature","precipProbability","windSpeed"]
    dc1, dc2 = st.columns(2)
    with dc1:
        feat_x = st.selectbox("Primary feature (x-axis)", cont_opts, index=0)
    with dc2:
        interact = st.selectbox("Interaction color", [f for f in cont_opts if f != feat_x], index=0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("LightGBM","XGBoost"),
                        horizontal_spacing=0.12)

    for col_i, (sv, label) in enumerate([(lgb_sv,"LGB"),(xgb_sv,"XGB")], start=1):
        if feat_x not in feat_names or interact not in feat_names:
            st.warning(f"Feature '{feat_x}' not found in feature names.")
            break
        fi   = feat_names.index(feat_x)
        ii   = feat_names.index(interact)
        fv   = X_shap[:, fi]
        sv_f = sv[:, fi]
        cv   = X_shap[:, ii]

        fig.add_trace(go.Scatter(
            x=fv, y=sv_f, mode="markers",
            marker=dict(size=4, color=cv, colorscale="Viridis", opacity=0.4,
                        showscale=(col_i == 2),
                        colorbar=dict(title=interact, len=0.5, y=0.5)),
            showlegend=False,
            hovertemplate=f"{feat_x}: %{{x:.2f}}<br>SHAP: %{{y:.3f}}<extra>{label}</extra>",
        ), row=1, col=col_i)

        # Binned trend
        bin_df = (pd.DataFrame({"fv":fv,"sv":sv_f})
                  .assign(bin=lambda d: pd.cut(d["fv"], bins=30))
                  .groupby("bin", observed=True)[["fv","sv"]].mean().dropna())
        fig.add_trace(go.Scatter(
            x=bin_df["fv"], y=bin_df["sv"], mode="lines",
            line=dict(color="black", width=2), showlegend=False,
        ), row=1, col=col_i)

    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(height=440, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      margin=dict(t=50,b=60,l=60,r=80),
                      title=f"SHAP dependence: {feat_x}  (color = {interact})")
    fig.update_xaxes(title_text=feat_x, showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(title_text=f"SHAP value for {feat_x}", showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)


# ── Waterfall ─────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("SHAP Waterfall — individual prediction breakdown")

    wf_source = st.radio(
        "Prediction source",
        ["Pick from test sample", "Use last Predictor input"],
        horizontal=True,
    )

    if wf_source == "Use last Predictor input" and "last_src" in st.session_state:
        ss = st.session_state
        # Build a single row for the selected service tier
        tier_pick = st.selectbox("Service tier to explain",
                                 list(from_utils := {"Lyft":"Lyft","UberX":"Uber"}.keys()) or
                                 ["Lyft","UberX"])
        from utils.data import SERVICE_MAP as SM
        cab_pick  = SM.get(tier_pick, "Lyft")
        row_df    = build_input_row(ss["last_src"], ss["last_dst"], ss["last_hour"],
                                    ss["last_weekend"], ss["last_temp"], ss["last_precip"],
                                    ss["last_wind"], ss["last_surge"], ss["last_distance"],
                                    tier_pick, cab_pick)
        X_row  = prep.transform(row_df[NUM_FEATURES + CAT_FEATURES])
        if _is_linear:
            import shap as _shap
            _lr_m, _rd_m = load_linear_models()
            _wf_m  = _lr_m if model_sel == "Linear" else _rd_m
            _wf_bg = np.load(__import__("pathlib").Path(__file__).resolve().parents[2] / "model" / "X_shap.npy")
            _wf_exp = _shap.LinearExplainer(_wf_m, _wf_bg)
            sv_row = _wf_exp.shap_values(X_row)[0]
            base   = float(_wf_exp.expected_value)
            pred_v = _wf_m.predict(X_row)[0]
        else:
            lgb_exp, xgb_exp = load_explainers()
            exp    = lgb_exp if model_sel == "LightGBM" else xgb_exp
            sv_row = exp.shap_values(X_row)[0]
            base   = lgb_base if model_sel == "LightGBM" else xgb_base
            pred_v = (lgb if model_sel == "LightGBM" else xgb).predict(X_row)[0]
        y_true = None
        idx_label = f"{ss['last_src']} → {ss['last_dst']} · {tier_pick}"
    else:
        sample_i = st.slider("Test sample index", 0, N_SHAP - 1, 0)
        sv_row   = sv_main[sample_i]
        base     = base_main
        pred_v   = pred_main[sample_i]
        y_true   = y_shap[sample_i]
        idx_label = f"Test sample {sample_i}"

    # Build waterfall data
    top12     = np.argsort(np.abs(sv_row))[::-1][:12]
    other_val = sv_row.sum() - sv_row[top12].sum()
    contribs  = list(sv_row[top12]) + [other_val]
    labels    = [feat_names[i] for i in top12] + ["Other features"]

    order    = np.argsort(contribs)
    contribs = [contribs[i] for i in order]
    labels   = [labels[i]   for i in order]

    measures = ["absolute"] + ["relative"] * len(contribs) + ["total"]
    x_vals   = [base] + contribs + [base + sv_row.sum()]
    y_labels = [f"Base (${base:.2f})"] + labels + [f"Prediction (${pred_v:.2f})"]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=measures,
        x=x_vals, y=y_labels,
        decreasing=dict(marker=dict(color="#4C78A8")),
        increasing=dict(marker=dict(color="#E45756")),
        totals=dict(marker=dict(color="#54A24B")),
        text=[f"${base:.2f}"] + [f"{v:+.2f}" for v in contribs] + [f"${pred_v:.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="#ddd", dash="dot", width=1)),
    ))
    fig.add_vline(x=base, line_dash="dash", line_color="grey", line_width=1)

    subtitle = f"Predicted: ${pred_v:.2f}"
    if y_true is not None:
        subtitle += f" · Actual: ${y_true:.2f} · Error: ${abs(y_true - pred_v):.2f}"

    fig.update_layout(
        title=f"<b>{model_sel} Waterfall — {idx_label}</b><br><sup>{subtitle}</sup>",
        height=560,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=80,b=40,l=220,r=90),
        xaxis_title="Price ($)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(":blue[Blue bars] reduce price · :red[Red bars] increase price · :green[Green] = final prediction")


# ── How accurate? ─────────────────────────────────────────────────────────────
with tabs[4]:
    from utils.model import load_model_metrics
    metrics = load_model_metrics()

    if not metrics:
        st.warning("model_metrics.json not found. Add metrics to the save-artifacts cell.")
    else:
        # TL;DR headline + metric cards first
        _lgb_mae = metrics.get("LightGBM", {}).get("mae", None)
        _xgb_mae = metrics.get("XGBoost",  {}).get("mae", None)
        _lgb_r2  = metrics.get("LightGBM", {}).get("r2",  None)
        if _lgb_mae and _lgb_r2:
            st.markdown(
                f'<p style="font-size:17px; font-weight:600; color:#0F172A; margin-bottom:4px;">'
                f'LightGBM achieves MAE = <span style="color:#059669">${_lgb_mae:.2f}</span> '
                f'and R² = <span style="color:#059669">{_lgb_r2:.4f}</span> on the held-out test set</p>',
                unsafe_allow_html=True,
            )

        # Metric cards
        _m_cols = st.columns(len(metrics))
        for _col, (_mname, _mvals) in zip(_m_cols, metrics.items()):
            _col.metric(
                _mname,
                f"MAE ${_mvals.get('mae', 0):.3f}",
                f"R² {_mvals.get('r2', 0):.4f}",
            )

        st.divider()

        # Comparison chart
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("MAE — lower is better",
                                            "RMSE — lower is better",
                                            "R² — higher is better"),
                            horizontal_spacing=0.1)

        _MCOLORS = SERIES_COLORS[:len(metrics)]
        model_names = list(metrics.keys())

        for ci, (metric, label) in enumerate([("mae","MAE"),("rmse","RMSE"),("r2","R²")], start=1):
            vals = [metrics[m][metric] for m in model_names]
            best = min(vals) if metric != "r2" else max(vals)
            cols_ = ["#059669" if v == best else _MCOLORS[i % len(_MCOLORS)]
                     for i, v in enumerate(vals)]
            fig.add_trace(go.Bar(
                x=model_names, y=vals, marker_color=cols_,
                text=[f"{v:.4f}" if metric == "r2" else f"${v:.2f}" for v in vals],
                textposition="outside", showlegend=False,
            ), row=1, col=ci)

        fig.update_layout(height=380, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                          margin=dict(t=60, b=60, l=40, r=40),
                          font=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
                          title="<b>Test set metrics — all models</b> "
                                "<sup style='color:#64748B'>Green = best</sup>")
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

    # SHAP agreement: which features do LGB and XGB agree on?
    st.markdown("**Do LightGBM and XGBoost agree on feature importance?**")
    top_lgb = pd.Series(np.abs(lgb_sv).mean(axis=0), index=feat_names).nlargest(N_TOP)
    top_xgb = pd.Series(np.abs(xgb_sv).mean(axis=0), index=feat_names).nlargest(N_TOP)
    union   = sorted(set(top_lgb.index) | set(top_xgb.index))
    agree_df = pd.DataFrame({
        "Feature":   union,
        "LGB mean|SHAP|": [top_lgb.get(f, 0) for f in union],
        "XGB mean|SHAP|": [top_xgb.get(f, 0) for f in union],
    }).sort_values("LGB mean|SHAP|", ascending=False)

    fig2 = go.Figure()
    for col, label, color in [
        ("LGB mean|SHAP|", "LightGBM", "#4C78A8"),
        ("XGB mean|SHAP|", "XGBoost",  "#E45756"),
    ]:
        fig2.add_trace(go.Bar(
            x=agree_df[col], y=agree_df["Feature"],
            orientation="h", name=label, marker_color=color, opacity=0.75,
        ))
    fig2.update_layout(
        barmode="group", height=max(400, len(union)*28),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20,b=40,l=180,r=40),
        xaxis_title="mean |SHAP value|",
        legend_title="Model",
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Per-platform MAE breakdown ─────────────────────────────────────────────
    st.markdown("**Per-platform prediction error — does the model treat Uber and Lyft equally?**")
    # cab_type_Uber is feature index 20, cab_type_Lyft is index 19 in X_shap
    _cab_lyft_idx = feat_names.index("cab_type_Lyft") if "cab_type_Lyft" in feat_names else None
    _cab_uber_idx = feat_names.index("cab_type_Uber") if "cab_type_Uber" in feat_names else None

    if _cab_lyft_idx is not None and _cab_uber_idx is not None:
        _mask_uber = X_shap[:, _cab_uber_idx] > 0.5
        _mask_lyft = X_shap[:, _cab_lyft_idx] > 0.5

        _plat_rows = []
        for _plat, _mask in [("Uber", _mask_uber), ("Lyft", _mask_lyft)]:
            for _mname, _pred_v in [("LightGBM", lgb_pred), ("XGBoost", xgb_pred)]:
                _err = np.abs(_pred_v[_mask] - y_shap[_mask])
                _plat_rows.append({
                    "Platform": _plat, "Model": _mname,
                    "MAE ($)": round(float(_err.mean()), 3),
                    "RMSE ($)": round(float(np.sqrt((_err**2).mean())), 3),
                    "n": int(_mask.sum()),
                })
        _plat_df = pd.DataFrame(_plat_rows)
        _pp1, _pp2 = st.columns(2)
        with _pp1:
            st.dataframe(_plat_df, use_container_width=True, hide_index=True)
        with _pp2:
            _pfig = go.Figure()
            for _mname, _color in [("LightGBM","#4C78A8"),("XGBoost","#E45756")]:
                _sub = _plat_df[_plat_df["Model"] == _mname]
                _pfig.add_trace(go.Bar(
                    x=_sub["Platform"], y=_sub["MAE ($)"],
                    name=_mname, marker_color=_color,
                    text=[f"${v:.3f}" for v in _sub["MAE ($)"]],
                    textposition="outside",
                ))
            _pfig.update_layout(
                barmode="group", height=300,
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(t=20, b=40, l=60, r=20),
                yaxis_title="MAE ($)", title="MAE by platform",
            )
            _pfig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(_pfig, use_container_width=True)

        _lgb_uber_mae = float(_plat_df[(_plat_df.Platform=="Uber") & (_plat_df.Model=="LightGBM")]["MAE ($)"].values[0])
        _lgb_lyft_mae = float(_plat_df[(_plat_df.Platform=="Lyft") & (_plat_df.Model=="LightGBM")]["MAE ($)"].values[0])
        st.caption(
            f"LightGBM MAE: Uber ${_lgb_uber_mae:.2f} vs Lyft ${_lgb_lyft_mae:.2f} "
            f"(Δ = ${abs(_lgb_lyft_mae - _lgb_uber_mae):.2f}). "
            "Small gap → model does not systematically favour either platform."
        )
    else:
        st.info("Platform dummy features not found in feat_names — per-platform breakdown unavailable.")
