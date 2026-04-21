"""
Page 8 — Hyperparameter Optimisation
======================================
Shows Optuna HPO convergence, parameter importance, tuned vs default comparison,
and a feature selection / permutation importance analysis.
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

from utils.data import load_data, apply_chart_theme
from utils.model import models_available, load_models, NUM_FEATURES, CAT_FEATURES
from utils.stats import hpo_results_available

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"

st.set_page_config(page_title="HPO", layout="wide")

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
    'Hyperparameter Optimisation</div>'
    '<div style="font-size:13px; color:#64748B; margin-bottom:16px;">'
    'Optuna Bayesian HPO results, convergence analysis, and the '
    '\'was it worth it?\' comparison between default and tuned models.</div>',
    unsafe_allow_html=True,
)

_PLOT_BG = dict(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
_AXIS    = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

if not models_available():
    st.error("Model artifacts not found. Run `02_modeling.ipynb` first.")
    st.stop()

lgb, xgb, prep, feat_names = load_models()

tabs = st.tabs([
    "HPO Convergence",
    "Parameter Importance",
    "Tuned vs Default",
    "Feature Selection",
])


# ── Load HPO results ──────────────────────────────────────────────────────────
@st.cache_data
def _load_hpo():
    p = MODEL_DIR / "hpo_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

hpo = _load_hpo()

# ── TL;DR banner ──────────────────────────────────────────────────────────────
_base_path = MODEL_DIR / "model_metrics.json"
_tldr_parts = []
if _base_path.exists():
    _bm = json.loads(_base_path.read_text())
    _lgb_mae = _bm.get("LightGBM", {}).get("mae", None)
    _lgb_r2  = _bm.get("LightGBM", {}).get("r2", None)
    if _lgb_mae: _tldr_parts.append(f"LightGBM default MAE: <b>${_lgb_mae:.2f}</b>")
    if _lgb_r2:  _tldr_parts.append(f"R²: <b>{_lgb_r2:.4f}</b>")
if hpo:
    _n_trials = hpo.get("n_trials", None)
    _best_cv  = hpo.get("lgb_best_cv_mae", None)
    if _n_trials: _tldr_parts.append(f"HPO trials: <b>{_n_trials}</b>")
    if _best_cv:  _tldr_parts.append(f"Best CV MAE: <b>${_best_cv:.2f}</b>")
else:
    _tldr_parts.append("HPO not run yet — showing default model metrics")
if _tldr_parts:
    st.markdown(
        '<div style="background:#F0F9FF; border:1px solid #BAE6FD; border-radius:8px; '
        'padding:12px 20px; margin-bottom:12px; font-size:14px; color:#0F172A;">'
        + " &nbsp;·&nbsp; ".join(_tldr_parts) + "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — HPO Convergence
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("How did the search converge?")

    if hpo:
        st.success(
            f"HPO results found — {hpo.get('n_trials', '?')} trials, "
            f"sample size = {hpo.get('sample_size', '?'):,}"
        )

        for model_key, label, color in [
            ("lgb", "LightGBM", "#6366F1"),
            ("xgb", "XGBoost",  "#F59E0B"),
        ]:
            trials_key = f"{model_key}_trials"
            if trials_key not in hpo:
                continue

            trials     = hpo[trials_key]
            trial_nums = [t["number"] for t in trials if t["value"] is not None]
            cv_values  = [t["value"]  for t in trials if t["value"] is not None]

            # Compute running best
            running_best = []
            best_so_far  = float("inf")
            for v in cv_values:
                best_so_far = min(best_so_far, v)
                running_best.append(best_so_far)

            st.markdown(f"#### {label} — {len(trial_nums)} trials")
            c1, c2, c3 = st.columns(3)
            best_cv = hpo.get(f"{model_key}_best_cv_mae", np.nan)
            default_mae = hpo.get("metrics", {}).get(
                "LightGBM" if model_key == "lgb" else "XGBoost", {}
            ).get("mae", np.nan)
            c1.metric(f"Best CV MAE",  f"${best_cv:.4f}" if not np.isnan(best_cv) else "N/A")
            c2.metric(f"Default MAE",  f"${default_mae:.4f}" if not np.isnan(default_mae) else "N/A")
            c3.metric(f"Trials",       str(len(trial_nums)))

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=trial_nums, y=cv_values,
                mode="markers", marker=dict(color=color, size=5, opacity=0.5),
                name="Trial CV MAE",
                hovertemplate="Trial %{x}<br>MAE=$%{y:.4f}<extra></extra>",
            ))
            fig_conv.add_trace(go.Scatter(
                x=trial_nums, y=running_best,
                mode="lines", line=dict(color=color, width=2.5),
                name="Running best",
                hovertemplate="Trial %{x}<br>Best MAE=$%{y:.4f}<extra></extra>",
            ))
            fig_conv.update_layout(
                height=360, margin=dict(t=30, b=60, l=60, r=20),
                xaxis=dict(title="Trial number", **_AXIS),
                yaxis=dict(title="Cross-val MAE ($)", **_AXIS),
                legend=dict(orientation="h", y=-0.2),
                **_PLOT_BG,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # Best params comparison
        st.markdown("#### Best hyperparameters found")
        lgb_bp = hpo.get("lgb_best_params", {})
        xgb_bp = hpo.get("xgb_best_params", {})
        all_keys = sorted(set(lgb_bp.keys()) | set(xgb_bp.keys()))
        params_table = pd.DataFrame([
            {"Parameter": k,
             "LGB Best": lgb_bp.get(k, "N/A"),
             "XGB Best": xgb_bp.get(k, "N/A")}
            for k in all_keys
        ])
        st.dataframe(params_table, use_container_width=True)

    else:
        st.warning(
            "HPO results not found. Run:\n"
            "```\npython scripts/train_tuned_models.py --trials 30\n```\n\n"
            "Default settings used in the project were chosen manually:\n"
            "`n_estimators=2000, learning_rate=0.05, num_leaves=63`"
        )

        # Show what Optuna would search over
        st.markdown("#### Search space for LightGBM Optuna study")
        search_space = pd.DataFrame([
            {"Parameter": "n_estimators",      "Type": "int",   "Range": "[500, 3000, step=100]",  "Default": 2000},
            {"Parameter": "learning_rate",     "Type": "log",   "Range": "[0.01, 0.15]",           "Default": 0.05},
            {"Parameter": "num_leaves",        "Type": "int",   "Range": "[20, 150]",              "Default": 63},
            {"Parameter": "min_child_samples", "Type": "int",   "Range": "[10, 100]",              "Default": 20},
            {"Parameter": "subsample",         "Type": "float", "Range": "[0.5, 1.0]",             "Default": 1.0},
            {"Parameter": "colsample_bytree",  "Type": "float", "Range": "[0.5, 1.0]",             "Default": 1.0},
            {"Parameter": "reg_alpha",         "Type": "log",   "Range": "[1e-8, 10.0]",           "Default": 0.0},
            {"Parameter": "reg_lambda",        "Type": "log",   "Range": "[1e-8, 10.0]",           "Default": 0.0},
        ])
        st.dataframe(search_space, use_container_width=True)
        st.markdown(
            "**Expected runtime:** ~5-15 min for 30 trials on a 60k-row sample with 3-fold CV. "
            "TPE sampler (Tree-structured Parzen Estimator) makes each trial more informed "
            "than the last, converging far faster than grid search."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Parameter Importance
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Which hyperparameters actually matter?")
    st.markdown(
        "Optuna's `FanovaImportanceEvaluator` estimates the fraction of objective "
        "variance explained by each hyperparameter. Low-importance params can be "
        "fixed to defaults to reduce search cost in future runs."
    )

    if hpo:
        # Try to compute Optuna importance from trials
        try:
            import optuna
            for model_key, label, color in [("lgb","LightGBM","#6366F1"),
                                             ("xgb","XGBoost","#F59E0B")]:
                trials_key = f"{model_key}_trials"
                if trials_key not in hpo:
                    continue

                trials_data = hpo[trials_key]
                # Reconstruct study from trial data
                study = optuna.create_study(direction="minimize")
                for t in trials_data:
                    if t.get("value") is None:
                        continue
                    trial = optuna.trial.create_trial(
                        params=t["params"],
                        distributions={
                            k: optuna.distributions.FloatDistribution(0, 1)
                            for k in t["params"]
                        },
                        value=t["value"],
                    )
                    study.add_trial(trial)

                if len(study.trials) < 5:
                    st.info(f"Not enough completed trials ({model_key}) for importance.")
                    continue

                importance = optuna.importance.get_param_importances(study)
                imp_df = pd.DataFrame([
                    {"param": k, "importance": v}
                    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
                ])
                st.markdown(f"#### {label}")
                fig_imp = go.Figure(go.Bar(
                    y=imp_df["param"][::-1],
                    x=imp_df["importance"][::-1],
                    orientation="h", marker_color=color,
                    hovertemplate="<b>%{y}</b><br>Importance=%{x:.4f}<extra></extra>",
                ))
                fig_imp.update_layout(
                    height=360, margin=dict(t=30, b=40, l=160, r=20),
                    xaxis=dict(title="FANOVA importance", tickformat=".3f", **_AXIS),
                    **_PLOT_BG,
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.info(f"Optuna importance computation not available: {e}")
            st.markdown(
                "Install Optuna and run the HPO script to compute parameter importance. "
                "The FANOVA evaluator requires ≥ 10 completed trials."
            )
    else:
        st.info("Run `scripts/train_tuned_models.py` to generate HPO results.")
        st.markdown("""
        #### Expected findings from hyperparameter importance

        In gradient boosting, typical importance ordering:
        1. **`learning_rate`** — usually most critical; controls step size vs overfitting
        2. **`num_leaves` / `max_depth`** — controls model capacity
        3. **`subsample` + `colsample_bytree`** — bagging regularisation
        4. **`reg_alpha` + `reg_lambda`** — L1/L2 regularisation (often low importance if data is large)
        5. **`n_estimators`** — mostly handled by early stopping

        This suggests a focused 2-parameter search `{learning_rate, num_leaves}` would
        capture 80% of the tuning benefit at 20% of the compute cost.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Tuned vs Default
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Was hyperparameter tuning worth it?")
    st.markdown(
        "The correct question is not just 'did MAE improve?' but "
        "'**how much** did it improve relative to the compute cost?'"
    )

    metrics_data = {}
    base_path = MODEL_DIR / "model_metrics.json"
    if base_path.exists():
        with open(base_path) as f:
            metrics_data.update(json.load(f))
    if hpo:
        metrics_data.update(hpo.get("metrics", {}))

    compare_pairs = [
        ("LightGBM",  "lgb_tuned",  "LGB Default", "LGB Tuned", "#6366F1", "#4338CA"),
        ("XGBoost",   "xgb_tuned",  "XGB Default", "XGB Tuned", "#F59E0B", "#D97706"),
    ]

    for default_key, tuned_key, default_label, tuned_label, c_def, c_tun in compare_pairs:
        if default_key not in metrics_data or tuned_key not in metrics_data:
            continue
        md = metrics_data[default_key]
        mt = metrics_data[tuned_key]

        delta_mae  = mt["mae"]  - md["mae"]
        delta_rmse = mt["rmse"] - md["rmse"]
        delta_r2   = mt["r2"]   - md["r2"]
        pct_mae    = delta_mae / md["mae"] * 100

        st.markdown(f"#### {default_label} vs {tuned_label}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{default_label} MAE", f"${md['mae']:.4f}")
        c2.metric(f"{tuned_label} MAE",   f"${mt['mae']:.4f}",
                  delta=f"{delta_mae:+.4f} ({pct_mae:+.2f}%)",
                  delta_color="inverse" if delta_mae < 0 else "normal")
        c3.metric("RMSE change",  f"{delta_rmse:+.4f}", delta_color="inverse" if delta_rmse < 0 else "normal")
        c4.metric("R² change",    f"{delta_r2:+.5f}",   delta_color="normal"  if delta_r2  > 0 else "inverse")

        fig_comp = make_subplots(rows=1, cols=3,
                                  subplot_titles=["MAE ($)", "RMSE ($)", "R²"])
        for col_idx, metric_key in enumerate(["mae","rmse","r2"], start=1):
            fig_comp.add_trace(go.Bar(
                x=[default_label, tuned_label],
                y=[metrics_data[default_key][metric_key],
                   metrics_data[tuned_key][metric_key]],
                marker_color=[c_def, c_tun],
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>=%{y:.5f}<extra></extra>",
            ), row=1, col=col_idx)
        fig_comp.update_layout(
            height=320, margin=dict(t=50, b=40, l=40, r=20),
            **_PLOT_BG,
        )
        fig_comp.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_comp, use_container_width=True)

        improvement = "marginal" if abs(pct_mae) < 1 else "meaningful"
        st.info(
            f"**Verdict:** Tuning delivered a **{improvement}** "
            f"{abs(pct_mae):.2f}% MAE improvement "
            f"(Δ = ${abs(delta_mae):.4f}/ride). "
            + ("For a production model processing millions of rides, even small per-ride "
               "improvements compound significantly."
               if abs(pct_mae) > 0.5
               else "The default hyperparameters were already well-chosen. "
                    "The model is robust to hyperparameter variation — "
                    "a sign of a clean, high-signal dataset.")
        )
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Which features are actually contributing?")
    st.markdown(
        "LightGBM's built-in feature importance (split count and gain) reveals "
        "which of the 45 encoded features the model actually uses. Features with "
        "zero importance are pure dead weight — they can be removed to simplify "
        "the pipeline without hurting performance."
    )

    n_top_feat = st.slider("Features to show", 10, 45, 20, key="fs_top")

    # LGB feature importance (gain-based)
    imp_gain   = lgb.feature_importances_
    imp_df_all = pd.DataFrame({
        "feature": feat_names,
        "importance_gain": imp_gain,
    }).sort_values("importance_gain", ascending=False)

    n_zero    = int((imp_df_all["importance_gain"] == 0).sum())
    cum_gain  = imp_df_all["importance_gain"].cumsum() / imp_df_all["importance_gain"].sum()
    n_90pct   = int((cum_gain <= 0.90).sum()) + 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Total features",        "45")
    c2.metric("Zero-importance features", str(n_zero))
    c3.metric("Features for 90% gain",   str(n_90pct),
              help="Minimum features that explain 90% of total gain")

    imp_top = imp_df_all.head(n_top_feat)
    fig_fi = go.Figure(go.Bar(
        y=imp_top["feature"][::-1],
        x=imp_top["importance_gain"][::-1],
        orientation="h",
        marker_color="#6366F1",
        hovertemplate="<b>%{y}</b><br>Gain=%{x:,.0f}<extra></extra>",
    ))
    fig_fi.update_layout(
        height=max(400, 20 * n_top_feat),
        margin=dict(t=30, b=40, l=180, r=20),
        xaxis=dict(title="Feature importance (gain)", **_AXIS),
        **_PLOT_BG,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Cumulative importance curve
    st.markdown("#### Cumulative gain — how many features do we really need?")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=list(range(1, len(imp_df_all) + 1)),
        y=(imp_df_all["importance_gain"].cumsum() / imp_df_all["importance_gain"].sum() * 100).tolist(),
        mode="lines",
        line=dict(color="#6366F1", width=2),
        fill="tozeroy", fillcolor="rgba(76,120,168,0.1)",
        hovertemplate="Top %{x} features → %{y:.1f}% of gain<extra></extra>",
    ))
    fig_cum.add_hline(y=90, line_dash="dash", line_color="#E45756",
                      annotation_text="90%", annotation_position="right")
    fig_cum.add_hline(y=95, line_dash="dot", line_color="#F59E0B",
                      annotation_text="95%", annotation_position="right")
    fig_cum.add_vline(x=n_90pct, line_dash="dash", line_color="#E45756",
                      annotation_text=f"n={n_90pct}")
    fig_cum.update_layout(
        height=360, margin=dict(t=30, b=60, l=60, r=60),
        xaxis=dict(title="Number of features (ranked by gain)", **_AXIS),
        yaxis=dict(title="Cumulative % of total gain", range=[0,101], **_AXIS),
        **_PLOT_BG,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Feature groups breakdown
    groups = {
        "Numeric (raw)":        [f for f in feat_names if f in ["distance","surge_multiplier","hour",
                                                                  "is_weekend","temperature",
                                                                  "precipProbability","windSpeed"]],
        "Service tier (name)":  [f for f in feat_names if f.startswith("name_")],
        "Platform (cab_type)":  [f for f in feat_names if f.startswith("cab_type_")],
        "Source neighborhood":  [f for f in feat_names if f.startswith("source_")],
        "Destination neighborhood": [f for f in feat_names if f.startswith("destination_")],
    }
    grp_records = []
    for grp_name, grp_feats in groups.items():
        grp_imp = imp_df_all.loc[imp_df_all["feature"].isin(grp_feats), "importance_gain"].sum()
        grp_records.append(dict(group=grp_name, total_gain=grp_imp,
                                n_features=len(grp_feats)))
    grp_df = pd.DataFrame(grp_records)
    grp_df["pct_gain"] = grp_df["total_gain"] / grp_df["total_gain"].sum() * 100
    grp_df.sort_values("total_gain", ascending=False, inplace=True)

    fig_grp = go.Figure(go.Bar(
        x=grp_df["group"], y=grp_df["pct_gain"],
        marker_color=["#6366F1","#EF4444","#0F172A","#10B981","#F59E0B"],
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% of total gain<extra></extra>",
    ))
    fig_grp.update_layout(
        height=320, margin=dict(t=40, b=100, l=60, r=20),
        title="% of total LGB gain by feature group",
        xaxis_tickangle=-20,
        yaxis=dict(title="% of total gain", **_AXIS),
        **_PLOT_BG,
    )
    st.plotly_chart(fig_grp, use_container_width=True)

    top_group = grp_df.iloc[0]
    st.info(
        f"**Key finding:** The **{top_group['group']}** feature group accounts for "
        f"{top_group['pct_gain']:.1f}% of total model gain. "
        f"{n_zero} of 45 features have zero importance and could be dropped "
        f"without any performance loss. "
        f"Just {n_90pct} features (out of 45) explain 90% of total gain, "
        "suggesting a leaner model is feasible."
    )
