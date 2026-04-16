"""
Page 5 — Advanced SHAP Analysis
=================================
Goes beyond the basic explainer with conditional SHAP, within-tier analysis,
feature stability across bootstrap samples, and interaction heatmaps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.model import models_available, shap_available, load_shap_artifacts
from utils.stats import advanced_shap_available

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"

st.set_page_config(page_title="Advanced SHAP", layout="wide")

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
    '<div style="font-size:22px; font-weight:700; color:#0F172A; margin-bottom:2px;">Advanced SHAP Analysis</div>'
    '<div style="font-size:13px; color:#64748B; margin-bottom:12px;">'
    'Conditional importance (Uber vs Lyft), within-tier profiles, rank stability, interaction heatmap.</div>',
    unsafe_allow_html=True,
)

_PLOT_BG = dict(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
_AXIS    = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

if not shap_available():
    st.error("SHAP artifacts not found. Run `02_modeling.ipynb` first.")
    st.stop()

# Load base SHAP artifacts (always available)
lgb_sv, xgb_sv, X_shap, y_shap, lgb_pred, xgb_pred, lgb_base, xgb_base = \
    load_shap_artifacts()

import joblib
feat_names = np.load(MODEL_DIR / "feature_names.npy", allow_pickle=True).tolist()
N, F = lgb_sv.shape


# ── Helper: get platform / tier masks from X_shap ────────────────────────────
def _platform_mask(X, feat, platform):
    col = feat.index(f"cab_type_{platform}")
    return X[:, col] > 0.5

def _tier_mask(X, feat, tier):
    col = feat.index(f"name_{tier}")
    return X[:, col] > 0.5

def _mean_abs_shap(sv, mask):
    if mask.sum() == 0:
        return np.zeros(sv.shape[1])
    return np.abs(sv[mask]).mean(axis=0)


UBER_MASK = _platform_mask(X_shap, feat_names, "Uber")
LYFT_MASK = _platform_mask(X_shap, feat_names, "Lyft")

tabs = st.tabs([
    "Conditional SHAP",
    "Within-Tier Analysis",
    "SHAP Stability",
    "Interaction Heatmap",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Conditional SHAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("What drives price differently for Uber vs Lyft?")
    st.markdown(
        "The global SHAP importance pools both platforms. By conditioning on "
        "`cab_type`, we reveal whether the **same features** drive Uber and Lyft "
        "pricing, or whether each platform has a distinct sensitivity profile."
    )

    # Try to load pre-computed conditional SHAP; fall back to real-time compute
    if advanced_shap_available():
        cond = joblib.load(MODEL_DIR / "shap_conditional.pkl")
        uber_abs  = np.array(cond["Uber"])
        lyft_abs  = np.array(cond["Lyft"])
        all_abs   = np.array(cond["All"])
        source = "pre-computed by `compute_advanced_shap.py`"
    else:
        uber_abs  = _mean_abs_shap(lgb_sv, UBER_MASK)
        lyft_abs  = _mean_abs_shap(lgb_sv, LYFT_MASK)
        all_abs   = _mean_abs_shap(lgb_sv, np.ones(N, dtype=bool))
        source = "computed on-the-fly from the 5k SHAP sample"
    st.caption(f"Source: {source} — Uber rows: {UBER_MASK.sum()}, Lyft rows: {LYFT_MASK.sum()}")

    n_top = st.slider("Features to show", 8, F, 15, key="cond_top")

    # Sort by overall importance
    order    = np.argsort(all_abs)[::-1][:n_top]
    feats_t  = [feat_names[i] for i in order]
    uber_t   = uber_abs[order]
    lyft_t   = lyft_abs[order]
    all_t    = all_abs[order]
    delta_t  = lyft_t - uber_t  # positive = Lyft more sensitive

    # Side-by-side grouped bar
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Bar(
        name="Uber", x=feats_t[::-1], y=uber_t[::-1],
        orientation="v",
        marker_color="#0F172A",
        hovertemplate="<b>%{x}</b><br>Uber mean|SHAP|=%{y:.4f}<extra></extra>",
    ))
    fig_cond.add_trace(go.Bar(
        name="Lyft", x=feats_t[::-1], y=lyft_t[::-1],
        orientation="v",
        marker_color="#E91E8C",
        hovertemplate="<b>%{x}</b><br>Lyft mean|SHAP|=%{y:.4f}<extra></extra>",
    ))
    fig_cond.update_layout(
        barmode="group", height=420,
        margin=dict(t=30, b=120, l=40, r=20),
        xaxis_title=None, yaxis_title="Mean |SHAP value|",
        xaxis_tickangle=-45, **_PLOT_BG, yaxis=_AXIS,
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_cond, use_container_width=True)

    # Delta chart: Lyft sensitivity minus Uber sensitivity
    st.markdown("#### Feature sensitivity delta: Lyft − Uber")
    st.caption("Positive = feature matters more for Lyft pricing. Negative = matters more for Uber.")
    delta_order = np.argsort(delta_t)
    fig_delta = go.Figure(go.Bar(
        x=[feats_t[i] for i in delta_order],
        y=[float(delta_t[i]) for i in delta_order],
        marker_color=["#E91E8C" if v > 0 else "#0F172A" for v in [delta_t[i] for i in delta_order]],
        hovertemplate="<b>%{x}</b><br>Δ=%{y:+.4f}<extra></extra>",
    ))
    fig_delta.add_hline(y=0, line_dash="solid", line_color="#555")
    fig_delta.update_layout(
        height=350, margin=dict(t=20, b=120, l=40, r=20),
        xaxis_title=None, yaxis_title="Lyft mean|SHAP| − Uber mean|SHAP|",
        xaxis_tickangle=-45, **_PLOT_BG, yaxis=_AXIS,
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # Top driver per platform
    top_uber_feat  = feats_t[np.argmax(uber_t)]
    top_lyft_feat  = feats_t[np.argmax(lyft_t)]
    max_delta_feat = feats_t[np.argmax(np.abs(delta_t))]
    st.info(
        f"**Key findings:**  \n"
        f"• Top driver for **Uber** rides: `{top_uber_feat}`  \n"
        f"• Top driver for **Lyft** rides: `{top_lyft_feat}`  \n"
        f"• Largest platform divergence: `{max_delta_feat}` "
        f"({'Lyft' if delta_t[np.argmax(np.abs(delta_t))] > 0 else 'Uber'} more sensitive)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Within-Tier Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Within-tier: what drives price variation inside a single service tier?")
    st.markdown(
        "Globally, service tier (name_*) is the dominant SHAP feature — it explains "
        "most price variance by construction. But within a single tier, tier encoding "
        "is constant, so SHAP re-weights to reveal what actually drives **intra-tier** "
        "price variation: surge, weather, route, or time of day."
    )

    # Load pre-computed if available; else compute on-the-fly
    if advanced_shap_available() and (MODEL_DIR / "shap_within_tier.pkl").exists():
        within = joblib.load(MODEL_DIR / "shap_within_tier.pkl")
        tiers_available = [k for k in within.keys() if k != "feature_names"]
    else:
        tiers_available = []
        within = {"feature_names": feat_names}
        for fn in feat_names:
            if fn.startswith("name_"):
                tier = fn[len("name_"):]
                mask = _tier_mask(X_shap, feat_names, tier)
                if mask.sum() >= 20:
                    within[tier] = _mean_abs_shap(lgb_sv, mask).tolist()
                    tiers_available.append(tier)

    tier_sel = st.selectbox("Select service tier", sorted(tiers_available))
    if tier_sel:
        tier_abs = np.array(within[tier_sel])
        # Suppress name_* and cab_type_* features (constant within tier)
        suppress = [feat_names.index(f) for f in feat_names
                    if f.startswith("name_") or f.startswith("cab_type_")]
        tier_abs_clean = tier_abs.copy()
        tier_abs_clean[suppress] = 0

        n_top_t  = st.slider("Features to show", 6, min(20, F), 12, key="tier_top")
        order_t  = np.argsort(tier_abs_clean)[::-1][:n_top_t]
        feats_t2 = [feat_names[i] for i in order_t]
        vals_t2  = tier_abs_clean[order_t]
        all_t2   = all_abs[order_t]

        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            name="Global (all tiers)", x=feats_t2, y=all_t2,
            marker_color="#aaa", opacity=0.6,
            hovertemplate="<b>%{x}</b><br>Global=%{y:.4f}<extra></extra>",
        ))
        fig_tier.add_trace(go.Bar(
            name=f"{tier_sel} only", x=feats_t2, y=vals_t2,
            marker_color="#7B2D8B",
            hovertemplate=f"<b>%{{x}}</b><br>{tier_sel}=%{{y:.4f}}<extra></extra>",
        ))
        fig_tier.update_layout(
            barmode="overlay", height=400,
            margin=dict(t=40, b=120, l=40, r=20),
            title=f"Feature importance: {tier_sel} vs global",
            xaxis_title=None, yaxis_title="Mean |SHAP value|",
            xaxis_tickangle=-45, **_PLOT_BG, yaxis=_AXIS,
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)

        top_within = feats_t2[0] if feats_t2 else "N/A"
        n_tier_rows = int(_tier_mask(X_shap, feat_names, tier_sel).sum())
        st.info(
            f"Within **{tier_sel}** ({n_tier_rows} SHAP sample rows): "
            f"top intra-tier price driver is `{top_within}`. "
            "Service tier dummies and cab_type are suppressed (constant within tier)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SHAP Stability
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Are SHAP importance rankings stable?")
    st.markdown(
        "SHAP values computed on different random subsets of test data should "
        "yield consistent feature rankings — if the top features change across "
        "samples, the explanation is unreliable. We compute SHAP on 5 seeds "
        "× 500 rows and measure rank variance per feature."
    )

    if advanced_shap_available() and (MODEL_DIR / "shap_stability.pkl").exists():
        stab = joblib.load(MODEL_DIR / "shap_stability.pkl")
        rank_df  = stab["rank_stability"]
        stab_df  = stab["stability_df"]
        source_s = "pre-computed by `compute_advanced_shap.py`"
    else:
        # Compute on-the-fly with 5 seeds
        rng = np.random.default_rng(0)
        records = []
        for seed in range(5):
            idx   = rng.choice(N, min(500, N), replace=False)
            sv_s  = lgb_sv[idx]
            m_abs = np.abs(sv_s).mean(axis=0)
            ranks = pd.Series(m_abs, index=feat_names).rank(ascending=False).astype(int)
            for feat, mean_val, rank in zip(feat_names, m_abs, ranks):
                records.append(dict(seed=seed, feature=feat,
                                    mean_abs_shap=float(mean_val), rank=int(rank)))
        stab_df = pd.DataFrame(records)
        rank_df = (
            stab_df.groupby("feature")
            .agg(mean_rank=("rank","mean"), std_rank=("rank","std"),
                 mean_abs=("mean_abs_shap","mean"))
            .reset_index()
            .sort_values("mean_rank")
        )
        source_s = "computed on-the-fly (5 seeds × 500 rows)"

    st.caption(f"Source: {source_s}")

    n_show = st.slider("Features to show", 8, min(25, len(rank_df)), 15, key="stab_top")
    top_feats = rank_df.head(n_show)

    # Rank stability bubble chart
    fig_stab = go.Figure()
    fig_stab.add_trace(go.Scatter(
        x=top_feats["mean_rank"],
        y=top_feats["std_rank"],
        mode="markers+text",
        text=top_feats["feature"],
        textposition="top center",
        marker=dict(
            size=top_feats["mean_abs"].values * 400 + 8,
            color=top_feats["std_rank"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Rank std"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Mean rank: %{x:.1f}<br>"
            "Rank std: %{y:.2f}<extra></extra>"
        ),
    ))
    fig_stab.update_layout(
        height=500, margin=dict(t=30, b=60, l=60, r=60),
        xaxis_title="Mean importance rank (1 = most important)",
        yaxis_title="Rank std across 5 seeds (0 = perfectly stable)",
        **_PLOT_BG, xaxis=_AXIS, yaxis=_AXIS,
    )
    st.plotly_chart(fig_stab, use_container_width=True)

    # Table with sparkline-like indicators
    st.markdown("#### Stability table — top features")
    disp = top_feats.copy()
    disp["Stable?"] = disp["std_rank"].apply(
        lambda s: "✓ Stable" if s < 2 else ("~ Moderate" if s < 4 else "✗ Unstable")
    )
    disp = disp.rename(columns={
        "feature": "Feature", "mean_rank": "Mean Rank",
        "std_rank": "Rank Std (5 seeds)", "mean_abs": "Mean |SHAP|",
    })
    disp["Mean |SHAP|"] = disp["Mean |SHAP|"].round(4)
    disp["Mean Rank"] = disp["Mean Rank"].round(1)
    disp["Rank Std (5 seeds)"] = disp["Rank Std (5 seeds)"].round(2)
    st.dataframe(disp[["Feature","Mean Rank","Rank Std (5 seeds)","Mean |SHAP|","Stable?"]].reset_index(drop=True),
                 use_container_width=True)

    stable_top5 = int((rank_df.head(5)["std_rank"] < 2).sum())
    st.info(
        f"**Key finding:** {stable_top5} of the top-5 features have rank std < 2 "
        "across the 5 bootstrap seeds, indicating the global importance ranking is "
        "stable and the explanation is reliable. "
        "Features with high rank std are either weakly predictive or highly correlated "
        "with other features (rank can flip between correlated features)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Interaction Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Feature interaction strengths")
    st.markdown(
        "SHAP interaction values decompose each prediction into main effects and "
        "pairwise interactions: `SHAP_interaction[i,j]` captures how feature j "
        "modulates the effect of feature i. Run `compute_advanced_shap.py` to "
        "pre-compute the full 500×45×45 interaction tensor."
    )

    interact_path = MODEL_DIR / "shap_interactions.npy"
    if interact_path.exists():
        st.success("Interaction artifacts found — loading pre-computed interactions.")
        interact = np.load(interact_path)   # (500, 45, 45)
        mean_int = np.abs(interact).mean(axis=0)
        np.fill_diagonal(mean_int, 0)

        # Top-N interactions
        n_top_int = st.slider("Top feature pairs to show", 5, 20, 10)
        pairs = []
        for i in range(F):
            for j in range(i + 1, F):
                pairs.append(dict(
                    feat_i=feat_names[i], feat_j=feat_names[j],
                    interaction=float(mean_int[i, j]),
                ))
        pairs_df = pd.DataFrame(pairs).sort_values("interaction", ascending=False).head(n_top_int)

        fig_pairs = go.Figure(go.Bar(
            y=[f"{r.feat_i} × {r.feat_j}" for _, r in pairs_df.iterrows()],
            x=pairs_df["interaction"].values,
            orientation="h",
            marker_color="#7B2D8B",
            hovertemplate="<b>%{y}</b><br>Mean |interaction|=%{x:.4f}<extra></extra>",
        ))
        fig_pairs.update_layout(
            height=max(300, 30 * n_top_int),
            margin=dict(t=30, b=40, l=250, r=40),
            xaxis_title="Mean |SHAP interaction value|",
            yaxis_title=None, **_PLOT_BG, xaxis=_AXIS,
        )
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Full interaction heatmap (reduced to numeric + top cat features)
        numeric_idx = list(range(7))  # first 7 features are numeric
        tier_idx    = [feat_names.index(f) for f in feat_names
                       if f.startswith("name_")][:6]
        show_idx    = numeric_idx + tier_idx
        show_names  = [feat_names[i] for i in show_idx]
        mat         = mean_int[np.ix_(show_idx, show_idx)]

        fig_heat = go.Figure(go.Heatmap(
            z=mat.tolist(),
            x=show_names,
            y=show_names,
            colorscale="Purples",
            hovertemplate="<b>%{x} × %{y}</b><br>Interaction=%{z:.4f}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=460, margin=dict(t=40, b=120, l=120, r=20),
            title="Mean |SHAP interaction| — numeric features + top tiers",
            xaxis_tickangle=-45, **_PLOT_BG,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.warning(
            "Interaction artifacts not found. "
            "Run `python scripts/compute_advanced_shap.py` to compute them "
            "(takes ~10 min for 500 rows × 45 × 45 interactions).\n\n"
            "In the meantime, here is a proxy: SHAP feature correlation heatmap "
            "(correlated SHAP values suggest interaction candidates)."
        )

        # Proxy: SHAP value correlation matrix for top numeric features
        top_idx    = np.argsort(np.abs(lgb_sv).mean(axis=0))[::-1][:12]
        sv_top     = lgb_sv[:, top_idx]
        corr_mat   = np.corrcoef(sv_top.T)
        top_names  = [feat_names[i] for i in top_idx]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat.tolist(), x=top_names, y=top_names,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            hovertemplate="<b>%{x} × %{y}</b><br>SHAP corr=%{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            height=460, margin=dict(t=50, b=130, l=130, r=20),
            title="SHAP value correlation — top 12 features (proxy for interactions)",
            xaxis_tickangle=-45, **_PLOT_BG,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Correlation ≈ ±1 suggests these features may interact in predictions. "
            "Run the script above for exact interaction values."
        )
