import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.data import load_data
from utils.model import models_available, shap_available

st.set_page_config(
    page_title="Uber & Lyft Price Intelligence",
    page_icon=":material/directions_car:",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
.stApp { background-color: #F8FAFC; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Expander label — secondary tone */
details summary p {
    font-size: 13px !important;
    color: #64748B !important;
}

/* st.info */
[data-testid="stAlert"][data-type="info"] {
    background: #F0F9FF;
    border-left: 4px solid #0284C7;
    border-radius: 0 6px 6px 0;
}

/* st.success */
[data-testid="stAlert"][data-type="success"] {
    background: #F0FDF4;
    border-left: 4px solid #16A34A;
    border-radius: 0 6px 6px 0;
}

/* st.warning */
[data-testid="stAlert"][data-type="warning"] {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 0 6px 6px 0;
}

/* Divider */
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar header ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:8px 0 16px 0; border-bottom:1px solid #E2E8F0; margin-bottom:14px;">
    <div style="font-weight:700; font-size:13px; color:#0F172A; letter-spacing:0.03em;">
        UBER & LYFT INTELLIGENCE
    </div>
    <div style="font-size:11px; color:#94A3B8; margin-top:2px;">
        Boston, MA &nbsp;·&nbsp; Nov – Dec 2018
    </div>
</div>
<div style="font-size:10px; font-weight:600; color:#94A3B8; letter-spacing:0.08em; margin-bottom:4px; margin-top:2px;">
    DATA &amp; EXPLORATION
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
    data_ok = True
except FileNotFoundError:
    data_ok = False

# ── Hero strip ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border-radius: 12px;
    padding: 32px 40px 28px 40px;
    margin-bottom: 8px;
">
    <div style="font-size:26px; font-weight:700; color:#FFFFFF; letter-spacing:-0.02em; margin-bottom:6px;">
        Uber &amp; Lyft Price Intelligence
    </div>
    <div style="font-size:14px; color:#94A3B8;">
        Boston, MA &nbsp;·&nbsp; November – December 2018 &nbsp;·&nbsp;
        LightGBM &amp; XGBoost price prediction &nbsp;·&nbsp; R² = 0.97
    </div>
</div>
""", unsafe_allow_html=True)

# ── Status bar ─────────────────────────────────────────────────────────────────
if data_ok:
    _model_ok = models_available()
    _shap_ok  = shap_available()

    def _pill(label, ok, ok_text="ready", warn_text="not found"):
        color = "#059669" if ok else "#D97706"
        bg    = "#F0FDF4" if ok else "#FFFBEB"
        icon  = "✓" if ok else "⚠"
        text  = ok_text if ok else warn_text
        return (
            f'<span style="background:{bg}; color:{color}; border:1px solid {color}33; '
            f'border-radius:20px; padding:3px 10px; font-size:12px; font-weight:600; '
            f'margin-right:8px; white-space:nowrap;">'
            f'{icon} {label}: {text}</span>'
        )

    st.markdown(
        '<div style="margin:12px 0 20px 0; display:flex; flex-wrap:wrap; gap:4px;">'
        + _pill("Dataset", True, f"{len(df):,} rides")
        + _pill("Models", _model_ok)
        + _pill("SHAP", _shap_ok)
        + _pill("Advanced models", (Path(__file__).parents[1] / "model" / "lgb_q50.pkl").exists(),
                ok_text="ready", warn_text="run scripts")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total rides",    f"{len(df):,}")
    m2.metric("Avg price",      f"${df['price'].mean():.2f}")
    m3.metric("Median price",   f"${df['price'].median():.2f}")
    m4.metric("Routes",         f"{df['route'].nunique()}")
    m5.metric("Service tiers",  f"{df['name'].nunique()}")
else:
    st.error("rideshare_kaggle.csv not found in data/ — place the dataset file and reload.")
    st.stop()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
def _nav_card(icon, title, body, accent="#6366F1"):
    return f"""
<div style="
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-left:4px solid {accent};
    border-radius:10px;
    padding:18px 20px 16px 20px;
    height:100%;
    box-sizing:border-box;
">
    <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:6px;">
        {icon}&nbsp; {title}
    </div>
    <div style="font-size:13px; color:#64748B; line-height:1.5;">
        {body}
    </div>
</div>"""

st.markdown("#### Core Analysis")
c1, c2, c3 = st.columns(3)
c1.markdown(_nav_card(
    "📊", "Ecosystem",
    "Pricing landscape before modeling. Six analytical questions answered interactively — "
    "platform comparison, tier pricing, price drivers, route overpricing, weather, and regimes.",
    accent="#6366F1",
), unsafe_allow_html=True)
c2.markdown(_nav_card(
    "💰", "Price Predictor",
    "What-if price engine. Enter any route and conditions, get predicted prices for every "
    "Uber & Lyft tier. Includes a <em>wait 1 hour</em> savings comparison.",
    accent="#E91E8C",
), unsafe_allow_html=True)
c3.markdown(_nav_card(
    "🔍", "SHAP Explainer",
    "Model explanation panel. Global importance, beeswarm and dependence plots, "
    "per-prediction waterfall breakdowns, and model comparison.",
    accent="#0F172A",
), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("#### Advanced Analysis")
d1, d2, d3, d4, d5 = st.columns(5)
d1.markdown(_nav_card(
    "🔬", "Advanced SHAP",
    "Conditional SHAP (Uber vs Lyft), within-tier profiles, rank stability, interaction heatmap.",
    accent="#8B5CF6",
), unsafe_allow_html=True)
d2.markdown(_nav_card(
    "🗂️", "Model Zoo",
    "Quantile prediction intervals, MLP baseline, Nov→Dec temporal validation, per-tier breakdown.",
    accent="#06B6D4",
), unsafe_allow_html=True)
d3.markdown(_nav_card(
    "📈", "Demand & Supply",
    "Surge probability atlas, surge classifier, demand heatmaps, best-time-to-travel.",
    accent="#F59E0B",
), unsafe_allow_html=True)
d4.markdown(_nav_card(
    "⚙️", "HPO",
    "Optuna Bayesian search, convergence plots, parameter importance, tuned vs default.",
    accent="#EF4444",
), unsafe_allow_html=True)
d5.markdown(_nav_card(
    "📋", "Statistical Rigor",
    "Bootstrap CIs, Welch t-tests, partial regression — all embedded inline in Ecosystem.",
    accent="#10B981",
), unsafe_allow_html=True)

st.divider()

# ── Script runner ──────────────────────────────────────────────────────────────
with st.expander("Unlock pre-computed results — run these scripts once"):
    st.markdown("""
    The Advanced Analysis pages work immediately. Running these scripts
    pre-computes heavier artifacts for richer results:

    ```bash
    # Hyperparameter optimisation (~10 min, 30 trials)
    python scripts/train_tuned_models.py

    # Quantile regression, MLP, time-ordered validation (~15 min)
    python scripts/train_advanced_models.py

    # Surge classifier + demand aggregation (~5 min)
    python scripts/train_surge_model.py

    # Advanced SHAP: conditional, stability, interactions (~5–15 min)
    python scripts/compute_advanced_shap.py
    # Skip the slow interaction step:
    python scripts/compute_advanced_shap.py --skip-interactions
    ```
    """)

st.caption("Use the sidebar to navigate between pages.")
