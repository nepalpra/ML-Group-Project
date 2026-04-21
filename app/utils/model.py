from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT  = Path(__file__).resolve().parents[2]
MODEL_DIR  = REPO_ROOT / "model"

NUM_FEATURES = ["distance","surge_multiplier","hour","is_weekend",
                "temperature","precipProbability","windSpeed"]
CAT_FEATURES = ["name","cab_type","source","destination"]

SERVICE_MAP = {
    "Shared": "Lyft", "Lyft": "Lyft", "Lyft XL": "Lyft",
    "Lux": "Lyft", "Lux Black": "Lyft", "Lux Black XL": "Lyft",
    "UberPool": "Uber", "UberX": "Uber", "UberXL": "Uber",
    "Black": "Uber", "Black SUV": "Uber", "WAV": "Uber",
}
ALL_SERVICES = list(SERVICE_MAP.keys())


def models_available() -> bool:
    return (MODEL_DIR / "lgb_model.pkl").exists() and \
           (MODEL_DIR / "preprocessor.pkl").exists()


def linear_models_available() -> bool:
    return (MODEL_DIR / "linear_model.pkl").exists() and \
           (MODEL_DIR / "ridge_model.pkl").exists()


def shap_available() -> bool:
    return (MODEL_DIR / "lgb_shap_vals.npy").exists()


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    import joblib
    lgb  = joblib.load(MODEL_DIR / "lgb_model.pkl")
    xgb  = joblib.load(MODEL_DIR / "xgb_model.pkl")
    prep = joblib.load(MODEL_DIR / "preprocessor.pkl")
    feat = np.load(MODEL_DIR / "feature_names.npy", allow_pickle=True).tolist()
    return lgb, xgb, prep, feat


@st.cache_resource(show_spinner="Loading linear models…")
def load_linear_models():
    import joblib
    lr    = joblib.load(MODEL_DIR / "linear_model.pkl")
    ridge = joblib.load(MODEL_DIR / "ridge_model.pkl")
    return lr, ridge


@st.cache_data(show_spinner="Computing linear SHAP values…")
def compute_linear_shap(model_name: str):
    import shap
    lr, ridge = load_linear_models()
    X_shap = np.load(MODEL_DIR / "X_shap.npy")
    m = lr if model_name == "Linear" else ridge
    exp = shap.LinearExplainer(m, X_shap)
    sv  = exp.shap_values(X_shap)
    return sv, float(exp.expected_value), m.predict(X_shap)


@st.cache_resource(show_spinner="Loading SHAP explainers…")
def load_explainers():
    import shap
    lgb, xgb, _, _ = load_models()
    lgb_exp = shap.TreeExplainer(lgb)
    xgb_exp = shap.TreeExplainer(xgb)
    return lgb_exp, xgb_exp


@st.cache_data(show_spinner="Loading SHAP artifacts…")
def load_shap_artifacts():
    lgb_sv        = np.load(MODEL_DIR / "lgb_shap_vals.npy")
    xgb_sv        = np.load(MODEL_DIR / "xgb_shap_vals.npy")
    X_shap        = np.load(MODEL_DIR / "X_shap.npy")
    y_shap        = np.load(MODEL_DIR / "y_shap.npy")
    lgb_pred_shap = np.load(MODEL_DIR / "lgb_pred_shap.npy")
    xgb_pred_shap = np.load(MODEL_DIR / "xgb_pred_shap.npy")
    lgb_base      = float(np.load(MODEL_DIR / "lgb_base.npy")[0])
    xgb_base      = float(np.load(MODEL_DIR / "xgb_base.npy")[0])
    return lgb_sv, xgb_sv, X_shap, y_shap, lgb_pred_shap, xgb_pred_shap, lgb_base, xgb_base


def load_model_metrics() -> dict:
    import json
    path = MODEL_DIR / "model_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def predict_tiers(source, destination, hour, is_weekend,
                  temperature, precip, wind, surge, preprocessor, lgb_model) -> pd.DataFrame:
    """Predict price for every service tier given route + conditions."""
    rows = [
        dict(distance=0, surge_multiplier=surge, hour=hour, is_weekend=int(is_weekend),
             temperature=temperature, precipProbability=precip, windSpeed=wind,
             name=name, cab_type=cab, source=source, destination=destination)
        for name, cab in SERVICE_MAP.items()
    ]
    df_pred = pd.DataFrame(rows)
    # distance filled later by caller
    X = preprocessor.transform(df_pred[NUM_FEATURES + CAT_FEATURES])
    df_pred["predicted_price"] = lgb_model.predict(X)
    return df_pred


def build_input_row(source, destination, hour, is_weekend,
                    temperature, precip, wind, surge, distance,
                    name, cab_type) -> pd.DataFrame:
    """Build a single prediction row for SHAP waterfall."""
    return pd.DataFrame([dict(
        distance=distance, surge_multiplier=surge, hour=hour,
        is_weekend=int(is_weekend), temperature=temperature,
        precipProbability=precip, windSpeed=wind,
        name=name, cab_type=cab_type, source=source, destination=destination,
    )])
