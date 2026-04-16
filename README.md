# Uber & Lyft Price Intelligence — Boston, MA

An end-to-end data science project that explores the Boston rideshare pricing landscape, trains supervised ML models to predict ride prices, and surfaces everything through an interactive multi-page Streamlit app with SHAP-powered explanations.

---

## What this project does

The app answers six analytical questions about rideshare pricing and pairs them with a live what-if prediction engine:

| Question | Where |
|---|---|
| Which platform is cheaper for the same route and conditions? | Ecosystem → Q1 |
| Which ride types are inherently expensive? | Ecosystem → Q2 |
| How much of price is distance vs surge vs product type? | Ecosystem → Q3 |
| Which routes are systematically overpriced? | Ecosystem → Q4 |
| How does weather shift price behavior? | Ecosystem → Q5 |
| Are there identifiable pricing regimes? | Ecosystem → Q6 |
| Given any route + conditions, what will Uber/Lyft cost? | Predictor |
| If I wait 1 hour, would price drop? | Predictor |
| Why did the model predict that price? | Explainer |

---

## Dataset

**Source:** [Uber and Lyft Dataset Boston, MA — Kaggle](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)

| Property | Value |
|---|---|
| File | `rideshare_kaggle.csv` |
| Raw rows | 693,071 |
| After cleaning (drop Uber Taxi NaN prices) | 637,976 |
| Columns | 57 |
| Date range | Nov – Dec 2018 |
| Neighborhoods | 12 (Boston) |
| Platforms | Uber, Lyft |
| Service tiers | 12 (6 per platform) |

Place the CSV at `data/rideshare_kaggle.csv` before running anything.

---

## Project structure

```
ml-price-pred-uber-lyft/
├── app/
│   ├── main.py                  # Landing page — status banners + navigation
│   ├── .streamlit/
│   │   └── config.toml          # Theme (colors, font)
│   ├── pages/
│   │   ├── 1_ecosystem.py       # EDA — 7-tab interactive analysis
│   │   ├── 2_predictor.py       # What-if price engine
│   │   └── 3_explainer.py       # SHAP explanation panel
│   └── utils/
│       ├── data.py              # Data loading, feature engineering, Q1–Q6 compute functions
│       └── model.py             # Model loading, prediction helpers, SHAP utilities
├── data/
│   └── rideshare_kaggle.csv     # Dataset (download separately)
├── model/                       # Saved artifacts (produced by 02_modeling.ipynb)
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   ├── linear_model.pkl
│   ├── ridge_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.npy
│   ├── model_metrics.json
│   ├── lgb_shap_vals.npy
│   ├── xgb_shap_vals.npy
│   ├── X_shap.npy  /  y_shap.npy
│   ├── lgb_pred_shap.npy  /  xgb_pred_shap.npy
│   ├── lgb_base.npy  /  xgb_base.npy
│   └── learning_curves.png
├── notebooks/
│   ├── notebookb28791c037.ipynb # Original Kaggle EDA notebook
│   └── 02_modeling.ipynb        # Feature engineering, training, SHAP, artifact export
├── requirements.txt
├── SETUP.md
└── README.md
```

---

## Setup

### 1 — Create the environment

**With conda (recommended):**
```bash
conda create -n ml-project-uber-lyft python=3.11 -y
conda activate ml-project-uber-lyft
pip install -r requirements.txt
```

**With venv:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First-time conda users: run `conda init zsh` then restart your terminal before `conda activate` will work.

### 2 — Register the Jupyter kernel

```bash
python -m ipykernel install --user \
  --name ml-project-uber-lyft \
  --display-name "Python (ml-project-uber-lyft)"
```

### 3 — Add the dataset

Download `rideshare_kaggle.csv` from Kaggle and place it at:
```
data/rideshare_kaggle.csv
```

### 4 — Train the models

```bash
jupyter notebook
```

Open and run **`notebooks/02_modeling.ipynb`** end-to-end. It will write all artifacts into `model/`.

### 5 — Launch the app

```bash
cd app
streamlit run main.py
```

Navigate to `http://localhost:8501`.

---

## App pages

### Landing page (`main.py`)
Status banners confirm whether the dataset, models, and SHAP artifacts are all present. Shows five dataset-level metrics and links to the three sections.

### Ecosystem (`1_ecosystem.py`)
Seven tabs, all driven by cached compute functions in `utils/data.py`:

| Tab | Content |
|---|---|
| Overview | Price histogram, Uber vs Lyft boxplot, ride count by tier |
| Q1 — Platform | Per-route price diff bar chart, source × destination heatmap, time-of-day breakdown |
| Q2 — Service Tiers | Price distribution boxplots per tier, median $/mile, IQR summary table |
| Q3 — Price Drivers | Standalone R² per factor, incremental R² waterfall, optional 3-D scatter with regression plane |
| Q4 — Routes | Route heatmap (avg price or $/mile), top-15 overpriced routes bar chart |
| Q5 — Weather | Avg price by weather condition × platform, temperature vs price scatter (LOWESS), precipitation vs price |
| Q6 — Regimes | KMeans (k=6) on hour/surge/weather/price, regime heatmap, avg price by regime, hour-of-day volume lines |

### Predictor (`2_predictor.py`)
Inputs: source, destination, hour, weekend toggle, surge multiplier, temperature, precipitation probability, wind speed.

Outputs for all 12 service tiers simultaneously:
- Horizontal bar chart of predicted prices
- "Wait 1 hour" delta (arrow + colour: green = cheaper, red = more expensive)
- Results table sortable by delta
- Natural-language insight: cheapest option and whether waiting saves money
- Historical price expander showing actual median prices for that route

Supports four models: LightGBM, XGBoost, Linear Regression, Ridge. Passes last input to the Explainer page via `st.session_state`.

### Explainer (`3_explainer.py`)
Five tabs:

| Tab | Content |
|---|---|
| Global Importance | Mean \|SHAP\| bar chart for every trained model side-by-side |
| Beeswarm | Per-sample SHAP scatter with red/blue feature-value colouring |
| Dependence | SHAP dependence plots for any continuous feature pair, LGB vs XGB side-by-side |
| Waterfall | Per-prediction breakdown — pick a test sample or use the last Predictor input |
| Model Comparison | MAE / RMSE / R² bar charts, LGB vs XGB feature importance agreement |

---

## Models and results

Features used:

| Type | Features |
|---|---|
| Numerical | `distance`, `surge_multiplier`, `hour`, `is_weekend`, `temperature`, `precipProbability`, `windSpeed` |
| Categorical | `name` (service tier), `cab_type`, `source`, `destination` |

Test-set metrics:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LightGBM** | **$1.05** | **$1.62** | **0.9695** |
| XGBoost | $1.06 | $1.64 | 0.9689 |
| Linear Regression | $1.74 | $2.48 | 0.9287 |
| Ridge | $1.74 | $2.48 | 0.9287 |

LightGBM and XGBoost are nearly identical in accuracy. Linear models capture 92.9% of variance — demonstrating that most of the price signal is already linear, with the remaining ~4 pp gained by non-linear interactions (especially between product type and distance).

---

## Key findings

- **Platform:** Uber is cheaper on the majority of routes; the gap is small (~$1–2) for standard tiers but widens for premium tiers where Lyft Lux pricing exceeds Uber Black on several routes.
- **Service tiers:** Lyft Lux Black XL and Uber Black SUV anchor the top of the price range. Lyft Shared and UberPool are consistently the cheapest, often under $10.
- **Price drivers:** Product type alone explains ~85% of variance. Adding distance brings it to ~90%; surge and platform add marginal incremental R².
- **Routes:** Short cross-neighbourhood hops (e.g. Beacon Hill → West End) carry the highest $/mile. Longer routes to Boston University or Northeastern University are relatively efficient.
- **Weather:** Rain and overcast conditions correlate with modestly higher prices (+$1–3 on average). Temperature below 38 °F coincides with slightly elevated surge activity.
- **Pricing regimes (KMeans k=6):** Morning Commute, Evening Commute, Nightlife, Surge Heavy, Bad Weather, Standard — each with a distinct hour-of-day signature.

---

## Dependencies

```
numpy, pandas, scipy
scikit-learn, lightgbm, shap, joblib
streamlit, plotly, streamlit-extras
jupyter, notebook, ipykernel, altair
```

Full pinned versions: `requirements.txt`.
