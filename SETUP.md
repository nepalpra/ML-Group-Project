# Setup Guide

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.11
- ~2 GB free disk space (dataset + model artifacts)

---

## 1. Create the conda environment

```bash
conda create -n ml-project-uber-lyft python=3.11 -y
conda activate ml-project-uber-lyft
pip install -r requirements.txt
```

---

## 2. Add the dataset

Download `rideshare_kaggle.csv` from [Kaggle](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma) and place it at:

```
data/rideshare_kaggle.csv
```

The file is ~150 MB (693,071 rides). After loading, `load_data()` drops rows with missing prices → 637,976 rows.

---

## 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user \
  --name ml-project-uber-lyft \
  --display-name "Python (ml-project-uber-lyft)"
```

---

## 4. Train the models (run notebooks in order)

```bash
jupyter notebook
```

Open and run both notebooks sequentially:

| Notebook | Purpose | Outputs |
|---|---|---|
| `notebooks/notebookb28791c037.ipynb` | EDA — 6 analytical questions | None (exploration only) |
| `notebooks/02_modeling.ipynb` | LightGBM + XGBoost pipeline | All core artifacts in `model/` |

Core artifacts saved to `model/` after `02_modeling.ipynb`:

```
model/
  lgb_model.pkl          # LightGBM regressor
  xgb_model.pkl          # XGBoost regressor
  preprocessor.pkl       # sklearn ColumnTransformer
  feature_names.npy      # encoded feature names (45 features)
  lgb_shap_vals.npy      # SHAP values — LightGBM
  xgb_shap_vals.npy      # SHAP values — XGBoost
  X_shap.npy             # input matrix used for SHAP
  y_shap.npy             # true prices for SHAP sample
  lgb_base.npy           # SHAP expected values — LightGBM
  xgb_base.npy           # SHAP expected values — XGBoost
  model_metrics.json     # MAE / RMSE / R² for all models
  linear_model.pkl       # Linear regression (optional)
  ridge_model.pkl        # Ridge regression (optional)
```

---

## 5. Generate advanced artifacts (optional, for pages 5–8)

Each script is independent. Run any or all:

```bash
# Advanced SHAP: interaction values, conditional SHAP, within-tier SHAP, stability
python scripts/compute_advanced_shap.py

# Model Zoo: quantile regression, MLP, time-split validation, learning curves
python scripts/train_advanced_models.py

# Demand & Supply: surge classifier + surge magnitude regressor
python scripts/train_surge_model.py

# HPO: Optuna Bayesian search (30 trials each for LGB + XGB)
python scripts/train_tuned_models.py --trials 30
```

Additional artifacts saved to `model/` after these scripts:

```
model/
  advanced_metrics.json      # per-platform MAE, SHAP stability
  shap_interactions.npy      # pairwise SHAP interaction values
  shap_conditional.pkl       # conditional SHAP by surge regime
  shap_within_tier.pkl       # SHAP values split by service tier
  shap_stability.pkl         # bootstrap SHAP stability estimates
  shap_beeswarm_lgb.png      # beeswarm plot (static export)
  lgb_q10/q50/q90.pkl        # quantile regression models
  mlp_model.pkl              # MLP neural network
  time_split_metrics.json    # Nov → Dec temporal validation
  learning_curves.png        # learning curve plot
  tuned_lgb_model.pkl        # Optuna-tuned LightGBM
  tuned_xgb_model.pkl        # Optuna-tuned XGBoost
  hpo_results.json           # full Optuna trial history
  surge_classifier.pkl       # LGB surge binary classifier
  surge_regressor.pkl        # LGB surge magnitude regressor
  surge_preprocessor.pkl     # preprocessor for surge models
  surge_metrics.json         # ROC-AUC, avg precision, calibration
  demand_agg.pkl             # pre-aggregated demand heatmaps
```

---

## 6. Launch the app

```bash
cd app
streamlit run main.py
```

The app opens at `http://localhost:8501`. Pages that require missing artifacts display a warning banner and degrade gracefully — you do not need all artifacts to run the app.

---

## App pages and their requirements

| Page | File | Requires |
|---|---|---|
| Main dashboard | `main.py` | Nothing (status auto-detected) |
| ① Ecosystem (EDA) | `pages/1_ecosystem.py` | Dataset only |
| ② Price Predictor | `pages/2_predictor.py` | Core model artifacts |
| ③ Explainer (SHAP) | `pages/3_explainer.py` | Core model + SHAP artifacts |
| ⑤ Advanced SHAP | `pages/5_advanced_shap.py` | `compute_advanced_shap.py` output |
| ⑥ Model Zoo | `pages/6_model_zoo.py` | `train_advanced_models.py` output |
| ⑦ Demand & Supply | `pages/7_demand_supply.py` | Dataset (+ optional `train_surge_model.py`) |
| ⑧ HPO | `pages/8_hpo.py` | Core models (+ optional `train_tuned_models.py`) |

---

## Updating dependencies

```bash
pip install -r requirements.txt --upgrade
```

---

## Troubleshooting

**`ModuleNotFoundError` when running scripts directly**
Always run scripts inside the activated conda environment:
```bash
conda activate ml-project-uber-lyft
python scripts/train_surge_model.py
```

**Streamlit can't find `utils.*`**
Each page file inserts its parent directory into `sys.path`. If running a page directly (not via `streamlit run main.py`), ensure you launch from the `app/` directory.

**`KeyError` on model metrics keys**
`model_metrics.json` uses keys `'LightGBM'`, `'XGBoost'`, `'Linear'`, `'Ridge'` — not lowercase abbreviations.

**Altair charts fail in notebooks**
`vegafusion` is not installed. Cells 0–23 in the EDA notebook use Altair for small aggregates; cells 24+ use Plotly. This is expected behavior.
