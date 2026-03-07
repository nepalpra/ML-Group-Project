# Environment Setup

## 1. Create the conda environment

```bash
conda create -n ml-project-uber-lyft python=3.11 -y
conda activate ml-project-uber-lyft
pip install -r requirements.txt
```

## 2. Register the kernel for Jupyter

```bash
python -m ipykernel install --user --name ml-project-uber-lyft --display-name "Python (ml-project-uber-lyft)"
```

## 3. Add the dataset

Download `rideshare_kaggle.csv` and place it at:
```
data/rideshare_kaggle.csv
```

## 4. Run the notebooks (in order)

```bash
jupyter notebook
```

Open and run:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_modeling.ipynb`

## 5. Launch the app

```bash
cd app
streamlit run main.py
```

## Updating dependencies

```bash
pip install -r requirements.txt --upgrade
```
