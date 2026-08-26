# Representation-Adequacy Analysis Code

Analysis code accompanying **"Do Standardized Residential Energy Models Represent the Variables That Matter? Evidence from Explainable Machine Learning"** (Jie Li a, Guido Cervone, José P. Duarte, Ute Poerschke, Lisa D. Iulo),submitted to *Energy and AI*.

The study uses a standards-linked XAI/SHAP framework (XGBoost
models, TreeSHAP attribution) applied to two national residential energy
datasets, **RECS** and **EULP/ResStock**, to diagnose representation-sensitive
priorities relevant to residential energy performance representation adequacy,
mapped to ANSI/RESNET/ICC 301-2022. SHAP importance is diagnostic evidence
about which inputs the models rely on most heavily, aggregated to underlying
building/household features. It does not identify causal drivers or
determinants of any outcome, and does not evaluate 301's specific equations.
All findings are framed accordingly throughout (see manuscript §2.4 for the
full framing).

## Repository structure

```
.
├── modeling/
│   ├── recs/
│   │   ├── 01\\\\\\\_preprocess\\\\\\\_recs.ipynb          # cleaning, encoding, 
│   │   ├── 02\\\\\\\_train\\\\\\\_models\\\\\\\_recs.ipynb        # RF / ElasticNet /  
│   │   └── 03\\\\\\\_importance\\\\\\\_recs.ipynb          # SHAP computation and 
│   └── eulp/
│       ├── 01\\\\\\\_preprocess\\\\\\\_eulp.ipynb
│       ├── 02\\\\\\\_train\\\\\\\_models\\\\\\\_eulp\\\\\\\_optimized.ipynb
│       └── 03\\\\\\\_importance\\\\\\\_eulp.ipynb
├── 5-fold CV/
│   ├── run\\\\\\\_robustness\\\\\\\_recs.py    # 5-fold CV, bootstrap CI, VIF  (RECS)
│   └── run\\\\\\\_robustness\\\\\\\_eulp.py    # 5-fold CV, bootstrap CI, VIF  (EULP)
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Paper-to-code mapping

|Manuscript element|Source|
|-|-|
|§2.1–2.2 preprocessing, feature encoding|`modeling/{recs,eulp}/01\\\\\\\_preprocess\\\\\\\_\\\\\\\*.ipynb`|
|§2.3 model training (RF, ElasticNet, XGBoost, NN)|`modeling/{recs,eulp}/02\\\\\\\_train\\\\\\\_models\\\\\\\_\\\\\\\*.ipynb`|
|§2.3–3.2 SHAP attribution, aggregation to underlying features, Tables 3–5 (representation-sensitive priorities, Strong/Partial/Conditional tiers)|`modeling/{recs,eulp}/03\\\\\\\_importance\\\\\\\_\\\\\\\*.ipynb`|
|§3.1 k-fold CV (Table 2), bootstrap CIs (Table 3), VIF diagnostics (Appendix E)|`5-fold CV/run\\\\\\\_robustness\\\\\\\_{recs,eulp}.py`|

The robustness scripts are meant to be run **after** the corresponding
`02\\\\\\\_train\\\\\\\_models\\\\\\\_\\\\\\\*` notebook, using the same trained model objects (or, if running standalone, they reconstruct the final models from
`tuning\\\\\\\_logs.json`, exactly as documented in each script's docstring).

## Environment

```bash
pip install -r requirements.txt
```

Developed with Python 3.11. Package versions were not pinned during the
original runs; if you need exact reproducibility of the k-fold CV and
bootstrap results down to the last digit, pin `scikit-learn`, `xgboost`, and
`shap` to the versions noted in your own environment, since minor version
changes in these libraries can shift results at the 3rd–4th decimal place.

By default, each notebook/script uses relative project directories
(`./01\\\\\\\_RECS\\\\\\\_national`, `./02\\\\\\\_EULP\\\\\\\_national`) for inputs/outputs. The
`run\\\\\\\_robustness\\\\\\\_\\\\\\\*.py` scripts also accept a `PROJECT\\\\\\\_ROOT\\\\\\\_OVERRIDE`
environment variable if you want to point them at a different location:

```bash
PROJECT\\\\\\\_ROOT\\\\\\\_OVERRIDE=/path/to/01\\\\\\\_RECS\\\\\\\_national python scripts/run\\\\\\\_robustness\\\\\\\_recs.py
```

## Data availability

This repository contains **code only** — no raw or processed household-level
data files are included. The datasets are public and can be obtained from:

* **RECS** (Residential Energy Consumption Survey): U.S. Energy Information
Administration, https://www.eia.gov/consumption/residential/
* **EULP / ResStock** (End-Use Load Profiles): National Renewable Energy
Laboratory, https://www.nrel.gov/buildings/end-use-load-profiles

Place the raw files as `01\\\\\\\_Data/eulp.csv` (EULP) or the equivalent RECS input
under each project root before running `01\\\\\\\_preprocess\\\\\\\_\\\\\\\*`.

## Diagnostic, not causal — and a note on terminology

SHAP importance values describe how much each feature contributes to a
model's *prediction*, not how much it *causes* the outcome. Throughout this
repository and the associated manuscript, the retained/aggregated features
are referred to as **representation-sensitive priorities** — i.e., inputs
whose representation in a dataset or rating engine most strongly shapes
model output — rather than as "drivers" or "determinants," since those terms
imply a causal relationship that SHAP attribution does not establish.

## Citation

If you use this code, please cite the associated article (details will be
added on acceptance of the paper) and/or this repository via `CITATION.cff`.

## License

MIT — see `LICENSE`.

