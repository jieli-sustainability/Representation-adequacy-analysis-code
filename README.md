# Representation-Adequacy Analysis Code
Analysis code accompanying "Do Standardized Residential Energy Models Represent the Variables That Matter? Evidence from Explainable Machine Learning" (Jie Li, Guido Cervone, José P. Duarte, Ute Poerschke, Lisa D. Iulo), submitted to *Energy and AI*.

The study uses a standards-linked XAI/SHAP framework (XGBoost models, TreeSHAP attribution) applied to two national residential energy datasets, RECS and EULP/ResStock, to diagnose representation-sensitive priorities relevant to residential energy performance representation adequacy, mapped to ANSI/RESNET/ICC 301-2022. SHAP importance is diagnostic evidence about which inputs the models rely on most heavily, aggregated to underlying building/household features. It does not identify causal drivers or determinants of any outcome, and does not evaluate 301's specific equations.

All findings are framed accordingly throughout (see manuscript §2.4 for the full framing).

# Repository structure
```
.
├── modeling/
│   ├── recs/
│   │   ├── 01\_preprocess\_recs.ipynb          # cleaning, encoding, 
│   │   ├── 02\_train\_models\_recs.ipynb        # RF / ElasticNet / XGBoost /NN
│   │   └── 03\_importance\_recs.ipynb          # SHAP computation and 
│   └── eulp/
│       ├── 01\_preprocess\_eulp.ipynb
│       ├── 02\_train\_models\_eulp.ipynb
│       └── 03\_importance\_eulp.ipynb
├── robustness/
│   ├── run\_robustness\_recs.py    # 5-fold CV, bootstrap CI, VIF  (RECS)
│   └── run\_robustness\_eulp.py    # 5-fold CV, bootstrap CI, VIF  (EULP)
├── requirements.txt
├── LICENSE
└── CITATION.cff
```
# Paper-to-code mapping
Manuscript element	Source
§2.1–2.2 preprocessing, feature encoding	`modeling/{recs,eulp}/01\_preprocess\_\*.ipynb`
§2.3 model training (RF, ElasticNet, XGBoost, NN)	`modeling/{recs,eulp}/02\_train\_models\_\*.ipynb`
§2.3–3.2 SHAP attribution, aggregation to underlying features, Tables 3–5 (representation-sensitive priorities, Strong/Partial/Conditional tiers)	`modeling/{recs,eulp}/03\_importance\_\*.ipynb`
§3.1 k-fold CV (Table 2), bootstrap CIs (Table 3), VIF diagnostics (Appendix E)	`robustness/run\_robustness\_{recs,eulp}.py`
The robustness scripts are meant to be run after the corresponding
`02\_train\_models\_\*` notebook, using the same trained model objects (or, if running standalone, they reconstruct the final models from
`tuning\_logs.json`, exactly as documented in each script's docstring).
# Environment
```bash
pip install -r requirements.txt
```
Developed with Python 3.11. The package versions used for the archived
analysis are specified in `requirements.txt`. Minor differences in
scikit-learn, XGBoost, or SHAP versions may produce small numerical
differences in cross-validation and bootstrap results.
# Data availability
This repository contains code only — no raw or processed household-level
data files are included. The datasets are public and can be obtained from:
RECS (Residential Energy Consumption Survey): U.S. Energy Information
Administration, https://www.eia.gov/consumption/residential/
EULP / ResStock (End-Use Load Profiles): National Renewable Energy
Laboratory, https://www.nrel.gov/buildings/end-use-load-profiles
Place the raw datasets in the corresponding project data directories
before running the preprocessing notebooks. For EULP, the expected input
is `01\_Data/eulp.csv`. See `01\_preprocess\_recs.ipynb` for the required
RECS input filename and structure.
# Diagnostic, not causal — and a note on terminology
SHAP importance values describe how much each feature contributes to a
model's prediction, not how much it causes the outcome. Throughout this
repository and the associated manuscript, the retained/aggregated features
are referred to as representation-sensitive priorities — i.e., inputs
whose representation in a dataset or rating engine most strongly shapes
model output — rather than as "drivers" or "determinants," since those terms
imply a causal relationship that SHAP attribution does not establish.
Citation
If you use this code, please cite both the archived software release and the associated article.
License
MIT — see `LICENSE`.
