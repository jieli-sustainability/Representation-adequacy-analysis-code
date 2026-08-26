"""
Robustness checks for RECS model training notebook.
Place/run after all four final models (rf_final, enet_final, xgb_final, nn_final)
and X_train / y_train are defined.

Covers:
  Comment 3a — 5-fold CV metrics  (→ Table 2)
  Comment 3b — Bootstrap CI on underlying-feature importance  (→ Table 3)
  Comment 5  — VIF on retained numeric features  (→ Appendix E)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
N_JOBS = -1
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT_OVERRIDE", "./01_RECS_national"))
DATA_DIR     = PROJECT_ROOT / "03_Outcomes" / "01-Processed_Datasets"
MODEL_DIR    = PROJECT_ROOT / "03_Outcomes" / "02-Models"
SHAP_DIR     = PROJECT_ROOT / "03_Outcomes" / "04_shap"
IMPORTANCE_DIR = PROJECT_ROOT / "03_Outcomes" / "05_Importance"
IMP_CSV      = PROJECT_ROOT / "01_Data" / "Feature Importance" / "RECS_feature_preprocessed.csv"

IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

# ── Reconstruct final models from saved tuning logs ──────────────────────────
# (Skip this block if running inside the notebook where variables already exist)
with open(MODEL_DIR / "tuning_logs.json") as f:
    logs = json.load(f)

X_train_df = pd.read_csv(DATA_DIR / "X_train_scaled.csv").drop(columns=["DOEID"], errors="ignore")
y_train    = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].to_numpy(dtype=np.float32)
X_train    = X_train_df.to_numpy(dtype=np.float32, copy=True)

rf_final   = RandomForestRegressor(
    **logs["RF"]["best_params"],
    n_estimators=800, n_jobs=N_JOBS, random_state=SEED, bootstrap=True, max_samples=0.8)
enet_final = ElasticNet(
    **logs["ElasticNet"]["best_params"],
    random_state=SEED, max_iter=5_000, selection="random")
xgb_final  = xgb.XGBRegressor(
    objective="reg:squarederror", tree_method="hist", max_bin=256,
    n_estimators=max(50, int(logs["XGBoost"]["best_iteration"])),
    learning_rate=0.05, n_jobs=N_JOBS, random_state=SEED,
    **logs["XGBoost"]["best_params"])
nn_final   = MLPRegressor(
    **logs["NN"]["best_params"],
    random_state=SEED, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, max_iter=600, verbose=False)

# ── Helpers ───────────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "R2": float(r2_score(y_true, y_pred))}

# ═════════════════════════════════════════════════════════════════════════════
# COMMENT 3a — 5-fold cross-validation
# ═════════════════════════════════════════════════════════════════════════════
def kfold_cv_metrics(model, X, y, k=5, seed=SEED):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for tr, te in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        rows.append(regression_metrics(y[te], m.predict(X[te])))
    df = pd.DataFrame(rows)
    return df.mean(), df.std()

cv_rows = {}
for name, model in {"RF": rf_final, "ElasticNet": enet_final,
                    "XGBoost": xgb_final, "NN": nn_final}.items():
    mean_m, std_m = kfold_cv_metrics(model, X_train, y_train, k=5)
    cv_rows[name] = ({f"{k}_mean": mean_m[k] for k in mean_m.index} |
                     {f"{k}_sd":   std_m[k]  for k in std_m.index})
    print(f"{name}: R2={mean_m['R2']:.4f} ± {std_m['R2']:.4f}")

cv_df = pd.DataFrame(cv_rows).T
cv_df.to_csv(MODEL_DIR / "kfold_cv_metrics.csv")
print("\n[3a] kfold_cv_metrics.csv saved")
print(cv_df.to_string(float_format=lambda x: f"{x:.4f}"))

# ═════════════════════════════════════════════════════════════════════════════
# COMMENT 3b — Bootstrap CI on underlying-feature importance
# ═════════════════════════════════════════════════════════════════════════════
df_imp = pd.read_csv(IMP_CSV)
df_imp["importance"] = pd.to_numeric(df_imp["importance"], errors="coerce")
df_imp = df_imp.dropna(subset=["importance"])

shap_values      = np.load(SHAP_DIR / "xgb_shap_values.npy")
with open(MODEL_DIR / "features.json") as f:
    raw_feature_names = json.load(f)
feat_to_underlying = df_imp.set_index("feature")["Underlying factor"].to_dict()

def bootstrap_underlying_importance(shap_values, raw_feature_names,
                                    feat_to_underlying, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed)
    n    = shap_values.shape[0]
    raw_df = pd.DataFrame(shap_values, columns=raw_feature_names)
    mapped = [c for c in raw_df.columns if c in feat_to_underlying]
    raw_df = raw_df[mapped]
    labels = [feat_to_underlying[c] for c in mapped]
    boot   = []
    for _ in range(n_boot):
        idx      = rng.integers(0, n, n)
        mean_abs = raw_df.iloc[idx].abs().mean(axis=0).to_numpy()
        agg      = pd.Series(mean_abs, index=labels).groupby(level=0).sum()
        boot.append(100 * agg / agg.sum())
    boot_df = pd.DataFrame(boot)
    return pd.DataFrame({"mean_pct": boot_df.mean(),
                         "ci_lower": boot_df.quantile(0.025),
                         "ci_upper": boot_df.quantile(0.975)
                         }).sort_values("mean_pct", ascending=False)

feature_ci_df = bootstrap_underlying_importance(
    shap_values, raw_feature_names, feat_to_underlying, n_boot=1000)
feature_ci_df.to_csv(IMPORTANCE_DIR / "underlying_feature_bootstrap_ci.csv")
print("\n[3b] underlying_feature_bootstrap_ci.csv saved")
print(feature_ci_df.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

# ═════════════════════════════════════════════════════════════════════════════
# COMMENT 5 — VIF on retained numeric features
# ═════════════════════════════════════════════════════════════════════════════
RECS_NUMERIC_RETAINED = [
    "SQFTEST", "TOTROOMS", "HDD65", "HDD30YR_PUB",
    "DBT99", "YEARMADERANGE", "WINDOWS", "NHSLDMEM",
]

X_full = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
X_vif  = X_full[RECS_NUMERIC_RETAINED].dropna().assign(const=1)
vif    = pd.Series(
    [variance_inflation_factor(X_vif.values, i) for i in range(len(RECS_NUMERIC_RETAINED))],
    index=RECS_NUMERIC_RETAINED).sort_values(ascending=False)

vif.to_csv(IMPORTANCE_DIR / "vif_recs.csv")
print("\n[5] vif_recs.csv saved")
print(vif.to_string(float_format=lambda x: f"{x:.2f}"))
