# Version 10: 5-Fold Cross-Validation Ensemble
# Scores: Public 0.92357 | Private 0.92284
# Runtime: 1 hour 23 minutes
# Framework: CatBoost + LightGBM + XGBoost + Logistic Regression Meta-Model

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

TARGET = "loan_paid_back"
ID = "id"

y = train[TARGET].values
train_idx = train.index.values
test_ids = test[ID].values

# Prepare feature tables
X_cb = train.drop([TARGET, ID], axis=1).copy()   # for CatBoost (keep original dtype for categorical)
X_cb_test = test.drop(ID, axis=1).copy()

X_num = X_cb.copy()  # for LGB/XGB (will label-encode categoricals)
X_num_test = X_cb_test.copy()

# Identify categorical columns
cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()

# Label encode for LGB/XGB
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_num[c], X_num_test[c]], axis=0).astype(str)
    le.fit(combined)
    X_num[c] = le.transform(X_num[c].astype(str))
    X_num_test[c] = le.transform(X_num_test[c].astype(str))
    label_encoders[c] = le

# Convert to numpy for speed when needed
X_num_values = X_num.values
X_num_test_values = X_num_test.values

# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF and test predictions containers
oof_cat = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))

test_pred_cat = np.zeros(len(test))
test_pred_lgb = np.zeros(len(test))
test_pred_xgb = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num_values, y), 1):
    # Prepare fold data
    X_tr_num, X_val_num = X_num.iloc[tr_idx], X_num.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    X_tr_num_vals = X_tr_num.values
    X_val_num_vals = X_val_num.values

    # CatBoost (use original string categories)
    X_tr_cb = X_cb.iloc[tr_idx]
    X_val_cb = X_cb.iloc[val_idx]

    cb_model = CatBoostClassifier(
        iterations=1800,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        eval_metric="AUC",
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), cat_features=cat_cols)
    oof_cat[val_idx] = cb_model.predict_proba(X_val_cb)[:, 1]
    test_pred_cat += cb_model.predict_proba(X_cb_test)[:, 1] / n_splits

    # LightGBM (no early stopping - fixed rounds)
    lgb_train = lgb.Dataset(X_tr_num_vals, label=y_tr)
    params_lgb = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 4,
        "seed": 42,
        "verbose": -1
    }
    lgb_model = lgb.train(params_lgb, lgb_train, num_boost_round=1200)
    oof_lgb[val_idx] = lgb_model.predict(X_val_num_vals)
    test_pred_lgb += lgb_model.predict(X_num_test_values) / n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_tr_num_vals, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val_num_vals)[:, 1]
    test_pred_xgb += xgb_model.predict_proba(X_num_test_values)[:, 1] / n_splits

    # Optional: quick fold AUC prints
    fold_auc_cat = roc_auc_score(y_val, oof_cat[val_idx])
    fold_auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    fold_stack_auc = roc_auc_score(y_val, (oof_cat[val_idx] + oof_lgb[val_idx] + oof_xgb[val_idx]) / 3)
    print(f"Fold {fold} AUCs -> Cat: {fold_auc_cat:.5f}, LGB: {fold_auc_lgb:.5f}, XGB: {fold_auc_xgb:.5f}, Blend: {fold_stack_auc:.5f}")

# Overall OOF AUCs
print("OOF AUCs:",
      f"Cat {roc_auc_score(y, oof_cat):.5f}",
      f"LGB {roc_auc_score(y, oof_lgb):.5f}",
      f"XGB {roc_auc_score(y, oof_xgb):.5f}",
      f"Blend {(roc_auc_score(y, (oof_cat + oof_lgb + oof_xgb) / 3)):.5f}")

# Prepare level-1 training data (stacking)
X_meta_train = np.vstack([oof_cat, oof_lgb, oof_xgb]).T
X_meta_test = np.vstack([test_pred_cat, test_pred_lgb, test_pred_xgb]).T

# Meta-model (Logistic Regression)
meta_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
meta_model.fit(X_meta_train, y)
meta_oof_pred = meta_model.predict_proba(X_meta_train)[:, 1]
print("Meta OOF AUC:", roc_auc_score(y, meta_oof_pred))

# Final test predictions
final_test_pred = meta_model.predict_proba(X_meta_test)[:, 1]

# Save submission (Kaggle expects submission.csv)
submission = sample.copy()
submission[TARGET] = final_test_pred
submission.to_csv("submission.csv", index=False)
print("submission.csv created")
