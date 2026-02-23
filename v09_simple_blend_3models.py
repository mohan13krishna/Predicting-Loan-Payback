# Version 9: Simple 3-Model Blend
# Scores: Public 0.92290 | Private 0.92234
# Runtime: 3m 51s (FASTEST)
# Framework: CatBoost + LightGBM + XGBoost (Equal Weight: 1/3 each)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

y = train["loan_paid_back"]
X = train.drop(["loan_paid_back", "id"], axis=1)
test_X = test.drop("id", axis=1)

cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_X[col] = le.transform(test_X[col])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== CATBOOST =====
cat_model = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.02,
    l2_leaf_reg=3,
    eval_metric="AUC",
    verbose=0,
    random_seed=42
)
cat_model.fit(X_train, y_train)

# ===== LIGHTGBM =====
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 4,
    "verbose": -1,
    "seed": 42
}

train_lgb = lgb.Dataset(X_train, y_train)
model_lgb = lgb.train(lgb_params, train_lgb, num_boost_round=1500)

# ===== XGBOOST =====
xgb_model = xgb.XGBClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.85,
    eval_metric="auc",
    random_state=42
)

xgb_model.fit(X_train, y_train)

# ===== SIMPLE BLEND (Equal Weight) =====
p_cat = cat_model.predict_proba(test_X)[:, 1]
p_lgb = model_lgb.predict(test_X)
p_xgb = xgb_model.predict_proba(test_X)[:, 1]

final_pred = (p_cat + p_lgb + p_xgb) / 3

# ===== SUBMISSION =====
submission = sample.copy()
submission["loan_paid_back"] = final_pred
submission.to_csv("submission.csv", index=False)
print("V9 submission.csv created - Score: 0.92290 public, 0.92234 private")
