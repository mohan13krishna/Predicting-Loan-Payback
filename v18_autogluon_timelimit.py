# Version 18: AutoGluon with Time-Limited Training
# Scores: Public 0.92223 | Private 0.92145
# Runtime: 1 hour 8 minutes
# Framework: AutoGluon TabularPredictor (best_quality preset, 3600s limit)

import pandas as pd
from autogluon.tabular import TabularPredictor

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

TARGET = "loan_paid_back"

print("Training AutoGluon with best_quality preset...")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Train AutoGluon with quality preset and time limit
predictor = TabularPredictor(
    label=TARGET,
    eval_metric="roc_auc",
    problem_type="binary"
).fit(
    train,
    presets="best_quality",
    time_limit=3600  # 1 hour time limit for quality
)

print("\nMaking predictions on test set...")
# Predict on test - get probabilities for positive class
preds = predictor.predict_proba(test)[1]  # probability of class 1

# Create submission
print("Creating submission file...")
sub = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": preds
})

sub.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv saved!")
print(f"V18 Score: 0.92223 public, 0.92145 private")
print("\nAutogluon Model Summary:")
print(predictor.get_model_names()[:5])  # Show top 5 models
