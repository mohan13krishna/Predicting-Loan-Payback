# Version 20: AutoGluon Optimized (WINNER)
# Scores: Public 0.92380 | Private 0.92304
# Runtime: 4 hours 9 minutes
# Framework: AutoGluon TabularPredictor (best_quality, extended time_limit=3600+)
# RANK: 1255 / 3724 (Top 33.7%)

import pandas as pd
from autogluon.tabular import TabularPredictor

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

TARGET = "loan_paid_back"

print("=" * 80)
print("VERSION 20: AutoGluon Optimized Ensemble (WINNING MODEL)")
print("=" * 80)
print(f"\nDataset Info:")
print(f"  Training samples: {len(train):,}")
print(f"  Test samples: {len(test):,}")
print(f"  Features: {len(train.columns) - 2}")
print(f"  Target: {TARGET}")

print("\n" + "=" * 80)
print("Training AutoGluon with optimal settings...")
print("=" * 80)

# Train AutoGluon with EXTENDED time for best quality
predictor = TabularPredictor(
    label=TARGET,
    eval_metric="roc_auc",
    problem_type="binary"
).fit(
    train,
    presets="best_quality",
    time_limit=3600,  # Extended training time
    dynamic_stacking=True,  # Enable dynamic stacking detection
    num_stack_levels=2  # Multi-level stacking
)

print("\n" + "=" * 80)
print("Model Training Complete!")
print("=" * 80)

# Get model leaderboard
leaderboard = predictor.leaderboard(test, silent=True)
print("\nTop 10 Models:")
print(leaderboard.head(10)[['model', 'score_test', 'pred_time_test']])

# Make predictions
print("\nGenerating test predictions...")
preds = predictor.predict_proba(test)[1]  # probability of positive class

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": preds
})

submission.to_csv("/kaggle/working/submission.csv", index=False)

print("\n" + "=" * 80)
print("SUBMISSION COMPLETE!")
print("=" * 80)
print(f"✅ Submission file saved: submission.csv")
print(f"📊 Expected Public Score:  0.92380")
print(f"📊 Expected Private Score: 0.92304")
print(f"🏆 Expected Rank: 1255 / 3724 (Top 33.7%)")
print(f"⏱️  Total Runtime: 4 hours 9 minutes")
print("=" * 80 + "\n")

# Show prediction statistics
print("Prediction Statistics:")
print(f"  Mean predicted probability: {preds.mean():.6f}")
print(f"  Min predicted probability:  {preds.min():.6f}")
print(f"  Max predicted probability:  {preds.max():.6f}")
print(f"  Std dev:                    {preds.std():.6f}")
