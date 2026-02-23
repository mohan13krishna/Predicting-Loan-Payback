# 📊 Version History & Performance Tracking

## 🏆 Competition Results: Playground Series - Season 5, Episode 11

### 🥇 Best Performing Solutions

| Version | Approach | Public Score | Private Score | Runtime | Status | Key Features |
|---------|----------|--------------|---------------|---------|--------|--------------|
| **v20** | **AutoGluon Optimized** | **0.92380** | **0.92304** | 4h 9m | ✅ **WINNER** | Time-limited AutoML, best_quality preset, dynamic stacking |
| v18 | AutoGluon + Analysis | 0.92223 | 0.92145 | 1h 8m | ✅ Passed | Focus on model inspection & feature importance |
| v10 | 5-Fold CV Ensemble | 0.92357 | 0.92284 | 1h 23m | ✅ Passed | CatBoost + LightGBM + XGBoost, Logistic Regression meta-model |
| v9 | Simple Blend | 0.92290 | 0.92234 | 3m 51s | ✅ Passed | Equal-weight ensemble of 3 models |
| v7 | CatBoost Stratified | 0.92189 | 0.92138 | 26m 12s | ✅ Passed | Single strong CatBoost model |

---

## 📝 Detailed Version Breakdown

### ✅ TOP TIER (0.922+)

#### **V20: AutoGluon Optimized** 🏆
- **Scores:** Public 0.92380 | Private 0.92304
- **Runtime:** 4 hours 9 minutes
- **Framework:** AutoGluon TabularPredictor
- **Strategy:** 
  - Time limit: 3600+ seconds per model
  - Presets: best_quality
  - Dynamic stacking enabled
  - Automatic hyperparameter tuning
  - 2-level stacking (L1-L2-L3)
  - WeightedEnsemble of LightGBM variants
- **Key Models:**
  - LightGBM_BAG_L2 (0.9223 validation AUC)
  - WeightedEnsemble_L3 (0.9221 validation AUC)
  - LightGBMXT_BAG_L2 (0.9212 validation AUC)
- **Why It Won:**
  - Fully automated hyperparameter search
  - Exploration of 110+ model configurations
  - Dynamic stacking detection prevents overfitting
  - Extended training time maximizes model quality

---

#### **V18: AutoGluon with Feature Analysis** 
- **Scores:** Public 0.92223 | Private 0.92145
- **Runtime:** 1 hour 8 minutes
- **Framework:** AutoGluon (optimized configuration)
- **Difference from V20:** Shorter time limit, more conservative settings
- **Status:** Strong alternative when time is limited
- **Lessons:** Even with reduced resources, AutoGluon performs competitively

---

#### **V10: 5-Fold Cross-Validation Ensemble** 
- **Scores:** Public 0.92357 | Private 0.92284
- **Runtime:** 1 hour 23 minutes
- **Framework:** CatBoost + LightGBM + XGBoost + Logistic Regression
- **Architecture:**
  ```
  Base Models (5-fold CV):
  ├── CatBoost (1800 iterations, depth=8, lr=0.03)
  ├── LightGBM (1200 rounds, num_leaves=48, lr=0.03)
  └── XGBoost (900 estimators, depth=6, lr=0.03)
  
  Meta-Model:
  └── Logistic Regression (5-fold OOF predictions)
  ```
- **Out-of-Fold (OOF) Strategy:**
  - Generates OOF predictions for all 3 base models
  - Uses OOF predictions as features for meta-model
  - Prevents data leakage through stratified k-fold
- **Fold Results:**
  - Fold 1 Blend AUC: 0.92336
  - Fold 2 Blend AUC: 0.92282
  - Fold 3 Blend AUC: 0.92132
  - Fold 4 Blend AUC: 0.92229
  - Fold 5 Blend AUC: 0.92181
  - Overall OOF AUC: 0.92232
- **Why It Performed Well:**
  - Diverse base models (gradient boosting + logistic regression)
  - Proper OOF stacking prevents overfitting
  - 5-fold CV ensures robust cross-model blending

---

### ⭐ HIGH TIER (0.920+)

#### **V9: Simple Equal-Weight Blend**
- **Scores:** Public 0.92290 | Private 0.92234
- **Runtime:** 3m 51s (FASTEST)
- **Framework:** CatBoost (2000 iter) + LightGBM (1500 rounds) + XGBoost (900 est)
- **Strategy:** Simple (1/3 + 1/3 + 1/3) averaging
- **Advantage:** Minimal complexity, quick execution, solid performance
- **Code Simplicity:** Easy to reproduce and modify

---

#### **V15: Heavy CatBoost Single Model**
- **Scores:** Public 0.92176 | Private 0.92108
- **Runtime:** 19m 24s
- **Framework:** CatBoost (2500 iterations) with hyperparameter tuning
- **Parameters:**
  - iterations: 2500
  - learning_rate: 0.017
  - depth: 8
  - l2_leaf_reg: 5
  - eval_metric: AUC
- **Why Slightly Lower:** Single model lacks ensemble diversity

---

#### **V7: CatBoost Stratified Split**
- **Scores:** Public 0.92189 | Private 0.92138
- **Runtime:** 26m 12s
- **Framework:** CatBoost with stratified train/valid split
- **Configuration:**
  - Iterations: 2500
  - Depth: 8
  - Learning Rate: 0.015
  - Stratified split (80/20)
- **Developer:** Rakesh Kolipaka

---

### 📊 MID TIER (0.914-0.919)

#### **V14: Random Forest Classifier**
- **Scores:** Public 0.91455 | Private 0.91421
- **Runtime:** 3m 3s
- **Framework:** scikit-learn RandomForest
- **Limitations:** Tree-based but non-boosting, lower capacity
- **Lesson:** Ensemble methods > single decision trees

---

#### **V3: CatBoost with Extended Training**
- **Scores:** Public 0.92243 | Private 0.92163
- **Runtime:** 38m 13s
- **Framework:** CatBoost (3000 iterations, depth=10)
- **Config:**
  - iterations: 3000
  - learning_rate: 0.015 (slower learning)
  - depth: 10
  - l2_leaf_reg: 3
- **Overfitting Note:** Deeper trees caused slight overfitting vs v7

---

#### **V2: CatBoost Standard**
- **Scores:** Public 0.92214 | Private 0.92146
- **Runtime:** 18m 7s
- **Framework:** CatBoost (2000 iterations)
- **Status:** Solid baseline model

---

#### **V1: Initial LightGBM**
- **Scores:** Public 0.92308 | Private 0.92212
- **Runtime:** 1m 34s
- **Framework:** LightGBM (1500 rounds)
- **Lesson:** Strong baseline, but ensemble > single model

---

### ❌ FAILED/EXPERIMENTAL VERSIONS

| Version | Approach | Issue | Lesson |
|---------|----------|-------|--------|
| v19 | Complex Ensemble | Failed after 49m | Time limit management critical |
| v16 | Intermediate blend | Cancelled | Redundant approach |
| v17 | Ensemble composition | 28s runtime | Incomplete implementation |
| v13, v12, v11, v6, v5, v4 | Various | Failed/Timeout | Insufficient resources or bugs |

---

## 🎯 Key Insights from Experimentation

### ✅ What Worked
1. **AutoML (AutoGluon)** - Automated search beats manual tuning
2. **Gradient Boosting** - XGBoost, LightGBM, CatBoost are superior
3. **Ensemble Methods** - Combining 3+ models > single model
4. **Extended Training** - More iterations/rounds = better performance (diminishing returns)
5. **Native Categorical Support** - CatBoost's optimized handling crucial
6. **5-Fold Stratified CV** - Proper validation prevents overfitting
7. **OOF Stacking** - Meta-learning improves predictions

### ❌ What Didn't Work
1. **Random Forest** - Lower AUC (0.914) vs boosting (0.923)
2. **Simple Train/Test Split** - Less robust than K-fold CV
3. **Equal Feature Importance** - Ensemble weights matter
4. **Time Pressure** - Interrupted training hurts models
5. **Shallow Trees** - Insufficient model capacity
6. **No Categorical Handling** - Feature type mismatch penalties

---

## 📈 Performance Timeline

```
v1  (0.9231) ──┐
v2  (0.9221)   ├─→ Baseline LightGBM/CatBoost
v3  (0.9224)   │
                │
v7  (0.9219) ──┤─→ Pure CatBoost focus
v8  (N/A)      │
v9  (0.9229)   ├─→ Simple 3-model blend
v10 (0.9236) ──┤─→ Sophisticated 5-fold + Logistic stack (HIGH)
                │
v14 (0.9146) ──┤─→ Random Forest attempt (WEAK)
v15 (0.9218)   │
                │
v16-v19 ────────┤─→ AutoGluon exploration
v18 (0.9222)   ├─→ AutoGluon (time-limited)
v20 (0.9238) ──┴─→ AutoGluon (optimal) ⭐ WINNER
```

---

## 🔬 Model Hyperparameters Comparison

### CatBoost Evolution
```
v2:  iterations=2000,  lr=0.02,  depth=8,  l2=3
v3:  iterations=3000,  lr=0.015, depth=10, l2=3   (OVERFITTED)
v7:  iterations=2500,  lr=0.015, depth=8,  l2=5   (OPTIMIZED)
v15: iterations=2500,  lr=0.017, depth=8,  l2=5   (BEST SINGLE)
```

### LightGBM Configuration
```
All versions:
  - learning_rate: 0.03
  - num_leaves: 31-48
  - feature_fraction: 0.85-0.90
  - bagging_fraction: 0.85-0.90
  - num_boost_round: 1200-1500
Result: Consistent ~0.922 AUC
```

### XGBoost Configuration
```
All versions:
  - n_estimators: 900
  - learning_rate: 0.03
  - max_depth: 6
  - subsample: 0.9
  - colsample_bytree: 0.85
Result: Weakest of the 3 (~0.9212)
```

---

## 💡 Recommendations for Future Competitions

1. **Start with AutoGluon** - Usually beats manual tuning
2. **Allocate 60%+ time for AutoML** - Quality > quantity of models
3. **Use 5-fold stratified CV** - Standard best practice
4. **Ensemble diverse models** - Different architectures matter
5. **Monitor train/validation curves** - Catch overfitting early
6. **Use OOF predictions for stacking** - Better meta-learning signal
7. **Log all experiments** - Track what works/fails
8. **Test before submitting** - Verify submission format matches requirements

---

## 📚 Competition Statistics

- **Total Versions Attempted:** 20
- **Successful Submissions:** 16
- **Failed/Incomplete:** 4
- **Best Score:** 0.92380 (v20)
- **Best AUC Improvement:** 0.9238 - 0.9146 = 0.0092 (+1.0%)
- **Execution Time Saved:** v9 vs v20 = 3m 51s vs 4h 9m (v9 = 1.6% inference time)
- **Final Rank:** 1255 / 3724 (Top 33.7%)

---

## 🎓 Team Learning Summary

**What Team Phoenix Learned:**
- AutoML frameworks are production-ready for competitions
- Ensemble diversity beats individual model strength
- Time management critical in fixed-duration competitions
- Categorical features deserve special handling
- Cross-validation prevents misleading validation scores
- Stacking with OOF predictions significantly improves results

---

**Last Updated:** February 23, 2026  
**Competition Status:** Late Submission Period  
**Repository:** [Predicting-Loan-Payback](https://github.com/mohan13krishna/Predicting-Loan-Payback)
