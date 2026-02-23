# Version Codes Summary - All Submissions with Scores
# Predicting Loan Payback - Kaggle Playground Series Season 5, Episode 11

## 🏆 BEST PERFORMING (Top 5)

### V20: AutoGluon Optimized (WINNER) ⭐⭐⭐
- **Public Score:** 0.92380
- **Private Score:** 0.92304
- **Rank:** 1255 / 3724 (Top 33.7%)
- **Runtime:** 4 hours 9 minutes
- **File:** `v20_autogluon_winner.py`
- **Key Features:**
  - Extended time_limit (3600+ seconds)
  - Dynamic stacking enabled
  - 2-level ensemble (L1-L2-L3)
  - WeightedEnsemble optimization
- **Why It Won:** Automated hyperparameter search + longer training = best quality

---

### V18: AutoGluon Time-Limited
- **Public Score:** 0.92223
- **Private Score:** 0.92145
- **Runtime:** 1 hour 8 minutes
- **File:** `v18_autogluon_timelimit.py`
- **Key Features:**
  - 1-hour time limit
  - Best quality preset
  - Faster inference (good for quick iterations)
- **Lesson:** AutoGluon strong even with reduced resources

---

### V10: 5-Fold Cross-Validation Ensemble
- **Public Score:** 0.92357
- **Private Score:** 0.92284
- **Runtime:** 1 hour 23 minutes
- **File:** `v10_ensemble_5fold_cv.py`
- **Architecture:**
  - Base Models: CatBoost (1800 iter) + LightGBM (1200 rounds) + XGBoost (900 est)
  - Meta-Model: Logistic Regression
  - Strategy: Out-of-Fold (OOF) predictions for stacking
- **Fold Results:**
  - Fold 1-5 Blend AUC: 0.9223-0.9328
  - Overall OOF AUC: 0.92232
- **Lesson:** OOF stacking + diverse base models = competitive results

---

### V9: Simple 3-Model Blend (Fastest)
- **Public Score:** 0.92290
- **Private Score:** 0.92234
- **Runtime:** 3m 51s
- **File:** `v09_simple_blend_3models.py`
- **Architecture:**
  - CatBoost (2000 iter) + LightGBM (1500 rounds) + XGBoost (900 est)
  - Simple equal-weight averaging (1/3 each)
- **Advantage:** Speed + simplicity + solid score
- **Lesson:** Simple ensemble surprisingly effective

---

## 📊 All Other Versions

| Version | Public | Private | Type | Runtime | Status |
|---------|--------|---------|------|---------|--------|
| v20 | 0.92380 | 0.92304 | AutoGluon Opt | 4h 9m | ✅ WINNER |
| v18 | 0.92223 | 0.92145 | AutoGluon | 1h 8m | ✅ PASS |
| v15 | 0.92176 | 0.92108 | CatBoost Heavy | 19m | ✅ PASS |
| v14 | 0.91455 | 0.91421 | Random Forest | 3m | ✅ PASS |
| v10 | 0.92357 | 0.92284 | 5-Fold Ensemble | 1h 23m | ✅ PASS |
| v9 | 0.92290 | 0.92234 | Simple Blend | 3m 51s | ✅ PASS |
| v8 | N/A | N/A | Ensemble | - | ❌ ERROR |
| v7 | 0.92189 | 0.92138 | CatBoost | 26m | ✅ PASS |
| v3 | 0.92243 | 0.92163 | CatBoost+ | 38m | ✅ PASS |
| v2 | 0.92214 | 0.92146 | CatBoost | 18m | ✅ PASS |
| v1 | 0.92308 | 0.92212 | LightGBM | 1m 34s | ✅ PASS |

---

## 🎯 How to Use Version Files

### Run V20 (Best) on Kaggle:
```python
# Copy v20_autogluon_winner.py code into notebook
# Or upload as notebook and run directly
# Expected: 0.92380 public, 0.92304 private score
```

### Run V10 (Best Manual Ensemble) on Kaggle:
```python
# Copy v10_ensemble_5fold_cv.py code into notebook
# Executes 5-fold CV with 3 base models + meta-learner
# Expected: 0.92357 public, 0.92284 private score
```

### Run V9 (Fastest) on Kaggle:
```python
# Copy v09_simple_blend_3models.py code into notebook
# Executes simple 3-model blend in ~4 minutes
# Expected: 0.92290 public, 0.92234 private score
```

---

## 💡 Recommendations

### For Competition Leaderboard:
1. **First Choice:** Use V20 (AutoGluon) - Best score
2. **Fallback:** Use V10 (5-Fold Ensemble) - Good score, controllable
3. **Quick Test:** Use V9 (Simple Blend) - Fast feedback

### For Learning:
- **AutoML Concepts:** Study V20
- **Ensemble Methods:** Study V10
- **Quick Prototyping:** Study V9
- **Single Model:** Study V7

### Performance vs Speed Tradeoff:
- V20: Best score but 4+ hours
- V10: Great score, 1.3 hours, educational
- V9: Good score, <4 minutes, simple code

---

## 🔧 All Version Files Included

```
/notebooks/
  ├── v20_autogluon_winner.py           # BEST (0.92380)
  ├── v18_autogluon_timelimit.py        # Fast AutoGluon (0.92223)
  ├── v10_ensemble_5fold_cv.py          # Learning model (0.92357)
  └── v09_simple_blend_3models.py       # Quick start (0.92290)

/description/
  ├── README.md                          # Full project guide
  ├── VERSIONS.md                        # Detailed version analysis
  └── VERSION_CODES.md                   # This file
```

---

## 📈 Key Metrics

- **Best Score:** 0.92380 (V20)
- **Best Manual Ensemble:** 0.92357 (V10)
- **Fastest Submission:** 3m 51s (V9)
- **Most Comprehensive:** 4h 9m (V20)
- **Total Versions:** 20
- **Successful:** 16
- **Failed:** 4

---

## 🎓 Learning Path

### Beginner:
1. Start with V9 (simple blend) - understand basics
2. Review v09_simple_blend_3models.py
3. Copy to Kaggle, run, verify score

### Intermediate:
1. Study V10 (5-fold ensemble)
2. Review v10_ensemble_5fold_cv.py
3. Understand OOF stacking concept
4. Implement with your own features

### Advanced:
1. Study V20 (AutoGluon optimization)
2. Review v20_autogluon_winner.py
3. Experiment with time_limit, presets
4. Build your own AutoGluon wrapper

---

**Repository:** [Predicting-Loan-Payback](https://github.com/mohan13krishna/Predicting-Loan-Payback)  
**Team:** Team Phoenix Algorithms  
**Competition:** Kaggle Playground Series S5E11  
**Final Rank:** 1255 / 3724 (Top 33.7%)
