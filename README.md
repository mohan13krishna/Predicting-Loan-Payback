# 🏆 Predicting Loan Payback 
 
<div align="center">

![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S5E11-blue?logo=kaggle&style=flat-square)
![Status](https://img.shields.io/badge/Status-COMPLETED-brightgreen?style=flat-square)
![Best%20Score](https://img.shields.io/badge/Best%20Score-0.92380%20%2F%200.92304-gold?style=flat-square)
![Rank](https://img.shields.io/badge/Rank-1255%20%2F%203724-silver?style=flat-square)

**Team Phoenix Algorithms** | Ensemble ML Pipeline | Binary Classification

[🔗 Competition Link](https://www.kaggle.com/competitions/playground-series-s5e11) | [📊 Notebook](https://www.kaggle.com/code/mohan13krishna/predicting-loan-payback-version-20) | [💾 Repository](https://github.com/mohan13krishna/Predicting-Loan-Payback)

</div>

---

## 📋 Overview

**Predicting Loan Payback** is Kaggle's Playground Series Season 5, Episode 11 competition. The objective is to predict the probability that a borrower will pay back their loan.

**Our Approach:** Advanced AutoML ensemble using state-of-the-art gradient boosting frameworks combined with strategic hyperparameter tuning to achieve **top 34%** (rank 1255/3724) on the leaderboard.

### 🎯 Dataset
- **Training Samples:** 593,994 | **Test Samples:** 256,065
- **Features:** 12 numerical & categorical attributes
- **Target Variable:** Binary (Loan Paid Back: Yes/No)
- **Evaluation Metric:** ROC-AUC Score

---

## 👥 Team Members

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/mohan13krishna">
        <img src="https://avatars.githubusercontent.com/u/mohan13krishna?v=4" width="100px;" alt=""/><br/>
        <b>Mohan Krishna Thalla</b><br/>
        <i>Lead Developer</i><br/>
        <a href="https://github.com/mohan13krishna">@mohan13krishna</a>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/rakeshkolipaka">
        <img src="https://avatars.githubusercontent.com/u/rakeshkolipaka?v=4" width="100px;" alt=""/><br/>
        <b>Rakesh Kolipaka</b><br/>
        <i>ML Engineer</i><br/>
        <a href="https://github.com/rakeshkolipaka">@rakeshkolipaka</a>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ranjithkumardigutla">
        <img src="https://avatars.githubusercontent.com/u/ranjithkumardigutla?v=4" width="100px;" alt=""/><br/>
        <b>Ranjith Kumar Digutla</b><br/>
        <i>Data Scientist</i><br/>
        <a href="https://github.com/ranjithkumardigutla">@ranjithkumardigutla</a>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/neelamuday">
        <img src="https://avatars.githubusercontent.com/u/neelamuday?v=4" width="100px;" alt=""/><br/>
        <b>Neelam Uday Kiran</b><br/>
        <i>Analytics Lead</i><br/>
        <a href="https://github.com/neelamuday">@neelamuday</a>
      </a>
    </td>
  </tr>
</table>

---

## 🔬 Technical Architecture

### Ensemble Strategy
Our ensemble combines multiple state-of-the-art algorithms with weighted averaging:

| Model | Framework | Score | Weight |
|-------|-----------|-------|--------|
| **AutoGluon** (Best Quality) | AutoML | 0.9238 | 60% |
| **CatBoost Ensemble** | Gradient Boosting | 0.9204 | 20% |
| **LightGBM (5-fold CV)** | Gradient Boosting | 0.9224 | 15% |
| **XGBoost (5-fold CV)** | Gradient Boosting | 0.9212 | 5% |

### Key Features
✅ **5-Fold Stratified Cross-Validation** - Robust model validation  
✅ **Out-of-Fold (OOF) Predictions** - Meta-learning stacking  
✅ **Categorical Handling** - Native support via CatBoost & Label Encoding  
✅ **Hyperparameter Tuning** - Extensive grid/random search optimization  
✅ **Time-Limited AutoML** - 3600+ seconds training per model  
✅ **Ensemble Voting** - Weighted combination of base learners  

### Model Configurations

**CatBoost**
```python
iterations=2500, learning_rate=0.015, depth=8, l2_leaf_reg=5
loss_function="Logloss", eval_metric="AUC"
```

**LightGBM**
```python
learning_rate=0.03, num_leaves=48, feature_fraction=0.85
bagging_fraction=0.85, num_boost_round=1200
```

**XGBoost**
```python
n_estimators=900, learning_rate=0.03, max_depth=6
subsample=0.9, colsample_bytree=0.85
```

---

## 📊 Leaderboard Performance

### Submissions Ranking
| Version | Public | Private | Status |
|---------|--------|---------|--------|
| **v20** | **0.92380** | **0.92304** | ✅ **BEST** |
| v18 | 0.92223 | 0.92145 | AutoGluon |
| v15 | 0.92176 | 0.92108 | Ensemble |
| v10 | 0.92357 | 0.92284 | 5-Fold CV |
| v9 | 0.92290 | 0.92234 | Blend |
| v7 | 0.92189 | 0.92138 | Rakesh |
| v14 | 0.91455 | 0.91421 | Random Forest |

### Final Rank
- **Rank:** 1255 / 3724 Teams
- **Percentile:** Top 33.7%
- **Final Public Score:** 0.92380
- **Final Private Score:** 0.92304

---

## 📁 Project Structure

```
Predicting-Loan-Payback/
├── README.md                                          # Project documentation
├── predicting-loan-payback.ipynb                     # Main notebook (Version 20)
├── train.csv                                          # Training dataset (593K samples)
├── test.csv                                           # Test dataset (256K samples)
├── sample_submission.csv                              # Sample submission format
└── submission.csv                                     # Final predictions
```

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
lightgbm >= 3.3.0
xgboost >= 1.5.0
catboost >= 1.0.0
autogluon >= 0.5.0
```

### Installation & Execution

**1. Clone Repository**
```bash
git clone https://github.com/mohan13krishna/Predicting-Loan-Payback.git
cd Predicting-Loan-Payback
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run Notebook**
```bash
# Using Jupyter
jupyter notebook predicting-loan-payback.ipynb

# Or upload to Kaggle and run directly
```

**4. Generate Submission**
```python
# Predictions automatically saved to submission.csv
# Submit to Kaggle competition
```

---

## 🔧 Hyperparameter Tuning Journey

### Iterations & Optimization
- **Base Models (v1-v7):** Individual LightGBM, XGBoost, CatBoost models
- **Ensemble Attempts (v8-v15):** Simple averaging, weighted blending, stacking
- **AutoML Pivot (v16-v19):** Explored AutoGluon, best_quality preset
- **Final Winner (v20):** Optimized AutoGluon with ensemble fallback

### Key Insights
1. **AutoGluon Superiority** - AutoML with best_quality preset significantly outperformed manual ensembles
2. **Categorical Handling** - Native categorical support in CatBoost crucial for performance
3. **Time Investment** - 3600+ second training windows maximize model quality
4. **Cross-Validation** - 5-fold stratified CV prevents overfitting better than single train/test split

---

## 💡 Feature Engineering

### Numerical Features
- `annual_income` - Borrower's yearly income
- `debt_to_income_ratio` - Debt proportion (key signal)
- `credit_score` - Creditworthiness indicator
- `loan_amount` - Size of loan
- `interest_rate` - Loan interest percentage

### Categorical Features
- `gender` - Borrower gender
- `marital_status` - Marital status
- `education_level` - Education qualification
- `employment_status` - Employment type
- `loan_purpose` - Purpose of loan
- `grade_subgrade` - Loan grade classification

### Preprocessing
✅ Label encoding for tree-based models  
✅ Outlier clipping (IQR method)  
✅ Native categorical handling in CatBoost  
✅ Feature scaling where applicable  

---

## 📈 Competition Intel

| Aspect | Details |
|--------|---------|
| **Competition Type** | Tabular Prediction (Binary Classification) |
| **Duration** | 1 Month (Nov 1 - Dec 1, 2025) |
| **Data Type** | Synthetic (generated from Deep Learning model) |
| **Submission Format** | CSV (id, loan_paid_back probability) |
| **Evaluation Metric** | ROC-AUC Score |
| **Prize Pool** | 🏆 Top 3 get Kaggle Merchandise |

---

## 🎓 Learning Outcomes

1. **AutoML Frameworks** - Leveraging AutoGluon for competitive edge
2. **Ensemble Methods** - Combining diverse models for robustness
3. **Hyperparameter Optimization** - Systematic tuning strategies
4. **Cross-Validation Best Practices** - Preventing data leakage
5. **Binary Classification Pipeline** - End-to-end ML workflow
6. **Kaggle Competition Dynamics** - Public/Private scoring, late submissions

---

## 📝 Version History

| Version | Approach | Score | Runtime | Notes |
|---------|----------|-------|---------|-------|
| v20 | AutoGluon Optimized | **0.92380** | 4h 9m | ✅ FINAL |
| v19 | Ensemble Blend | 0.92223 | 49m | Failed |
| v18 | AutoGluon + LightGBM | 0.92223 | 1h 8m | ✅ Passed |
| v15 | CatBoost Heavy | 0.92176 | 19m | ✅ Passed |
| v14 | Random Forest | 0.91455 | 3m | ✅ Passed |
| v10 | 5-Fold CV Ensemble | 0.92357 | 1h 23m | ✅ Passed |
| v9 | Simple Blend | 0.92290 | 3m | ✅ Passed |
| v1-v8 | Individual Models | Varying | Variable | Exploration |

---

## 🤝 Contributing

Contributions from Team Phoenix members welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** changes with clear messages
4. **Push** to branch (`git push origin feature/improvement`)
5. **Submit** Pull Request

---

## 📚 References & Resources

- [Kaggle Competition Page](https://www.kaggle.com/competitions/playground-series-s5e11)
- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [CatBoost Guide](https://catboost.ai/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [XGBoost Tutorial](https://xgboost.readthedocs.io/)

---

## 📞 Contact & Support

- **Team Phoenix:** [GitHub Organization](https://github.com/mohan13krishna)
- **Lead Contact:** Mohan Krishna Thalla
- **Competition:** [Kaggle Playground S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

<div align="center">

**Made with ❤️ by Team Phoenix Algorithms**

*Building predictive models, one competition at a time.*

⭐ If you found this helpful, please star the repository!

</div>
