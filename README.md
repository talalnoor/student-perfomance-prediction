
# Student Performance Prediction (Pass/Fail Classifier)

A machine learning pipeline that predicts whether a student will **pass or fail** based on demographic, social, and behavioral attributes — deliberately excluding prior period grades (G1, G2) to test real predictive power rather than grade leakage.

## Dataset

UCI Student Performance dataset — 649 students, 33 original attributes. Target (`pass_fail`) is derived from the final grade: **Pass = G3 ≥ 10**.

## Pipeline

1. **EDA** — class balance check, correlation analysis, distribution plots by class
2. **Feature Engineering**
   - LabelEncoder for binary categorical columns
   - One-hot encoding (`pd.get_dummies`) for multi-category columns
   - 2 engineered features: `parent_edu_avg`, `social_score`
   - StandardScaler for numeric features
3. **Train/Test Split** — 80/20, stratified, `random_state=42`
4. **Models trained & compared** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, feature importances, ROC curves
6. **Hyperparameter tuning** — GridSearchCV on Random Forest (`n_estimators`, `max_depth`, `min_samples_split`)

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.792 | 0.867 | 0.891 | 0.879 | 0.620 |
| Decision Tree | 0.746 | 0.853 | 0.845 | 0.849 | 0.523 |
| **Random Forest** | **0.808** | 0.851 | **0.936** | **0.892** | 0.643 |
| Gradient Boosting | 0.792 | 0.867 | 0.891 | 0.879 | 0.647 |

**Best model:** Random Forest. After GridSearchCV tuning (`max_depth=10, min_samples_split=10, n_estimators=200`), test ROC-AUC improved from 0.643 → 0.665.

**Most important feature across all models:** `failures` (number of past class failures).

Full written analysis: see [`report.md`](./report.md).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook project.ipynb
```

## Repo Structure

```
├── README.md              # this file
├── report.md               # 400-word Model Evaluation Report
├── project.ipynb           # full analysis notebook
├── student_dataset.csv     # dataset
└── requirements.txt        # dependencies
```

## Key Limitation

The dataset is imbalanced (84.6% pass / 15.4% fail). The model correctly identified only 6 of 20 actual failing students in the test set — a direct consequence of class imbalance. Future improvements: class-weighting, SMOTE oversampling, or richer behavioral features.
