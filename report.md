# Model Evaluation Report: Predicting Student Pass/Fail Outcomes

This project builds a classification model to predict whether a student will pass or fail, using the UCI Student Performance dataset (649 students, 33 attributes). The target was derived from the final grade (G3 ≥ 10 = Pass), and — deliberately — the prior period grades G1 and G2 were excluded from the feature set to avoid trivially leaking the answer and to test whether behavioral and demographic attributes alone can predict outcomes.

Exploratory analysis revealed a moderately imbalanced target: 84.6% of students passed versus 15.4% who failed. Correlation analysis showed no strong linear predictors; `failures` (past class failures) was the strongest at -0.38, followed by weaker signals from `studytime`, parental education, and alcohol consumption. This foreshadowed the modeling results: no single feature dominates, and prediction relies on combining many weak signals.

After encoding (LabelEncoder for binary categoricals, one-hot encoding for multi-category variables to avoid implying false ordinal relationships), engineering two new features (`parent_edu_avg`, `social_score`), and scaling with StandardScaler, four models were trained on an 80/20 stratified split: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

Random Forest performed best overall (Accuracy 0.808, Precision 0.851, Recall 0.936, F1 0.892, ROC-AUC 0.643), followed closely by Gradient Boosting (ROC-AUC 0.647). Decision Tree performed worst (ROC-AUC 0.523, near-random), consistent with a single tree's tendency to overfit on limited, imbalanced data. Across all models, `failures` was consistently the most important feature, reinforcing the correlation findings from EDA.

GridSearchCV tuning of Random Forest (27 combinations, 5-fold CV, optimizing F1) selected `max_depth=10, min_samples_split=10, n_estimators=200`. On the test set, tuning improved ROC-AUC from 0.643 to 0.665, though accuracy/precision/recall/F1 remained unchanged — the small test set (130 students) and inherently weak feature signal limited further gains despite a higher cross-validation F1 (0.929) during tuning itself.

**Key limitation:** the confusion matrix showed the tuned model correctly identified only 6 of 20 actual failing students, reflecting the cost of class imbalance — the model is biased toward predicting the majority "Pass" class. Future work could address this with class-weighting, SMOTE oversampling, or collecting richer behavioral features, since demographic/lifestyle attributes alone provide only moderate predictive power for this outcome.

