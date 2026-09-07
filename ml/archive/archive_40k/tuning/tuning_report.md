# Hyperparameter Tuning & Threshold Calibration Report

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Phase**: Step 9 – Hyperparameter Tuning & Probability Threshold Calibration  
**Primary Candidate**: XGBoost Classifier (Weighted)  
**Evaluation Scope**: Strict Train (28,006 records) + Validation Set (5,997 records).  
**Test Set Status**: **Completely untouched** (0 evaluation / 0 leakage).  

---

## 1. Cross-Validation & Search Methodology

To uphold the core principle of **zero asset-level data leakage**, hyperparameter optimization was performed on the training set using:
- **Search Strategy**: `RandomizedSearchCV` with fixed random seed (`RANDOM_SEED = 42`).
- **Inner Cross-Validation**: 4-Fold `StratifiedGroupKFold` grouped strictly on `asset_id` and stratified by `urgent`.
- **Optimization Objective**: F1-Score on the urgent class (`scoring='f1'`), preventing bias towards majority class accuracy.

---

## 2. Hyperparameter Search Spaces & Optimal Configurations

### A. XGBoost Classifier (Weighted) — 30 Iterations (120 Fits)
- **Search Range**:
  - `n_estimators`: `[100, 150, 200, 250]`
  - `max_depth`: `[4, 5, 6, 7, 8]`
  - `learning_rate`: `[0.03, 0.05, 0.08, 0.1, 0.15]`
  - `min_child_weight`: `[1, 3, 5, 7]`
  - `subsample`: `[0.7, 0.8, 0.9, 1.0]`
  - `colsample_bytree`: `[0.6, 0.7, 0.8, 0.9, 1.0]`
  - `gamma`: `[0, 0.1, 0.2, 0.5, 1.0]`
  - `reg_alpha`: `[0, 0.01, 0.1, 1.0]`
  - `reg_lambda`: `[0.5, 1.0, 2.0, 5.0]`
  - `scale_pos_weight`: `[1.5, 1.7, 1.814, 2.0, 2.2]`
- **Best Cross-Validation F1-Score**: **81.92%**
- **Optimal Hyperparameters**:
  ```python
  {
      'n_estimators': 150,
      'max_depth': 4,
      'learning_rate': 0.05,
      'min_child_weight': 5,
      'subsample': 1.0,
      'colsample_bytree': 0.8,
      'gamma': 0.5,
      'reg_alpha': 1.0,
      'reg_lambda': 1.0,
      'scale_pos_weight': 1.5
  }
  ```
- **Engineering Rationale**: A shallower tree depth (`max_depth=4` vs default 6) combined with a lower learning rate (`0.05` vs 0.1) and L1 regularization (`reg_alpha=1.0`) prevents overfitting on sparse one-hot defect categories, increasing generalization.

---

### B. Random Forest Classifier (Balanced) — 15 Iterations (60 Fits)
- **Search Range**:
  - `n_estimators`: `[100, 150, 200]`
  - `max_depth`: `[12, 16, 20, 25, None]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `max_features`: `['sqrt', 'log2', 0.3, 0.5]`
- **Best Cross-Validation F1-Score**: **81.64%**
- **Optimal Hyperparameters**:
  ```python
  {
      'n_estimators': 200,
      'max_depth': 20,
      'max_features': 0.3,
      'min_samples_leaf': 1,
      'min_samples_split': 2,
      'class_weight': 'balanced'
  }
  ```

---

### C. Logistic Regression (Balanced) — 9 Iterations (36 Fits)
- **Search Range**: `C`: `[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`
- **Best Cross-Validation F1-Score**: **78.40%**
- **Optimal Hyperparameters**:
  ```python
  {
      'C': 1.0,
      'penalty': 'l2',
      'solver': 'lbfgs',
      'class_weight': 'balanced'
  }
  ```

---

## 3. Baseline vs. Tuned Model Comparison (Validation Set: 5,997 Records)

| Model Name & Variant | Accuracy | Precision (`urgent=1`) | Recall (`urgent=1`) | F1-Score (`urgent=1`) | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 83.93% | 77.09% | 77.89% | 77.49% | 0.9111 | 0.8347 |
| **Logistic Regression (Tuned Balanced)**| 82.61% | 71.12% | 85.92% | 77.82% | 0.9110 | 0.8339 |
| **Random Forest (Default Baseline)** | 85.98% | **79.36%** | 81.78% | 80.55% | 0.9178 | 0.8361 |
| **Random Forest (Tuned Balanced)** | 86.01% | 76.75% | 86.95% | 81.53% | 0.9195 | 0.8409 |
| **XGBoost (Default Baseline)** | **86.34%** | 78.81% | 84.18% | 81.41% | 0.9267 | **0.8607** |
| **XGBoost (Tuned Weighted)** | 86.08% | 76.64% | **87.46%** | **81.69%** | **0.9281** | 0.8605 |

### Key Tuning Gains:
1. **XGBoost (Tuned Weighted)** reached the overall highest **ROC-AUC (0.9281)** and highest **F1-Score (81.69%)** at threshold 0.50.
2. **Random Forest (Tuned Balanced)** improved ROC-AUC from $0.9178 \rightarrow 0.9195$ and Recall from $81.78\% \rightarrow 86.95\%$.
3. **Generalization Stability**: Tuned XGBoost demonstrated virtually identical performance across cross-validation ($81.92\%$) and out-of-sample validation ($81.69\%$), confirming zero overfitting.

---

## 4. Systematic Probability Threshold Calibration (Tuned XGBoost)

*Validation Set: 5,997 records (3,867 Routine vs. 2,130 Urgent)*

| Decision Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | Specificity | False Negatives ($FN$) | False Positives ($FP$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.30** | 82.69% | 68.81% | **93.76%** | 79.37% | 76.60% | **133** | 905 |
| **0.35** | 83.86% | 70.85% | **92.68%** | 80.31% | 79.00% | **156** | 812 |
| **0.40** | 84.61% | 72.70% | **90.75%** | 80.73% | 81.23% | **197** | 726 |
| **0.45** *(Recommended)* | **85.46%** | **74.78%** | **89.11%** | **81.32%** | **83.45%** | **232** | **640** |
| **0.50** *(Default)* | 86.08% | 76.64% | 87.46% | 81.69% | 85.31% | 267 | 568 |
| **0.55** *(Peak F1)* | **86.61%** | **78.32%** | **86.15%** | **82.05%** | **86.86%** | 295 | **508** |
| **0.60** | 86.58% | 79.43% | 83.94% | 81.63% | 88.03% | 342 | 463 |
| **0.65** | 86.53% | 80.40% | 82.07% | 81.23% | 88.98% | 382 | 426 |
| **0.70** | 86.03% | 81.67% | 78.22% | 79.90% | 90.33% | 464 | 374 |

---

## 5. Threshold Analysis for Railway Operations & Final Recommendation

### Trade-Off Dynamics in Railway Automatic Block Planning:
1. **Cost of False Negatives ($FN$)**: An undetected urgent defect risks in-service catastrophic failure, broken rail, OHE entanglement, or unpredicted train stalling, leading to network-wide cascading delays or safety incidents.
2. **Cost of False Positives ($FP$)**: Scheduling an unnecessary emergency block window creates temporary track capacity constraints and avoidable train timetable re-profiling.

### Operating Point Selection:
- **Peak F1 Threshold ($\tau = 0.55$)**: Achieves the mathematical optimum with $F1 = 82.05\%$, $86.61\%$ accuracy, and $78.32\%$ precision, but misses 295 urgent defects ($13.85\%$ miss rate).
- **Recommended Operational Threshold ($\tau = 0.45$)**:
  - **Recall**: **89.11%** (captures 1,898 out of 2,130 urgent work orders).
  - **Precision**: **74.78%** (nearly 3 out of every 4 flagged tasks are genuine urgent interventions).
  - **F1-Score**: **81.32%** (robust balance).
  - **False Negatives**: Reduced by **63 cases (21.4% safety hazard reduction)** compared to $\tau = 0.55$.
  - **False Positives**: Modest 640 false positives across nearly 6,000 tasks (16.5% of non-urgent tasks).

**Final Recommendation**: **Tuned XGBoost with decision threshold $\tau = 0.45$** is the optimal operating configuration for deployment in the SIH26027 Automatic Block Planning pipeline.

---

## 6. Files & Artifacts Created

1. **Tuning & Calibration Script**: [ml/models/tuning/tune_models.py](file:///c:/Users/acous/Desktop/sih2026/ml/models/tuning/tune_models.py)
2. **Tuning Results CSV**: [ml/models/tuning/tuning_results.csv](file:///c:/Users/acous/Desktop/sih2026/ml/models/tuning/tuning_results.csv)
3. **Threshold Calibration CSV**: [ml/models/tuning/threshold_calibration.csv](file:///c:/Users/acous/Desktop/sih2026/ml/models/tuning/threshold_calibration.csv)
4. **Tuned Model Binaries** (in [ml/models/tuned_models/](file:///c:/Users/acous/Desktop/sih2026/ml/models/tuned_models/)):
   - `models/tuned_models/xgboost_tuned.joblib` *(Top Recommended Model)*
   - `models/tuned_models/random_forest_tuned.joblib`
   - `models/tuned_models/logistic_regression_tuned.joblib`

*(Stopped as requested. The test set remains completely unaccessed.)*
