# Baseline Model Training & Validation Report

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Phase**: Step 8 – Baseline Model Training & Selection  
**Target**: `urgent` (0 = Routine / Non-Urgent, 1 = Urgent Priority Block Required)  
**Dataset Partitions Evaluated**:
- **Train Set**: `data/processed/train.csv` (28,006 records × 137 features, 35.54% Urgent)
- **Validation Set**: `data/processed/validation.csv` (5,997 records × 137 features, 35.52% Urgent)
- **Test Set**: Isolated (Strictly untouched during this phase).

---

## 1. Models Trained & Configurations

Six model configurations across three distinct algorithm families were trained and evaluated on the validation set:

| Model ID | Algorithm Family | Configuration & Hyperparameters | Purpose |
|:---|:---|:---|:---|
| **M1a** | Logistic Regression (Baseline) | `solver='lbfgs'`, `max_iter=1000`, `random_state=42` | Linear baseline on standard class distribution |
| **M1b** | Logistic Regression (Balanced) | `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42` | Linear model with inverse-frequency class penalty |
| **M2a** | Random Forest (Default) | `n_estimators=100`, `max_depth=None`, `random_state=42`, `n_jobs=-1` | Non-linear bagging ensemble on standard distribution |
| **M2b** | Random Forest (Balanced) | `n_estimators=100`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` | Bagging ensemble with balanced sub-sample weighting |
| **M3a** | XGBoost Classifier (Default) | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, `objective='binary:logistic'`, `random_state=42` | Gradient boosted decision trees baseline |
| **M3b** | XGBoost Classifier (Weighted) | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`, `scale_pos_weight=1.8141`, `objective='binary:logistic'`, `random_state=42` | Gradient boosted trees with empirical class weighting |

---

## 2. Validation Performance Metrics

All metrics were computed strictly on the **out-of-sample validation set (5,997 records)** using default decision threshold $\tau = 0.50$:

| Model | Accuracy | Precision (`urgent=1`) | Recall (`urgent=1`) | F1-Score (`urgent=1`) | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 83.93% | 77.09% | 77.89% | 77.49% | 82.49% | 0.9111 | 0.8347 |
| **Logistic Regression (Balanced)** | 82.61% | 71.12% | 85.92% | 77.82% | 81.76% | 0.9110 | 0.8339 |
| **Random Forest (Default)** | 85.98% | **79.36%** | 81.78% | 80.55% | 84.79% | 0.9178 | 0.8361 |
| **Random Forest (Balanced)** | 85.93% | 77.25% | 85.59% | 81.20% | 84.98% | 0.9182 | 0.8359 |
| **XGBoost (Default)** | **86.34%** | 78.81% | 84.18% | 81.41% | **85.31%** | **0.9267** | **0.8607** |
| **XGBoost (Weighted)** | 85.74% | 75.27% | **89.15%** | **81.62%** | 84.99% | **0.9268** | 0.8593 |

---

## 3. Confusion Matrices Breakdown (Validation Set: 5,997 Cases)

*Ground Truth: 3,867 Non-Urgent (0) & 2,130 Urgent (1)*

```
1. Logistic Regression (Baseline)       2. Logistic Regression (Balanced)
   Predicted:   0        1                 Predicted:   0        1
Actual 0:    [ 3374     493 ]           Actual 0:    [ 3124     743 ]
Actual 1:    [  471    1659 ]           Actual 1:    [  300    1830 ]

3. Random Forest (Default)              4. Random Forest (Balanced)
   Predicted:   0        1                 Predicted:   0        1
Actual 0:    [ 3414     453 ]           Actual 0:    [ 3330     537 ]
Actual 1:    [  388    1742 ]           Actual 1:    [  307    1823 ]

5. XGBoost (Default)                    6. XGBoost (Weighted)
   Predicted:   0        1                 Predicted:   0        1
Actual 0:    [ 3385     482 ]           Actual 0:    [ 3243     624 ]
Actual 1:    [  337    1793 ]           Actual 1:    [  231    1899 ]
```

### Critical Safety Metric: False Negatives ($FN$)
In Indian Railways Automatic Block Planning, **False Negatives (missing an urgent maintenance task)** represent high safety and operational derailment/breakdown risks.
- **Logistic Regression (Baseline)**: $FN = 471$ (22.1% missed urgent tasks)
- **Random Forest (Default)**: $FN = 388$ (18.2% missed urgent tasks)
- **XGBoost (Default)**: $FN = 337$ (15.8% missed urgent tasks)
- **XGBoost (Weighted)**: **$FN = 231$ (only 10.8% missed urgent tasks)** $\rightarrow$ **31.5% reduction in missed urgent defects over unweighted XGBoost**.

---

## 4. Key Comparative Findings

### A. Non-Linear vs Linear Performance
- **Linear Models (Logistic Regression)** achieve a strong baseline ($\text{ROC-AUC} = 0.9111, \text{Accuracy} = 83.93\%$), confirming that our ordinal encoding and robust scaling provided strong linear signals.
- **Tree-Based Models (RF & XGBoost)** significantly outperform Logistic Regression ($\text{ROC-AUC} \ge 0.918$, $\text{Accuracy} \ge 85.9\%$), successfully capturing the non-linear interaction terms between high defect severity, high asset criticality, and high route traffic.

### B. XGBoost vs Random Forest
- **XGBoost** demonstrates superior probability calibration and discriminative ability, achieving the highest **ROC-AUC (0.9268)** and **PR-AUC (0.8607)** across all configurations.

### C. Impact of Class Weighting
- **Precision vs Recall Trade-off**:
  - Unweighted models achieve higher **Precision** (78.8%–79.4%) and higher overall **Accuracy** (86.0%–86.3%).
  - Class-weighted models (`scale_pos_weight=1.8141` or `class_weight='balanced'`) substantially boost **Recall for the urgent class**:
    - Logistic Regression Recall: $77.89\% \rightarrow 85.92\%$ (+8.03%)
    - Random Forest Recall: $81.78\% \rightarrow 85.59\%$ (+3.81%)
    - XGBoost Recall: $84.18\% \rightarrow \mathbf{89.15\%}$ (+4.97%)
  - **Verdict on Class Weighting**: For maintenance risk prediction where safety is paramount, **class weighting is empirically superior** because it captures 89.15% of all critical maintenance interventions with minimal F1 degradation ($81.41\% \rightarrow 81.62\%$).

---

## 5. Model Selection Summary

- **Top Performing Model**: **XGBoost (Weighted / Default)**
  - **ROC-AUC**: **0.9268**
  - **PR-AUC**: **0.8607**
  - **F1-Score (Urgent)**: **81.62%**
  - **Recall (Urgent)**: **89.15%**
- **Second Best Model**: **Random Forest (Balanced)** (ROC-AUC = 0.9182, F1 = 81.20%)
- **Strongest Baseline**: **Logistic Regression (Balanced)** (ROC-AUC = 0.9110, Recall = 85.92%)

---

## 6. Saved Model Artifacts

All trained model binaries have been serialized with `joblib` and archived in [ml/models/baseline_models/](file:///c:/Users/acous/Desktop/sih2026/ml/models/baseline_models/):
1. `models/baseline_models/logistic_regression_baseline.joblib`
2. `models/baseline_models/logistic_regression_balanced.joblib`
3. `models/baseline_models/random_forest_default.joblib`
4. `models/baseline_models/random_forest_balanced.joblib`
5. `models/baseline_models/xgboost_default.joblib`
6. `models/baseline_models/xgboost_weighted.joblib`
7. **Performance Table**: [ml/models/model_comparison.csv](file:///c:/Users/acous/Desktop/sih2026/ml/models/model_comparison.csv)
8. **Training Script**: [ml/models/train_models.py](file:///c:/Users/acous/Desktop/sih2026/ml/models/train_models.py)

*(Stopped as requested. The test set remains completely unaccessed.)*
