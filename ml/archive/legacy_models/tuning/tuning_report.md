# TEJAS Machine Learning Pipeline: Group-Aware Hyperparameter Tuning & Threshold Calibration Report

**Dataset**: `data/tejas_pilot_master_dataset.csv` (10,000 records × 37 columns)  
**Tuning Method**: 4-Fold GroupKFold Cross-Validation on the 7,000-record training split grouped by `asset_id` (2,250 unique assets)  
**Objective**: Optimize for safety-critical Recall ($\ge 85\%$) and $F_2$-score while preventing physical asset leakage.  

---

## 1. Multi-Target Tuning Summary

| Target | Model Evaluated | CV Metric | Validation Metric | Test Metric | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`failure_within_30d_target`** | **Logistic Regression (Balanced, C=0.1)** | $F_2$: 0.7096, AUC: 0.8577 | ROC-AUC: **0.8499**, Recall@Cal: **91.64%** | Recall@Cal: **88.70%**, AUC: **0.8450** | **CHAMPION** |
| `failure_within_30d_target` | Random Forest (Balanced, max_depth=16) | $F_2$: 0.6919, AUC: 0.8597 | ROC-AUC: 0.8441 | N/A | Baseline |
| `failure_within_30d_target` | HistGradientBoosting (Balanced) | $F_2$: 0.6991, AUC: 0.8607 | ROC-AUC: 0.8423 | N/A | Baseline |
| **`block_required_target`** | **Logistic Regression (Balanced, C=0.1)** | $F_2$: 0.6872, AUC: 0.8245 | ROC-AUC: **0.8237**, Recall@Cal: **92.20%** | Recall@Cal: **91.04%**, AUC: **0.8092** | **CHAMPION** |
| `block_required_target` | Random Forest (Balanced, max_depth=16) | $F_2$: 0.6651, AUC: 0.8120 | ROC-AUC: 0.8105 | N/A | Baseline |
| `block_required_target` | HistGradientBoosting (Balanced) | $F_2$: 0.6710, AUC: 0.8155 | ROC-AUC: 0.8140 | N/A | Baseline |
| **`priority_score_target`** | **HistGradientBoosting Regressor** | CV $R^2$: 0.7410, MAE: 5.30 | Val $R^2$: **0.7493**, MAE: **5.25** | Test $R^2$: **0.7339**, MAE: **5.50** | **CHAMPION** |
| `priority_score_target` | Ridge Regression ($\alpha=10.0$) | CV $R^2$: 0.7360, MAE: 5.41 | Val $R^2$: 0.7392, MAE: 5.36 | Test $R^2$: 0.7268 | Baseline |
| `priority_score_target` | Random Forest Regressor | CV $R^2$: 0.7380, MAE: 5.35 | Val $R^2$: 0.7442, MAE: 5.34 | Test $R^2$: 0.7302 | Baseline |
| **`maintenance_duration_hours_target`** | **Ridge Regression ($\alpha=10.0$)** | CV $R^2$: 0.3310, MAE: 1.13 | Val $R^2$: **0.3589**, MAE: **1.12** | Test $R^2$: **0.2994**, MAE: **1.14** | **CHAMPION** |
| `maintenance_duration_hours_target` | HistGradientBoosting Regressor | CV $R^2$: 0.3120, MAE: 1.15 | Val $R^2$: 0.3280, MAE: 1.14 | Test $R^2$: 0.2655 | Baseline |

---

## 2. Threshold Calibration Table

Thresholds were systematically swept from $0.05$ to $0.95$ on the 1,500 validation samples.

### Target 1: `failure_within_30d_target`
- **Default (0.50)**: Recall = **78.51%**, Precision = **46.96%**, False Negatives = **72**
- **Calibrated Safety Threshold (0.260)**: Recall = **91.64%**, Precision = **38.76%**, False Negatives = **28** (61% reduction in missed failures)

### Target 2: `block_required_target`
- **Default (0.50)**: Recall = **73.54%**, Precision = **51.36%**, False Negatives = **95**
- **Calibrated Safety Threshold (0.270)**: Recall = **92.20%**, Precision = **35.33%**, False Negatives = **28** (71% reduction in missed blocks)
