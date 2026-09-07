# TEJAS Machine Learning Pipeline: Final Model Evaluation Report

**Dataset**: `data/tejas_pilot_master_dataset.csv` (10,000 records × 37 columns)  
**Partitioning**: GroupKFold by `asset_id` (Train: 7,000 | Val: 1,500 | Test: 1,500) — Zero Asset Leakage  
**Predictor Feature Dimensions**: 165 transformed features (Strict zero target leakage verified)  
**Evaluation Set**: 1,500 Untouched Test Samples across 484 unique physical assets  

---

## 1. Executive Summary: Champion Models Across All 4 Targets

| Target | Champion Architecture | Selection Criteria | Val Performance | Test Performance | Calibrated Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`failure_within_30d_target`** | Logistic Regression (Balanced, C=0.1) | F2 / Safety Recall $\ge$ 85% | Recall: **91.64%**, AUC: **0.8499** | Recall: **88.70%**, Prec: **37.87%**, F1: **0.5308**, AUC: **0.8450** | **0.260** (vs 0.50 default) |
| **`block_required_target`** | Logistic Regression (Balanced, C=0.1) | F2 / Safety Recall $\ge$ 85% | Recall: **92.20%**, AUC: **0.8237** | Recall: **91.04%**, Prec: **34.21%**, F1: **0.4973**, AUC: **0.8092** | **0.270** (vs 0.50 default) |
| **`priority_score_target`** | HistGradientBoosting Regressor | $R^2$ / MAE / Overfit Gap | $R^2$: **0.7493**, MAE: **5.25** | $R^2$: **0.7339**, MAE: **5.50**, Pearson $r$: **0.8568** | N/A (Continuous 0–100) |
| **`maintenance_duration_hours_target`** | Ridge Regression ($\alpha=10.0$) | Regularized $R^2$ / MAE | $R^2$: **0.3589**, MAE: **1.12h** | $R^2$: **0.2994**, MAE: **1.14h**, Pearson $r$: **0.5495** | N/A (Continuous Hours) |

---

## 2. Safety-Critical Classification Performance & Threshold Calibration

### Target 1: `failure_within_30d_target` (Risk of In-Service Catastrophic Failure)
- **Problem with Default Threshold (0.50)**: Missed 73 true failures on the test set (Recall = 78.84%).
- **Calibrated Safety Threshold (0.260)**: Reduced missed failures from 73 down to **39**, boosting Recall to **88.70%** (and **91.64%** on validation).
- **Confusion Matrix on Test Set (1,500 samples)**:
  - True Negatives (TN): **653**
  - False Positives (FP): **502** (flagged for inspection)
  - False Negatives (FN): **39** (missed failures)
  - True Positives (TP): **306** (prevented failures)
- **ROC-AUC**: **0.8450** | **PR-AUC**: **0.6505**

### Target 2: `block_required_target` (Traffic Block Window Requirement)
- **Problem with Default Threshold (0.50)**: Missed 93 track blocks (Recall = 73.95%).
- **Calibrated Safety Threshold (0.270)**: Reduced missed blocks from 93 down to **32**, achieving **91.04% Recall**.
- **Confusion Matrix on Test Set (1,500 samples)**:
  - True Negatives (TN): **518**
  - False Positives (FP): **625**
  - False Negatives (FN): **32**
  - True Positives (TP): **325**
- **ROC-AUC**: **0.8092** | **PR-AUC**: **0.6043**

---

## 3. Regression Targets Performance

### Target 3: `priority_score_target` (0–100 Granular Urgency Index)
- **Tuned Champion**: HistGradientBoosting Regressor (`learning_rate=0.05, max_depth=6, min_samples_leaf=20, l2_regularization=1.0`)
- **Test $R^2$**: **0.7339** (Val $R^2$: 0.7493) — Minimal generalization gap ($< 0.015$).
- **Test MAE**: **5.50 points** on a 0–100 scale.
- **Pearson Correlation**: **0.8568** ($p < 10^{-100}$).

### Target 4: `maintenance_duration_hours_target` (Physical Repair Hours)
- **Tuned Champion**: Ridge Regression ($\alpha=10.0$)
- **Test $R^2$**: **0.2994** (Val $R^2$: 0.3589).
- **Test MAE**: **1.14 hours** (RMSE: 1.46h).
- **Assessment**: Physical maintenance duration has high intrinsic variance; regularized Ridge prevents the heavy overfitting seen in deep tree models while providing solid baseline estimates.

---

## 4. Verification & Counterfactual Sanity Checks
- **Target Leakage**: Clean. 0 target or target-derived columns in predictor matrix.
- **Severity Monotonicity**: Verified. Severity LOW $\to$ CRITICAL monotonically elevates failure probability ($0.22 \to 0.88$) and priority score ($36.9 \to 61.3$).
- **Overdue Monotonicity**: Verified. Overdue days monotonically increases failure risk.
- **Operational Impact Monotonicity**: Verified. Increasing affected services increases block probability ($0.26 \to 1.00$).
- **Recurrence Monotonicity**: Verified. Prior recurrences monotonically scale failure risk ($0.28 \to 0.96$).
