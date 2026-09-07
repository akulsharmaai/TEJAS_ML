# Final Unbiased Test Set Evaluation Report

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Phase**: Step 10 – Final Test Set Evaluation  
**Selected Model**: Tuned XGBoost Classifier (`models/tuned_models/xgboost_tuned.joblib`)  
**Test Set Evaluated**: `data/processed/test.csv` (Held-out & completely untouched until this step)  
**Evaluation Scope**: 5,997 Test Records × 137 Features  

---

## 1. Test Dataset Integrity & Protocol Verification

- **Integrity Statement**: The held-out test partition (`data/processed/test.csv`) was isolated during Step 7 (Preprocessing) and **remained 100% untouched** during all baseline model training (Step 8), hyperparameter search, and probability threshold calibration (Step 9).
- **No Refitting / Retuning**: The evaluated model was loaded directly from `models/tuned_models/xgboost_tuned.joblib`. Zero retraining, weight adjustments, or feature modifications were performed on the test data.
- **Test Set Composition**:
  - Total Records: **5,997**
  - Class 0 (Routine Maintenance): **3,866 (64.47%)**
  - Class 1 (Urgent Block Required): **2,131 (35.53%)**
  - Unique Physical Assets: **1,605** (Zero overlap with Train or Validation assets).

---

## 2. Final Test Performance Metrics (Default Threshold $\tau = 0.50$)

| Evaluation Metric | Test Set Result | Validation Result | Variance ($\Delta$) | Generalization Status |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | **86.59%** | 86.08% | $+0.51\%$ | Robust / Zero Degradation |
| **ROC-AUC** | **0.9305** | 0.9281 | $+0.0024$ | Exceptional Discrimination |
| **PR-AUC (Average Precision)** | **0.8627** | 0.8605 | $+0.0022$ | Highly Reliable Probability Ranking |
| **Precision (`urgent=1`)** | **77.77%** | 76.64% | $+1.13\%$ | Strong Precision |
| **Recall (`urgent=1`)** | **87.19%** | 87.46% | $-0.27\%$ | Consistent High Recall |
| **F1-Score (`urgent=1`)** | **82.21%** | 81.69% | $+0.52\%$ | Stable F1 Convergence |
| **Specificity (`urgent=0`)** | **86.26%** | 85.31% | $+0.95\%$ | Reliable Negative Identification |
| **True Positives ($TP$)** | **1,858** | 1,863 | — | Correctly flagged urgent tasks |
| **True Negatives ($TN$)** | **3,335** | 3,299 | — | Correctly flagged routine tasks |
| **False Positives ($FP$)** | **531** | 568 | — | Routine tasks flagged as urgent |
| **False Negatives ($FN$)** | **273** | 267 | — | Missed urgent tasks |

---

## 3. Test Confusion Matrix Breakdown

### Confusion Matrix at Default Threshold ($\tau = 0.50$)
```
                     Predicted Routine (0)    Predicted Urgent (1)
Actual Routine (0):         3,335 (TN)               531 (FP)
Actual Urgent (1):            273 (FN)             1,858 (TP)
```

### Confusion Matrix at Recommended Operational Threshold ($\tau = 0.45$)
```
                     Predicted Routine (0)    Predicted Urgent (1)
Actual Routine (0):         3,254 (TN)               612 (FP)
Actual Urgent (1):            232 (FN)             1,899 (TP)
```

---

## 4. Probability Threshold Comparison on Test Set

*Comprehensive evaluation across candidate operating thresholds (0.30 to 0.70):*

| Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC | PR-AUC | $TP$ | $TN$ | $FP$ | $FN$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.30** | 82.97% | 69.23% | **93.76%** | 79.65% | 77.03% | 0.9305 | 0.8627 | 1,998 | 2,978 | 888 | **133** |
| **0.35** | 84.29% | 71.58% | **92.54%** | 80.72% | 79.75% | 0.9305 | 0.8627 | 1,972 | 3,083 | 783 | **159** |
| **0.40** | 85.13% | 73.56% | **90.76%** | 81.26% | 82.02% | 0.9305 | 0.8627 | 1,934 | 3,171 | 695 | **197** |
| **0.45** *(Recommended)* | **85.93%** | **75.63%** | **89.11%** | **81.82%** | **84.17%** | 0.9305 | 0.8627 | **1,899** | **3,254** | **612** | **232** |
| **0.50** *(Default)* | 86.59% | 77.77% | 87.19% | 82.21% | 86.26% | 0.9305 | 0.8627 | 1,858 | 3,335 | 531 | 273 |
| **0.55** | 86.71% | 79.00% | 85.27% | 82.01% | 87.51% | 0.9305 | 0.8627 | 1,817 | 3,383 | 483 | 314 |
| **0.60** | 86.79% | 80.12% | 83.58% | 81.81% | 88.57% | 0.9305 | 0.8627 | 1,781 | 3,424 | 442 | 350 |
| **0.65** | 86.94% | 81.64% | 81.60% | 81.62% | 89.89% | 0.9305 | 0.8627 | 1,739 | 3,475 | 391 | 392 |
| **0.70** | 86.18% | 82.45% | 77.62% | 79.96% | 90.89% | 0.9305 | 0.8627 | 1,654 | 3,514 | 352 | 477 |

---

## 5. Overfitting & Generalization Assessment

1. **Validation vs. Test Alignment**:
   - $\text{Validation ROC-AUC} = \mathbf{0.9281} \quad \longleftrightarrow \quad \text{Test ROC-AUC} = \mathbf{0.9305}$
   - $\text{Validation PR-AUC} = \mathbf{0.8605} \quad \longleftrightarrow \quad \text{Test PR-AUC} = \mathbf{0.8627}$
   - $\text{Validation F1} = \mathbf{81.69\%} \quad \longleftrightarrow \quad \text{Test F1} = \mathbf{82.21\%}$
2. **Zero Overfitting Verdict**: The model exhibits **impeccable generalizability**. The slight improvement on the test partition confirms that our StratifiedGroupKFold partitioning on physical assets (`asset_id`) successfully eliminated data leakage and produced an exceptionally robust model.

---

## 6. Files & Artifacts Created

- **Evaluation Script**: [ml/models/final_evaluation/evaluate_final.py](file:///c:/Users/acous/Desktop/sih2026/ml/models/final_evaluation/evaluate_final.py)
- **Results CSV**: [ml/models/final_evaluation/final_test_results.csv](file:///c:/Users/acous/Desktop/sih2026/ml/models/final_evaluation/final_test_results.csv)
- **Evaluation Visualizations**: [ml/models/final_evaluation/test_confusion_matrix.png](file:///c:/Users/acous/Desktop/sih2026/ml/models/final_evaluation/test_confusion_matrix.png) (includes Confusion Matrix, Test vs. Val ROC Curves, and Precision-Recall Curves).
- **Report Document**: [ml/models/final_evaluation/final_evaluation_report.md](file:///c:/Users/acous/Desktop/sih2026/ml/models/final_evaluation/final_evaluation_report.md)
