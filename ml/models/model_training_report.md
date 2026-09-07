# TEJAS Machine Learning Pipeline: Baseline Model Training Report

**Dataset**: `data/tejas_pilot_master_dataset.csv` (10,000 records × 37 columns)
**Splitting Strategy**: GroupKFold by `asset_id` (Train: 7,000 | Val: 1,500 | Test: 1,500) — Zero Asset Leakage
**Predictor Feature Dimensions**: 165 transformed features (Strict zero target leakage)

---

## 1. Executive Summary: Best Model Per Target

| Target | Best Architecture | Primary Metric | Val Performance | Test Performance |
| :--- | :--- | :--- | :--- | :--- |
| **failure_within_30d_target** (Classification) | Logistic Regression (Balanced) | F1 / Recall / ROC-AUC | F1: 0.5878, Recall: 0.7791, AUC: 0.8449 | F1: 0.5960, Recall: 0.7739, AUC: 0.8436 |
| **block_required_target** (Classification) | Logistic Regression (Balanced) | F1 / Recall / ROC-AUC | F1: 0.6002, Recall: 0.7298, AUC: 0.8221 | F1: 0.5749, Recall: 0.7255, AUC: 0.8068 |
| **priority_score_target** (Regression) | HistGradientBoosting Regressor | $R^2$ / MAE / RMSE | $R^2$: 0.7444, MAE: 5.32, RMSE: 6.78 | $R^2$: 0.7302, MAE: 5.52, RMSE: 7.00 |
| **maintenance_duration_hours_target** (Regression) | Ridge Regression | $R^2$ / MAE / RMSE | $R^2$: 0.3550, MAE: 1.12h, RMSE: 1.40h | $R^2$: 0.2904, MAE: 1.14h, RMSE: 1.47h |

---

## 2. Complete Model Comparison Table

| target | task_type | model | train_accuracy | val_accuracy | test_accuracy | val_precision | test_precision | val_recall | test_recall | val_f1 | test_f1 | val_roc_auc | test_roc_auc | val_pr_auc | test_pr_auc | val_confusion_matrix | test_confusion_matrix | overfit_gap_acc | train_r2 | val_r2 | test_r2 | val_mae | test_mae | val_rmse | test_rmse | overfit_gap_r2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| failure_within_30d_target | Classification | Logistic Regression (Balanced) | 0.7863 | 0.7560 | 0.7587 | 0.4720 | 0.4846 | 0.7791 | 0.7739 | 0.5878 | 0.5960 | 0.8449 | 0.8436 | 0.6562 | 0.6527 | [[873, 292], [74, 261]] | [[871, 284], [78, 267]] | 0.0303 | nan | nan | nan | nan | nan | nan | nan | nan |
| failure_within_30d_target | Classification | Random Forest (Balanced) | 0.8859 | 0.7793 | 0.7867 | 0.5043 | 0.5272 | 0.7045 | 0.7014 | 0.5878 | 0.6020 | 0.8382 | 0.8403 | 0.5962 | 0.6165 | [[933, 232], [99, 236]] | [[938, 217], [103, 242]] | 0.1065 | nan | nan | nan | nan | nan | nan | nan | nan |
| failure_within_30d_target | Classification | HistGradientBoosting (Balanced) | 0.8861 | 0.7740 | 0.7947 | 0.4957 | 0.5400 | 0.6896 | 0.7246 | 0.5768 | 0.6188 | 0.8275 | 0.8386 | 0.6167 | 0.6384 | [[930, 235], [104, 231]] | [[942, 213], [95, 250]] | 0.1121 | nan | nan | nan | nan | nan | nan | nan | nan |
| block_required_target | Classification | Logistic Regression (Balanced) | 0.7710 | 0.7673 | 0.7447 | 0.5097 | 0.4761 | 0.7298 | 0.7255 | 0.6002 | 0.5749 | 0.8221 | 0.8068 | 0.6065 | 0.5976 | [[889, 252], [97, 262]] | [[858, 285], [98, 259]] | 0.0037 | nan | nan | nan | nan | nan | nan | nan | nan |
| block_required_target | Classification | Random Forest (Balanced) | 0.8846 | 0.7793 | 0.7573 | 0.5306 | 0.4926 | 0.6769 | 0.6527 | 0.5949 | 0.5614 | 0.8028 | 0.7908 | 0.5579 | 0.5462 | [[926, 215], [116, 243]] | [[903, 240], [124, 233]] | 0.1052 | nan | nan | nan | nan | nan | nan | nan | nan |
| block_required_target | Classification | HistGradientBoosting (Balanced) | 0.8814 | 0.7713 | 0.7660 | 0.5181 | 0.5065 | 0.6379 | 0.6583 | 0.5718 | 0.5725 | 0.8099 | 0.8017 | 0.6022 | 0.5938 | [[928, 213], [130, 229]] | [[914, 229], [122, 235]] | 0.1101 | nan | nan | nan | nan | nan | nan | nan | nan |
| priority_score_target | Regression | Ridge Regression | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.7502 | 0.7392 | 0.7268 | 5.3630 | 5.5778 | 6.8498 | 7.0409 | 0.0109 |
| priority_score_target | Regression | Random Forest Regressor | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.8856 | 0.7442 | 0.7302 | 5.3379 | 5.5310 | 6.7848 | 6.9967 | 0.1415 |
| priority_score_target | Regression | HistGradientBoosting Regressor | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.8375 | 0.7444 | 0.7302 | 5.3162 | 5.5226 | 6.7817 | 6.9963 | 0.0931 |
| maintenance_duration_hours_target | Regression | Ridge Regression | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.3263 | 0.3550 | 0.2904 | 1.1215 | 1.1442 | 1.3995 | 1.4653 | -0.0286 |
| maintenance_duration_hours_target | Regression | Random Forest Regressor | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.5905 | 0.3240 | 0.2657 | 1.1402 | 1.1682 | 1.4327 | 1.4905 | 0.2665 |
| maintenance_duration_hours_target | Regression | HistGradientBoosting Regressor | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.5385 | 0.3264 | 0.2655 | 1.1381 | 1.1703 | 1.4301 | 1.4907 | 0.2120 |

---

## 3. Strongest Features by Target

### Top Features: `failure_within_30d_target_Logistic Regression (Balanced)`

| Feature | Importance / Weight |
| :--- | :--- |
| days_since_last_maintenance | 4.0021 |
| maintenance_overdue_days | 3.0949 |
| defect_severity_ord | 1.7089 |
| state_Delhi | 1.0786 |
| state_Goa | 0.8833 |
| state_Jammu and Kashmir | 0.8570 |
| state_Himachal Pradesh | 0.7551 |
| maintenance_interval_days | 0.6637 |
| state_Andhra Pradesh | 0.6242 |
| zone_SCR | 0.5883 |

### Top Features: `failure_within_30d_target_Random Forest (Balanced)`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 0.2102 |
| maintenance_overdue_days | 0.1137 |
| maintenance_overdue_ratio | 0.1019 |
| days_since_last_maintenance | 0.0865 |
| traffic_impact_ratio | 0.0505 |
| defect_duration_days | 0.0452 |
| asset_age_years | 0.0245 |
| failures_last_365d | 0.0241 |
| affected_services_if_blocked | 0.0236 |
| asset_criticality_score | 0.0228 |

### Top Features: `block_required_target_Logistic Regression (Balanced)`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 1.8050 |
| state_Himachal Pradesh | 1.0803 |
| state_Uttarakhand | 0.8603 |
| defect_type_fitting looseness | 0.7976 |
| defect_type_geometry deviation | 0.6596 |
| zone_ECoR | 0.6391 |
| state_Assam | 0.6018 |
| defect_type_boom locking fault | 0.5657 |
| defect_type_drainage blockage | 0.5385 |
| defect_type_conductor damage | 0.5195 |

### Top Features: `block_required_target_Random Forest (Balanced)`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 0.2174 |
| traffic_impact_ratio | 0.0651 |
| affected_services_if_blocked | 0.0627 |
| maintenance_overdue_days | 0.0543 |
| days_since_last_maintenance | 0.0542 |
| maintenance_overdue_ratio | 0.0510 |
| defect_duration_days | 0.0469 |
| operational_impact_score | 0.0429 |
| asset_criticality_score | 0.0429 |
| traffic_percentile | 0.0387 |

### Top Features: `priority_score_target_Ridge Regression`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 12.3248 |
| operational_impact_score | 4.5073 |
| state_Himachal Pradesh | 2.6229 |
| defect_type_connection looseness | 2.5478 |
| state_Puducherry | 2.5390 |
| state_Uttarakhand | 2.4812 |
| zone_? | 2.3304 |
| state_Kerala | 1.9947 |
| defect_type_joint heating | 1.8832 |
| state_Jammu and Kashmir | 1.8049 |

### Top Features: `priority_score_target_Random Forest Regressor`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 0.6546 |
| operational_impact_score | 0.1597 |
| maintenance_overdue_days | 0.0189 |
| maintenance_overdue_ratio | 0.0163 |
| asset_age_years | 0.0151 |
| days_since_last_maintenance | 0.0146 |
| defect_duration_days | 0.0134 |
| affected_services_if_blocked | 0.0116 |
| asset_criticality_score | 0.0116 |
| traffic_impact_ratio | 0.0112 |

### Top Features: `maintenance_duration_hours_target_Ridge Regression`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 0.9830 |
| state_Goa | 0.6893 |
| state_Uttarakhand | 0.6296 |
| state_Tripura | 0.5813 |
| state_Delhi | 0.4681 |
| asset_type_Traction Substation | 0.4249 |
| safety_function_traction power supply | 0.4249 |
| defect_type_DGA anomaly | 0.4069 |
| state_Puducherry | 0.4039 |
| zone_CR | 0.4023 |

### Top Features: `maintenance_duration_hours_target_Random Forest Regressor`

| Feature | Importance / Weight |
| :--- | :--- |
| defect_severity_ord | 0.3191 |
| department_S&T | 0.1064 |
| traffic_impact_ratio | 0.0516 |
| defect_duration_days | 0.0502 |
| asset_age_years | 0.0451 |
| asset_criticality_score | 0.0330 |
| days_since_last_maintenance | 0.0311 |
| maintenance_overdue_ratio | 0.0272 |
| operational_impact_score | 0.0271 |
| affected_services_if_blocked | 0.0247 |

