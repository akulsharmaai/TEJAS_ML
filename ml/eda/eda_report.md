# Exploratory Data Analysis (EDA) Report: SIH26027 Railway Maintenance Tasks

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Task**: Maintenance Task Urgency & Risk Scoring  
**Dataset**: `data/processed/railway_maintenance_tasks.csv`  
**Dataset Dimensions**: 40,000 Rows × 20 Columns  

---

## 1. Dataset Overview

### Schema & Data Types
The dataset contains **40,000 maintenance task records** across **20 attributes** (11 numerical and 9 categorical/identifier).

| Column Name | Data Type | Non-Null Count | Unique Values | Description |
|:---|:---|:---:|:---:|:---|
| `task_id` | Object (String) | 40,000 | 40,000 | Unique maintenance work-order ID |
| `asset_id` | Object (String) | 40,000 | 11,000 | Persistent physical asset tag |
| `department` | Object (String) | 40,000 | 3 | Engineering, S&T, Traction |
| `asset_type` | Object (String) | 40,000 | 23 | Specific railway asset category |
| `asset_age_years` | Float64 | 40,000 | 275 | Age in service (0.5 to 39.5 yrs) |
| `asset_criticality` | Object (String) | 40,000 | 4 | LOW, MEDIUM, HIGH, CRITICAL |
| `defect_type` | Object (String) | 40,000 | 40 | Specific observed defect / failure mode |
| `defect_severity` | Object (String) | 40,000 | 4 | LOW, MEDIUM, HIGH, CRITICAL |
| `num_open_defects` | Int64 | 40,000 | 6 | Concurrent open defects (1 to 6) |
| `days_since_defect` | Int64 | 40,000 | 85 | Days since defect logged (1 to 89) |
| `days_overdue` | Int64 | 40,000 | 66 | Days overdue beyond SLA (0 to 65) |
| `maintenance_frequency` | Object (String) | 40,000 | 6 | Mandated cycle (Weekly to Annual) |
| `days_since_last_maintenance` | Int64 | 40,000 | 433 | Days since last maintenance (2 to 437) |
| `previous_failures` | Int64 | 40,000 | 15 | Lifetime asset failures (0 to 14) |
| `failures_last_12_months` | Int64 | 40,000 | 7 | Recent failures in last year (0 to 6) |
| `trains_per_day` | Int64 | 40,000 | 218 | Daily total traffic (10 to 255) |
| `goods_trains_per_day` | Int64 | 40,000 | 117 | Daily freight train traffic (2 to 120) |
| `operational_impact` | Object (String) | 40,000 | 4 | LOW, MEDIUM, HIGH, CRITICAL |
| `maintenance_duration_hours` | Float64 | 40,000 | 567 | Required block window (0.50 to 6.98 hrs) |
| `urgent` | Int64 (Target) | 40,000 | 2 | Binary target (0 or 1) |

### Summary Statistics of Numerical Features

| Feature | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `asset_age_years` | 8.97 | 5.85 | 0.50 | 4.90 | 7.60 | 11.50 | 39.50 | +1.68 |
| `num_open_defects` | 2.29 | 1.36 | 1.00 | 1.00 | 2.00 | 3.00 | 6.00 | +0.76 |
| `days_since_defect` | 18.89 | 15.11 | 1.00 | 8.00 | 14.00 | 26.00 | 89.00 | +1.43 |
| `days_overdue` | 6.13 | 8.96 | 0.00 | 0.00 | 2.00 | 9.00 | 65.00 | +2.23 |
| `days_since_last_maintenance` | 96.71 | 96.84 | 2.00 | 32.00 | 53.00 | 107.00 | 437.00 | +1.58 |
| `previous_failures` | 3.13 | 2.19 | 0.00 | 2.00 | 3.00 | 4.00 | 14.00 | +0.94 |
| `failures_last_12_months` | 1.29 | 1.21 | 0.00 | 0.00 | 1.00 | 2.00 | 6.00 | +0.95 |
| `trains_per_day` | 100.35 | 54.43 | 10.00 | 45.00 | 107.00 | 144.00 | 255.00 | +0.13 |
| `goods_trains_per_day` | 43.71 | 29.08 | 2.00 | 12.00 | 46.00 | 66.00 | 120.00 | +0.22 |
| `maintenance_duration_hours` | 3.32 | 1.24 | 0.50 | 2.43 | 3.14 | 4.02 | 6.98 | +0.44 |

---

## 2. Data Quality Audit

| Quality Dimension | Test Applied | Result | Status |
|:---|:---|:---|:---:|
| **Missingness** | `df.isnull().sum()` across all 20 columns | Exactly `0` missing values across 800,000 cells | Passed |
| **Duplicates** | `df.duplicated()` | `0` duplicate task records | Passed |
| **Physical Constraints** | `goods_trains_per_day <= trains_per_day` | `0` violations (all freight <= total traffic) | Passed |
| **Range Checks** | Non-negative values for counts, age, durations | `0` negative values | Passed |
| **Asset Consistency** | Unique asset vs task IDs | 11,000 physical assets across 40,000 tasks (realistic re-inspection pool) | Passed |
| **Categorical Labels** | Whitespace, casing, and trailing characters | Consistent across all 9 categorical columns | Passed |
| **Outliers (IQR Method)**| 1.5 × IQR thresholding | Moderate positive tail outliers in overdue days (6.03%) and days since last maintenance (7.90%), consistent with genuine real-world delayed maintenance backlogs. | Expected / Valid |

---

## 3. Target Variable Analysis (`urgent`)

- **Class `0` (Routine / Schedulable)**: 25,787 (**64.47%**)
- **Class `1` (Urgent / Priority Block Required)**: 14,213 (**35.53%**)
- **Imbalance Evaluation**: The dataset exhibits a **healthy, moderate class balance (~1.81 : 1 ratio)**. It realistically represents railway operational priority backlogs where ~35% of detected maintenance tasks require prioritized traffic block scheduling. No extreme synthetic oversampling (e.g. SMOTE) is required during initial modeling.

---

## 4. Target vs. Feature Analysis

### A. Categorical Feature Impact on Urgency

| Feature | Category | Count | % Urgent (Class 1) | Observations |
|:---|:---|:---:|:---:|:---|
| **Defect Severity** | `CRITICAL` | 11,149 | **55.42%** | Critical defects are frequently urgent, but not 100% (allows for non-critical siding contexts). |
| | `HIGH` | 16,834 | **45.21%** | Substantial portion escalates to urgent based on traffic/overdue strain. |
| | `MEDIUM` | 8,948 | **4.02%** | Rarely urgent unless heavily compounded by overdue days or trunk traffic. |
| | `LOW` | 3,069 | **2.09%** | Almost exclusively routine. |
| **Asset Criticality** | `CRITICAL` | 10,679 | **56.94%** | Critical assets on trunk routes trigger urgent prioritization. |
| | `HIGH` | 17,568 | **44.56%** | High baseline urgency when coupled with severe defects. |
| | `MEDIUM` | 9,770 | **2.80%** | Handled via routine scheduled maintenance. |
| | `LOW` | 1,983 | **1.51%** | Negligible urgency. |
| **Operational Impact** | `CRITICAL` | 9,970 | **63.70%** | Strongest categorical indicator of urgency due to cascade delay risk. |
| | `HIGH` | 12,335 | **46.53%** | Moderate-to-high urgency. |
| | `MEDIUM` | 11,883 | **16.54%** | Intermediate buffer. |
| | `LOW` | 5,812 | **2.70%** | Low network disruption potential. |
| **Department** | `Traction` | 11,083 | **42.27%** | Higher average urgency due to high OHE failure criticality. |
| | `S&T` | 12,890 | **39.98%** | High urgency due to point machine & interlocking safety dependencies. |
| | `Engineering` | 16,027 | **27.29%** | Lower average urgency due to regular periodic track tamping / ballast cycles. |

### B. Numerical Feature Impact (Mean by Urgency Class)

| Feature | Routine (`urgent=0`) | Urgent (`urgent=1`) | Absolute Difference |
|:---|:---:|:---:|:---:|
| `trains_per_day` | 91.51 | **116.39** | **+24.88 trains/day** |
| `goods_trains_per_day` | 39.03 | **52.21** | **+13.18 trains/day** |
| `days_since_defect` | 22.19 | **12.91** | **-9.28 days** (Urgent defects are detected/acted on sooner) |
| `days_overdue` | 6.57 | **5.32** | Minor difference (non-linear threshold effect) |
| `failures_last_12_months` | 1.20 | **1.45** | **+20.8% higher failures** |
| `num_open_defects` | 2.10 | **2.62** | **+24.8% more open defects** |
| `maintenance_duration_hours` | 3.16 | **3.62** | **+0.46 hours** (Urgent tasks require longer repairs) |

---

## 5. Department-Level Comparative Analysis

| Department | Task Share | Urgency Rate | Avg Trains/Day | Avg Failures (12M) | Avg Duration |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Engineering** | 40.07% | 27.29% | 100.17 | 1.29 | 3.66 hrs |
| **S&T** | 32.23% | 39.98% | 100.35 | 1.28 | 2.50 hrs |
| **Traction** | 27.71% | 42.27% | 100.61 | 1.29 | 3.79 hrs |

- **Key Takeaway**: Engineering tasks require higher average duration (3.66 hrs for rail renewal, bridge bearing, tamping) with a 27.3% urgency rate. S&T tasks require shorter block windows (2.50 hrs) but carry a higher 40.0% urgency rate due to fail-safe interlocking trip hazards.

---

## 6. Correlation & Multicollinearity Analysis

### Linear Correlations with Target (`urgent`)
1. `days_since_defect`: **-0.2954** (Negative correlation: critical defects are captured and escalated quickly).
2. `trains_per_day`: **+0.2188** (Higher route density drives urgent block prioritization).
3. `goods_trains_per_day`: **+0.2168** (Heavy freight loads accelerate urgency).
4. `num_open_defects`: **+0.1855** (Multiple open defects elevate risk).
5. `maintenance_duration_hours`: **+0.1773** (Complex defects require more planning).
6. `previous_failures`: **+0.1126** (Historical breakdown prone assets).

### Inter-Feature Multicollinearity
- **`trains_per_day` $\leftrightarrow$ `goods_trains_per_day`**: $r = 0.9610$ (Extremely high collinearity). Freight traffic is a proportional subset of total traffic. **Recommendation for future preprocessing**: Consider deriving a `freight_ratio` or retaining one of the two to avoid collinearity in linear models.
- **`previous_failures` $\leftrightarrow$ `failures_last_12_months`**: $r = 0.6256$ (Expected lifecycle relationship).
- **`asset_age_years` $\leftrightarrow$ `previous_failures`**: $r = 0.5223$ (Older assets accumulate more historical failures).

---

## 7. Data Leakage Assessment

- **Single-Feature Rule Test**: Checked if any categorical level perfectly segregates `urgent = 1` or `urgent = 0`.
  - `defect_severity == 'CRITICAL'` $\rightarrow$ 55.4% urgent, 44.6% non-urgent.
  - `asset_criticality == 'CRITICAL'` $\rightarrow$ 56.9% urgent, 43.1% non-urgent.
  - `operational_impact == 'CRITICAL'` $\rightarrow$ 63.7% urgent, 36.3% non-urgent.
- **Verdict**: **NO DATA LEAKAGE DETECTED**. No single feature or deterministic combination allows trivial rule-based prediction of `urgent`. Successful prediction will require ML algorithms to capture non-linear feature interactions across severity, criticality, traffic, and history.

---

## 8. Saved Visualizations Catalog

The following 7 charts were generated and saved in `ml/eda/plots/`:
1. `01_target_distribution.png`: Bar chart and pie chart displaying the 64.5% / 35.5% class balance.
2. `02_numerical_feature_distributions.png`: 10-panel KDE and histogram distributions comparing routine vs urgent tasks.
3. `03_target_vs_categorical_features.png`: Crosstab counts for severity, criticality, operational impact, and department.
4. `04_target_vs_numerical_boxplots.png`: Boxplots showing separation on overdue days, defect age, traffic, and open defect clusters.
5. `05_correlation_heatmap.png`: Full Pearson correlation matrix across all numerical variables.
6. `06_department_comparisons.png`: Departmental urgency rates, severity proportions, and operational impact profiles.
7. `07_asset_type_urgency_rates.png`: Ranked horizontal bar chart comparing urgency across all 23 asset types against the dataset average.

---

## 9. Next Steps Summary (For Preprocessing Phase)

When the project transitions to data preprocessing and feature engineering:
1. **Multicollinearity Handling**: Create `freight_traffic_ratio = goods_trains_per_day / trains_per_day` or drop redundant traffic components for linear models.
2. **Categorical Encoding**: Apply Ordinal Encoding to naturally ordered variables (`defect_severity`, `asset_criticality`, `operational_impact`), Target/One-Hot Encoding to nominal variables (`department`, `asset_type`, `defect_type`).
3. **Feature Scaling**: RobustScaler / StandardScaler on skewed heavy-tail variables (`days_since_last_maintenance`, `days_overdue`).
4. **Train/Validation/Test Splitting**: Stratified splitting on `urgent` and grouped by `asset_id` to prevent data leakage across repeated inspections of the same asset.
