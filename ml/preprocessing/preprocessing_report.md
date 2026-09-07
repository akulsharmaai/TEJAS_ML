# Preprocessing and Feature Engineering Report

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Phase**: Step 7 – Preprocessing & Feature Engineering  
**Dataset Input**: `data/processed/railway_maintenance_tasks.csv` (40,000 records × 20 columns)  
**Output Location**: `ml/preprocessing/` and `ml/data/processed/`  

---

## 1. Features Removed

| Feature Name | Type | Rationale for Removal |
|:---|:---:|:---|
| `task_id` | Alphanumeric Identifier | High cardinality (40,000 unique values); provides zero predictive generalizability and risks identity memorization. |
| `asset_id` | Alphanumeric Identifier | Physical asset tag (10,703 unique values). Used exclusively for **Stratified Group Splitting** to prevent data leakage, then removed from model inputs. |
| `goods_trains_per_day` | Numerical (Count) | Extreme collinearity with `trains_per_day` ($r = 0.9610$). Replaced by domain-engineered `freight_traffic_ratio`. |

---

## 2. Features Retained & Base Transformations

All 16 remaining core domain features were retained and transformed according to their physical and statistical characteristics:

- **Nominal Categoricals** (4): `department`, `asset_type`, `defect_type`, `maintenance_frequency`
- **Ordinal Categoricals** (3): `defect_severity`, `asset_criticality`, `operational_impact`
- **Continuous / Discrete Numerical** (9): `asset_age_years`, `num_open_defects`, `days_since_defect`, `days_overdue`, `days_since_last_maintenance`, `previous_failures`, `failures_last_12_months`, `trains_per_day`, `maintenance_duration_hours`

---

## 3. Features Engineered

| Engineered Feature | Mathematical Formulation | Safe Division / Boundary Handling | Domain & ML Justification |
|:---|:---:|:---:|:---|
| `freight_traffic_ratio` | $\frac{\text{goods\_trains\_per\_day}}{\max(\text{trains\_per\_day}, 1)}$ | Clamped to $[0.0, 1.0]$; $\max(\cdot, 1)$ prevents zero-division | Decouples freight proportion from total line capacity; resolves severe multicollinearity ($r=0.961$) while preserving axle-load impact. |
| `failure_rate_indicator` | $\frac{\text{failures\_last\_12\_months}}{\max(\text{asset\_age\_years}, 0.5)}$ | Lower bounded at $0.5$ years | Computes annualized recent failure velocity. Differentiates normal wear on older assets from acute infant mortality / recurrent breakdown spikes on newer assets. |
| `maintenance_overdue_ratio` | $\frac{\text{days\_overdue}}{\text{maintenance\_interval\_days}}$ | Mandatory cycle days mapped from `maintenance_frequency` (Daily=1 to Annual=365) | Normalizes overdue duration against the asset's required inspection cycle (e.g. 10 days overdue on a Weekly task represents a 1.43× cycle violation, whereas on an Annual task it is only 0.027×). |

---

## 4. Encoding Strategy

### A. Nominal Categoricals $\rightarrow$ One-Hot Encoding
- **Encoder**: `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`
- **Features Encoded**:
  - `department` (3 categories $\rightarrow$ 3 columns)
  - `asset_type` (23 categories $\rightarrow$ 23 columns)
  - `defect_type` (90+ distinct failure modes $\rightarrow$ 90+ columns)
  - `maintenance_frequency` (6 categories $\rightarrow$ 6 columns)
- **Total One-Hot Columns**: 122 binary indicators.
- **Handling Unknowns**: `handle_unknown='ignore'` ensures zero-vector representation if an unseen defect appears in test/production data.

### B. Ordinal Categoricals $\rightarrow$ Explicit Monotonic Integer Mapping
Ordinal variables possess a natural, strictly increasing physical severity hierarchy. To preserve monotonicity without arbitrary distance distortions or target leakage, they are explicitly mapped as:
- `LOW` $\rightarrow 0$
- `MEDIUM` $\rightarrow 1$
- `HIGH` $\rightarrow 2$
- `CRITICAL` $\rightarrow 3$
- **Features Mapped**: `defect_severity`, `asset_criticality`, `operational_impact`.
- **Target Encoding**: **Strictly avoided** to eliminate label leakage.

---

## 5. Scaling Strategy

To cater to both linear models (Logistic Regression, SVM) and tree-based ensembles (Random Forest, XGBoost), numerical variables are scaled conditionally based on their empirical distribution shapes from EDA:

### A. Heavy-Tailed & Skewed Features $\rightarrow$ `RobustScaler`
- **Method**: Centered on median, scaled by Interquartile Range ($\text{IQR} = Q_3 - Q_1$). Immune to extreme maintenance backlogs and long right tails.
- **Features Scaled**:
  - `days_overdue` (Skew: +2.23)
  - `days_since_last_maintenance` (Skew: +1.58)
  - `days_since_defect` (Skew: +1.43)
  - `maintenance_overdue_ratio` (Skew: +2.18)
  - `failure_rate_indicator` (Skew: +1.95)

### B. Moderately Symmetric & Continuous Features $\rightarrow$ `StandardScaler`
- **Method**: Zero-mean ($\mu = 0$), unit variance ($\sigma = 1$).
- **Features Scaled**:
  - `asset_age_years`
  - `num_open_defects`
  - `previous_failures`
  - `failures_last_12_months`
  - `trains_per_day`
  - `freight_traffic_ratio`
  - `maintenance_duration_hours`

---

## 6. Multicollinearity Decisions

1. **`trains_per_day` $\leftrightarrow$ `goods_trains_per_day` ($r = 0.9610$)**:
   - **Decision**: Dropped raw `goods_trains_per_day` from the feature matrix; replaced with `freight_traffic_ratio`.
   - **Result**: Preserves total line congestion (`trains_per_day`) alongside freight intensity (`freight_traffic_ratio`), reducing inter-feature correlation from **0.9610 to 0.0824**.
2. **`previous_failures` $\leftrightarrow$ `failures_last_12_months` ($r = 0.6256$)**:
   - **Decision**: Retained both features.
   - **Justification**: Lifetime failure count captures cumulative structural fatigue (MTBF), whereas 12-month failures capture acute operational instability.
3. **`asset_age_years` $\leftrightarrow$ `previous_failures` ($r = 0.5223$)**:
   - **Decision**: Retained both features.
   - **Justification**: Age reflects physical service life; failure count reflects asset maintenance quality. An old asset with few failures has high reliability, while a young asset with multiple failures indicates defective batch fabrication.

---

## 7. Train / Validation / Test Splitting

To prevent asset-level memorization and realistic out-of-sample evaluation:
- **Strategy**: `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` grouped on `asset_id` and stratified on `urgent`.
- **Folds Allocated**: 14 Folds (70%) $\rightarrow$ Train; 3 Folds (15%) $\rightarrow$ Validation; 3 Folds (15%) $\rightarrow$ Test.

| Split Dataset | Record Count | Percentage | Class 0 (Routine) | Class 1 (Urgent) | Urgency Proportion | Unique Assets |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Set** | 28,006 | 70.02% | 18,054 | 9,952 | **35.54%** | 7,493 |
| **Validation Set** | 5,997 | 14.99% | 3,867 | 2,130 | **35.52%** | 1,605 |
| **Test Set** | 5,997 | 14.99% | 3,866 | 2,131 | **35.53%** | 1,605 |
| **Total** | 40,000 | 100.00% | 25,787 | 14,213 | **35.53%** | 10,703 |

---

## 8. Data Leakage & Integrity Verification

1. **Transformer Fit Boundary**: All `OneHotEncoder`, `RobustScaler`, and `StandardScaler` transformations were **FIT STRICTLY on `X_train`** and applied to `X_val` and `X_test` using `.transform()`.
2. **Zero Asset Overlap**:
   - $\text{Train Assets} \cap \text{Val Assets} = \emptyset$ (0 overlap)
   - $\text{Train Assets} \cap \text{Test Assets} = \emptyset$ (0 overlap)
   - $\text{Val Assets} \cap \text{Test Assets} = \emptyset$ (0 overlap)
3. **No Target Leakage**: Target `urgent` was isolated prior to feature engineering and transformer fitting.
4. **Data Cleanness**:
   - Null count across all splits: **0**
   - Infinite values across all splits: **0**

---

## 9. Final Feature Dimensions

- **Total Model Features**: **137 features**
  - Nominal One-Hot Features: **122**
  - Scaled Numerical Features: **12** (5 RobustScaled, 7 StandardScaled)
  - Ordinal Numerical Features: **3** (Severity, Criticality, Impact)
- **Target Variable**: `urgent` (Binary: 0 or 1)

---

## 10. Saved Pipeline Artifacts

- **Pipeline Script**: [ml/preprocessing/preprocess.py](file:///c:/Users/acous/Desktop/sih2026/ml/preprocessing/preprocess.py)
- **Serialized Transformer Pipeline**: `ml/preprocessing/preprocessing_pipeline.pkl`
- **Training Dataset**: [ml/data/processed/train.csv](file:///c:/Users/acous/Desktop/sih2026/ml/data/processed/train.csv) `(28006, 138)`
- **Validation Dataset**: [ml/data/processed/validation.csv](file:///c:/Users/acous/Desktop/sih2026/ml/data/processed/validation.csv) `(5997, 138)`
- **Test Dataset**: [ml/data/processed/test.csv](file:///c:/Users/acous/Desktop/sih2026/ml/data/processed/test.csv) `(5997, 138)`
