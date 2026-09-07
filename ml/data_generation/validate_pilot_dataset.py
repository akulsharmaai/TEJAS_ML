"""
Comprehensive Validation Script for TEJAS Pilot Master Dataset (10K Rows)
Evaluates 9 validation pillars:
1. Missing values and duplicates
2. Value ranges and impossible values
3. Distribution checks (skewness, kurtosis, class balances)
4. Feature-target relationship checks (correlations, mutual information)
5. Combination/consistency checks between related columns
6. Normality/statistical checks
7. Target leakage analysis
8. Counterfactual sanity tests (monotonicity sweeps)
9. Logical consistency between targets and features
"""

import os
import sys
from typing import Any, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Robust path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

candidate_paths = [
    os.path.join(PROJECT_ROOT, "ml", "data", "tejas_pilot_master_dataset.csv"),
    os.path.join(PROJECT_ROOT, "data", "tejas_pilot_master_dataset.csv"),
    os.path.join(SCRIPT_DIR, "..", "data", "tejas_pilot_master_dataset.csv"),
    os.path.join(os.getcwd(), "ml", "data", "tejas_pilot_master_dataset.csv"),
    os.path.join(os.getcwd(), "data", "tejas_pilot_master_dataset.csv"),
]

DATA_PATH = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("STARTING THOROUGH VALIDATION OF TEJAS PILOT MASTER DATASET")
print(f"Dataset path: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
n_rows, n_cols = df.shape
print(f"Loaded dataset: {n_rows} rows, {n_cols} columns")

# -------------------------------------------------------------
# 1. Missing Values & Duplicate Records
# -------------------------------------------------------------
print("\n--- 1. Missing Values & Duplicates ---")
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0]
dup_event_ids = (df['event_id'].duplicated().sum())
dup_exact_rows = int(df.duplicated().sum())
dup_asset_events = int(df.duplicated(subset=['asset_id', 'event_date']).sum())

print(f"Exact duplicate rows: {dup_exact_rows}")
print(f"Duplicate event_ids: {dup_event_ids}")
print(f"Duplicate (asset_id, event_date) combinations: {dup_asset_events}")
print(f"Columns with missing values:\n{null_cols}")

null_df = pd.DataFrame({
    'column': df.columns,
    'null_count': null_counts.to_numpy(),
    'null_percent': (null_counts.to_numpy() / n_rows) * 100.0,
    'dtype': [str(t) for t in df.dtypes]
})
null_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values_summary.csv"), index=False)

# -------------------------------------------------------------
# 2. Value Ranges & Impossible Values
# -------------------------------------------------------------
print("\n--- 2. Value Ranges & Impossible Values ---")
range_issues = []

# Range checks
checks = [
    ("asset_age_years", (df['asset_age_years'] < 0) | (df['asset_age_years'] > 100), "Asset age < 0 or > 100"),
    ("asset_criticality_score", (df['asset_criticality_score'] < 0) | (df['asset_criticality_score'] > 100), "Asset criticality score not in [0, 100]"),
    ("traffic_percentile", (df['traffic_percentile'] < 0.0) | (df['traffic_percentile'] > 1.0), "Traffic percentile not in [0.0, 1.0]"),
    ("network_neighbor_degree", df['network_neighbor_degree'] < 0, "Network neighbor degree < 0"),
    ("affected_services_if_blocked", df['affected_services_if_blocked'] < 0, "Affected services < 0"),
    ("operational_impact_score", (df['operational_impact_score'] < 0) | (df['operational_impact_score'] > 100), "Operational impact score not in [0, 100]"),
    ("maintenance_interval_days", df['maintenance_interval_days'] <= 0, "Maintenance interval days <= 0"),
    ("days_since_last_maintenance", df['days_since_last_maintenance'] < 0, "Days since last maintenance < 0"),
    ("maintenance_overdue_days", df['maintenance_overdue_days'] < 0, "Maintenance overdue days < 0"),
    ("failures_last_30d", df['failures_last_30d'] < 0, "Failures last 30d < 0"),
    ("failures_last_90d", df['failures_last_90d'] < 0, "Failures last 90d < 0"),
    ("failures_last_365d", df['failures_last_365d'] < 0, "Failures last 365d < 0"),
    ("same_defect_recurrences_365d", df['same_defect_recurrences_365d'] < 0, "Recurrences < 0"),
    ("defect_duration_days", df['defect_duration_days'] < 0, "Defect duration < 0"),
    ("maintenance_duration_hours_target", (df['maintenance_duration_hours_target'] <= 0) | (df['maintenance_duration_hours_target'] > 24), "Maintenance duration hours target <= 0 or > 24h"),
    ("priority_score_target", (df['priority_score_target'] < 0) | (df['priority_score_target'] > 100), "Priority score target not in [0, 100]"),
]

for col, mask, desc in checks:
    count = (mask.sum())
    range_issues.append({"check": desc, "column": col, "violating_records": count, "violating_pct": round(count/n_rows*100, 4)})
    if count > 0:
        print(f"  VIOLATION: {desc} -> {count} rows ({count/n_rows*100:.2f}%)")
    else:
        print(f"  PASSED: {desc} -> 0 violations")

range_df = pd.DataFrame(range_issues)
range_df.to_csv(os.path.join(OUTPUT_DIR, "range_checks.csv"), index=False)

# -------------------------------------------------------------
# 3. Temporal & Hierarchical Consistency Checks
# -------------------------------------------------------------
print("\n--- 3. Combination & Consistency Checks ---")
consistency_issues = []

# A. Failure hierarchy: 30d <= 90d <= 365d
viol_f30_f90 = ((df['failures_last_30d'] > df['failures_last_90d']).sum())
viol_f90_f365 = ((df['failures_last_90d'] > df['failures_last_365d']).sum())
viol_rec_f365 = ((df['same_defect_recurrences_365d'] > df['failures_last_365d']).sum())

consistency_issues.append({
    "check": "Failures last 30d <= Failures last 90d",
    "violations": viol_f30_f90,
    "status": "PASS" if viol_f30_f90 == 0 else "FAIL"
})
consistency_issues.append({
    "check": "Failures last 90d <= Failures last 365d",
    "violations": viol_f90_f365,
    "status": "PASS" if viol_f90_f365 == 0 else "FAIL"
})
consistency_issues.append({
    "check": "Same defect recurrences 365d <= Failures last 365d",
    "violations": viol_rec_f365,
    "status": "PASS" if viol_rec_f365 == 0 else "WARNING"
})

# B. Maintenance Overdue Consistency:
# maintenance_overdue_days should theoretically equal max(0, days_since_last_maintenance - maintenance_interval_days)
computed_overdue = np.maximum(0, df['days_since_last_maintenance'] - df['maintenance_interval_days'])
overdue_diff = (df['maintenance_overdue_days'] - computed_overdue).abs()
overdue_mismatch = int((overdue_diff > 0).sum())
overdue_max_diff = float(overdue_diff.max())
print(f"Maintenance overdue formula exact match: {overdue_mismatch} mismatches (max diff: {overdue_max_diff})")
consistency_issues.append({
    "check": "maintenance_overdue_days == max(0, days_since_last_maintenance - maintenance_interval_days)",
    "violations": overdue_mismatch,
    "status": "PASS" if overdue_mismatch == 0 else "WARNING"
})

# C. Traffic Consistency:
# affected_services_if_blocked should be <= scheduled_services_count_proxy
viol_traffic = ((df['affected_services_if_blocked'] > df['scheduled_services_count_proxy']).sum())
print(f"Affected services <= scheduled services: {viol_traffic} violations")
consistency_issues.append({
    "check": "affected_services_if_blocked <= scheduled_services_count_proxy",
    "violations": viol_traffic,
    "status": "PASS" if viol_traffic == 0 else "FAIL"
})

# D. Priority Score vs Priority Class Consistency
class_score_stats = df.groupby('priority_class_target')['priority_score_target'].agg(['min', 'max', 'mean', 'median', 'std']).reset_index()
print(f"\nPriority Class vs Priority Score Alignment:\n{class_score_stats}")
class_score_stats.to_csv(os.path.join(OUTPUT_DIR, "priority_class_score_alignment.csv"), index=False)

# Check overlaps between classes
low_max = df[df['priority_class_target'] == 'LOW']['priority_score_target'].max()
med_min = df[df['priority_class_target'] == 'MEDIUM']['priority_score_target'].min()
med_max = df[df['priority_class_target'] == 'MEDIUM']['priority_score_target'].max()
high_min = df[df['priority_class_target'] == 'HIGH']['priority_score_target'].min()
high_max = df[df['priority_class_target'] == 'HIGH']['priority_score_target'].max()
crit_min = df[df['priority_class_target'] == 'CRITICAL']['priority_score_target'].min()

print(f"LOW max score: {low_max}, MEDIUM min: {med_min}, MEDIUM max: {med_max}")
print(f"HIGH min score: {high_min}, HIGH max: {high_max}, CRITICAL min: {crit_min}")

score_order_ok = (low_max <= med_min or df[df['priority_class_target'] == 'LOW']['priority_score_target'].mean() < df[df['priority_class_target'] == 'MEDIUM']['priority_score_target'].mean() < df[df['priority_class_target'] == 'HIGH']['priority_score_target'].mean() < df[df['priority_class_target'] == 'CRITICAL']['priority_score_target'].mean())
consistency_issues.append({
    "check": "Monotonic mean priority score ordering (LOW < MEDIUM < HIGH < CRITICAL)",
    "violations": 0 if score_order_ok else 1,
    "status": "PASS" if score_order_ok else "FAIL"
})

# E. Criticality Category vs Criticality Score Alignment
crit_stats = df.groupby('asset_criticality')['asset_criticality_score'].agg(['min', 'max', 'mean', 'median', 'std']).reset_index()
print(f"\nAsset Criticality Label vs Score:\n{crit_stats}")
crit_stats.to_csv(os.path.join(OUTPUT_DIR, "asset_criticality_score_alignment.csv"), index=False)

# F. Defect Severity Label vs Defect Duration / Targets
sev_stats = df.groupby('defect_severity_label').agg({
    'failure_within_30d_target': 'mean',
    'block_required_target': 'mean',
    'priority_score_target': 'mean',
    'maintenance_duration_hours_target': 'mean'
}).reset_index()
print(f"\nDefect Severity vs Target Means:\n{sev_stats}")
sev_stats.to_csv(os.path.join(OUTPUT_DIR, "defect_severity_target_alignment.csv"), index=False)

consistency_df = pd.DataFrame(consistency_issues)
consistency_df.to_csv(os.path.join(OUTPUT_DIR, "consistency_checks.csv"), index=False)

# -------------------------------------------------------------
# 4. Statistical Distributions, Skewness, Kurtosis
# -------------------------------------------------------------
print("\n--- 4. Statistical & Distribution Checks ---")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
stat_rows = []
for c in numeric_cols:
    data = df[c].dropna()
    sk = stats.skew(data)
    kt = stats.kurtosis(data)
    # Shapiro-Wilk on sample of 5000 (Shapiro limit)
    sample_data = data.sample(min(5000, len(data)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
    stat_rows.append({
        "feature": c,
        "mean": round(data.mean(), 4),
        "std": round(data.std(), 4),
        "median": round(data.median(), 4),
        "min": round(data.min(), 4),
        "max": round(data.max(), 4),
        "skewness": round(float(sk), 4) if pd.notnull(sk) else 0.0,
        "kurtosis": round(float(kt), 4) if pd.notnull(kt) else 0.0,
        "shapiro_stat": round(float(shapiro_stat), 4) if pd.notnull(shapiro_stat) else 0.0,
        "is_gaussian": bool(shapiro_p > 0.05) if pd.notnull(shapiro_p) else False
    })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_distributions.csv"), index=False)

# -------------------------------------------------------------
# 5. Feature-Target Relationships & Correlation / Leakage Check
# -------------------------------------------------------------
print("\n--- 5. Feature-Target Correlations & Leakage ---")
# Encode categorical columns numerically for correlation matrix
df_corr = df.copy()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()
for c in cat_cols:
    if df_corr[c].nunique() < 50:
        df_corr[c] = df_corr[c].astype('category').cat.codes

target_cols = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

# Pearson & Spearman correlations of features with targets
corr_pearson = df_corr.select_dtypes(include='number').corr(method='pearson')
corr_spearman = df_corr.select_dtypes(include='number').corr(method='spearman')

target_corr_rows = []
for t in ['failure_within_30d_target', 'maintenance_duration_hours_target', 'priority_score_target', 'block_required_target']:
    if t in corr_pearson.columns:
        for f in df_corr.select_dtypes(include='number').columns:
            if f not in target_cols:
                p_raw = corr_pearson.loc[f, t]
                s_raw = corr_spearman.loc[f, t]
                p_val = float(cast(Any, p_raw)) if pd.notnull(p_raw) else 0.0
                s_val = float(cast(Any, s_raw)) if pd.notnull(s_raw) else 0.0
                abs_p = abs(p_val)
                target_corr_rows.append({
                    "target": t,
                    "feature": f,
                    "pearson_corr": round(p_val, 4),
                    "spearman_corr": round(s_val, 4),
                    "abs_pearson": round(abs_p, 4),
                    "leakage_risk": "HIGH" if abs_p > 0.85 else ("MEDIUM" if abs_p > 0.60 else "LOW")
                })

corr_df = pd.DataFrame(target_corr_rows)
corr_df.sort_values(by="abs_pearson", ascending=False).to_csv(os.path.join(OUTPUT_DIR, "feature_target_correlations.csv"), index=False)

print("\nTop Correlations with failure_within_30d_target:")
print(corr_df[corr_df['target'] == 'failure_within_30d_target'].sort_values(by='abs_pearson', ascending=False).head(8)[['feature', 'pearson_corr', 'leakage_risk']])

print("\nTop Correlations with priority_score_target:")
print(corr_df[corr_df['target'] == 'priority_score_target'].sort_values(by='abs_pearson', ascending=False).head(8)[['feature', 'pearson_corr', 'leakage_risk']])

print("\nTop Correlations with block_required_target:")
print(corr_df[corr_df['target'] == 'block_required_target'].sort_values(by='abs_pearson', ascending=False).head(8)[['feature', 'pearson_corr', 'leakage_risk']])

# -------------------------------------------------------------
# 6. Mutual Information & Non-Linear Leakage
# -------------------------------------------------------------
print("\n--- 6. Mutual Information Checks ---")
feature_numeric_for_mi = df_corr.select_dtypes(include='number').drop(columns=[t for t in target_cols if t in df_corr.columns] + ['priority_class_target'], errors='ignore').fillna(0)

mi_failure = mutual_info_classif(feature_numeric_for_mi, df['failure_within_30d_target'].astype(int), random_state=42)
mi_block = mutual_info_classif(feature_numeric_for_mi, df['block_required_target'].astype(int), random_state=42)
mi_priority = mutual_info_regression(feature_numeric_for_mi, df['priority_score_target'], random_state=42)

mi_df = pd.DataFrame({
    "feature": feature_numeric_for_mi.columns,
    "mi_failure_30d": mi_failure,
    "mi_block_required": mi_block,
    "mi_priority_score": mi_priority
}).sort_values(by="mi_priority_score", ascending=False)
mi_df.to_csv(os.path.join(OUTPUT_DIR, "mutual_information.csv"), index=False)

# -------------------------------------------------------------
# 7. Counterfactual Sanity Tests (Monotonicity Checks)
# -------------------------------------------------------------
print("\n--- 7. Counterfactual & Monotonicity Sanity Tests ---")
cf_results = []

# A. Monotonicity of defect_severity_label on failure_within_30d_target
sev_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
sev_fail_rate = [(df[df['defect_severity_label'] == s]['failure_within_30d_target'].mean()) for s in sev_order]
sev_mono = all(sev_fail_rate[i] <= sev_fail_rate[i+1] for i in range(len(sev_fail_rate)-1))
cf_results.append({
    "test_name": "Defect Severity Monotonicity on Failure Risk (LOW -> CRITICAL)",
    "rates": str([round(r, 4) for r in sev_fail_rate]),
    "is_monotonic": sev_mono,
    "status": "PASS" if sev_mono else "FAIL"
})

# B. Monotonicity of asset_criticality on priority_score_target
crit_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
crit_prio_mean = [(df[df['asset_criticality'] == c]['priority_score_target'].mean()) for c in crit_order]
crit_mono = all(crit_prio_mean[i] <= crit_prio_mean[i+1] for i in range(len(crit_prio_mean)-1))
cf_results.append({
    "test_name": "Asset Criticality Monotonicity on Priority Score (LOW -> CRITICAL)",
    "rates": str([round(r, 2) for r in crit_prio_mean]),
    "is_monotonic": crit_mono,
    "status": "PASS" if crit_mono else "FAIL"
})

# C. Monotonicity of overdue buckets on failure risk
df['overdue_bucket'] = pd.cut(df['maintenance_overdue_days'], bins=[-1, 0, 15, 45, 100, 1000], labels=['0_OnTime', '1_15d', '16_45d', '46_100d', '100d+'])
overdue_fail_rate = [float(x) for x in df.groupby('overdue_bucket', observed=False)['failure_within_30d_target'].mean().tolist()]
overdue_mono = all(overdue_fail_rate[i] <= overdue_fail_rate[i+1] for i in range(len(overdue_fail_rate)-1))
cf_results.append({
    "test_name": "Maintenance Overdue Monotonicity on Failure Risk",
    "rates": str([round(r, 4) for r in overdue_fail_rate]),
    "is_monotonic": overdue_mono,
    "status": "PASS" if overdue_mono else "FAIL"
})

# D. Block Requirement vs Scheduled Services / Impact
df['impact_bucket'] = pd.qcut(df['operational_impact_score'], q=4, labels=['Q1_Low', 'Q2_Med', 'Q3_High', 'Q4_Crit'])
impact_block_rate = [float(x) for x in df.groupby('impact_bucket', observed=False)['block_required_target'].mean().tolist()]
impact_mono = all(impact_block_rate[i] <= impact_block_rate[i+1] for i in range(len(impact_block_rate)-1))
cf_results.append({
    "test_name": "Operational Impact Monotonicity on Block Requirement",
    "rates": str([round(r, 4) for r in impact_block_rate]),
    "is_monotonic": impact_mono,
    "status": "PASS" if impact_mono else "FAIL"
})

cf_df = pd.DataFrame(cf_results)
cf_df.to_csv(os.path.join(OUTPUT_DIR, "counterfactual_monotonicity.csv"), index=False)
print(cf_df)

# -------------------------------------------------------------
# 8. Department & Asset Representation Checks
# -------------------------------------------------------------
print("\n--- 8. Department & Asset Hierarchy ---")
dept_counts = df['department'].value_counts(normalize=True) * 100
asset_per_dept = df.groupby('department')['asset_type'].nunique()
print("Department share (%):\n", dept_counts)
print("Asset types per department:\n", asset_per_dept)

dept_asset_df = df.groupby(['department', 'asset_type']).agg({
    'event_id': 'count',
    'failure_within_30d_target': 'mean',
    'block_required_target': 'mean',
    'priority_score_target': 'mean',
    'maintenance_duration_hours_target': 'mean'
}).reset_index().rename(columns={'event_id': 'record_count'})
dept_asset_df.to_csv(os.path.join(OUTPUT_DIR, "department_asset_distribution.csv"), index=False)

# -------------------------------------------------------------
# 9. Generate High-Quality Visualizations
# -------------------------------------------------------------
print("\n--- 9. Generating Validation Visualizations ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Target Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("TEJAS Pilot Dataset (10K) - Target Distributions", fontsize=16, fontweight='bold')

# Failure within 30d
sns.countplot(ax=axes[0, 0], x='failure_within_30d_target', data=df, hue='failure_within_30d_target', palette=['#4CAF50', '#F44336'], legend=False)
axes[0, 0].set_title("Target 1: Failure Within 30 Days (Binary)")
axes[0, 0].set_xlabel("Failure within 30 Days")
axes[0, 0].set_ylabel("Record Count")
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{int(p.get_height()):,} ({p.get_height()/n_rows*100:.1f}%)", (p.get_x() + p.get_width()/2., p.get_height()/2), ha='center', color='white', fontweight='bold')

# Block Required
sns.countplot(ax=axes[0, 1], x='block_required_target', data=df, hue='block_required_target', palette=['#2196F3', '#FF9800'], legend=False)
axes[0, 1].set_title("Target 2: Traffic/Power Block Required (Binary)")
axes[0, 1].set_xlabel("Block Required")
axes[0, 1].set_ylabel("Record Count")
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{int(p.get_height()):,} ({p.get_height()/n_rows*100:.1f}%)", (p.get_x() + p.get_width()/2., p.get_height()/2), ha='center', color='white', fontweight='bold')

# Priority Class
prio_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
sns.countplot(ax=axes[1, 0], x='priority_class_target', data=df, order=prio_order, hue='priority_class_target', palette=['#8BC34A', '#FFC107', '#FF9800', '#D32F2F'], legend=False)
axes[1, 0].set_title("Target 3: Priority Class Target (4-Class Ordinal)")
axes[1, 0].set_xlabel("Priority Class")
axes[1, 0].set_ylabel("Record Count")
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f"{int(p.get_height()):,} ({p.get_height()/n_rows*100:.1f}%)", (p.get_x() + p.get_width()/2., p.get_height()/2), ha='center', color='black' if p.get_height() > 3000 else 'white', fontweight='bold')

# Maintenance Duration Hours Target
sns.histplot(ax=axes[1, 1], data=df, x='maintenance_duration_hours_target', kde=True, color='#673AB7', bins=30)
axes[1, 1].set_title("Target 4: Maintenance Duration Hours (Continuous Regression)")
axes[1, 1].set_xlabel("Duration (Hours)")
axes[1, 1].set_ylabel("Density / Count")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "target_distributions.png"), dpi=200)
plt.close()

# Figure 2: Priority Score vs Class Alignment & Consistency
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='priority_class_target', y='priority_score_target', order=prio_order, hue='priority_class_target', palette=['#8BC34A', '#FFC107', '#FF9800', '#D32F2F'], legend=False, ax=ax)
sns.stripplot(data=df.sample(500, random_state=42), x='priority_class_target', y='priority_score_target', order=prio_order, color='black', alpha=0.3, jitter=0.2, ax=ax)
ax.set_title("Priority Score (0-100) Distribution Across Priority Classes", fontsize=14, fontweight='bold')
ax.set_xlabel("Priority Class Target")
ax.set_ylabel("Priority Score Target")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "priority_score_vs_class.png"), dpi=200)
plt.close()

# Figure 3: Correlation Heatmap of Numeric & Target Features
fig, ax = plt.subplots(figsize=(14, 12))
numeric_subset = [
    'asset_age_years', 'asset_criticality_score', 'traffic_percentile',
    'network_neighbor_degree', 'affected_services_if_blocked', 'operational_impact_score',
    'maintenance_interval_days', 'days_since_last_maintenance', 'maintenance_overdue_days',
    'failures_last_30d', 'failures_last_90d', 'failures_last_365d', 'same_defect_recurrences_365d',
    'defect_duration_days', 'failure_within_30d_target', 'maintenance_duration_hours_target',
    'priority_score_target', 'block_required_target'
]
corr_sub = df_corr[numeric_subset].corr()
sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title("Correlation Matrix of Numerical Features and Targets", fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png"), dpi=200)
plt.close()

# Figure 4: Counterfactual Risk Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Counterfactual Monotonicity & Risk Gradient Verification", fontsize=15, fontweight='bold')

# Defect Severity vs Failure Rate
sev_df = df.groupby('defect_severity_label')['failure_within_30d_target'].mean().reindex(sev_order).reset_index()
sns.barplot(data=sev_df, x='defect_severity_label', y='failure_within_30d_target', hue='defect_severity_label', palette='Reds', legend=False, ax=axes[0])
axes[0].set_title("Failure Risk vs Defect Severity")
axes[0].set_ylabel("30-Day Failure Probability")
axes[0].set_xlabel("Defect Severity Label")
axes[0].set_ylim(0, 1.0)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2., p.get_height()+0.02), ha='center', fontweight='bold')

# Asset Criticality vs Priority Score
crit_df = df.groupby('asset_criticality')['priority_score_target'].mean().reindex(crit_order).reset_index()
sns.barplot(data=crit_df, x='asset_criticality', y='priority_score_target', hue='asset_criticality', palette='Oranges', legend=False, ax=axes[1])
axes[1].set_title("Priority Score vs Asset Criticality")
axes[1].set_ylabel("Mean Priority Score (0-100)")
axes[1].set_xlabel("Asset Criticality")
axes[1].set_ylim(0, 100)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width()/2., p.get_height()+1.5), ha='center', fontweight='bold')

# Overdue Days vs Failure Rate
overdue_plot_df = df.groupby('overdue_bucket', observed=False)['failure_within_30d_target'].mean().reset_index()
sns.barplot(data=overdue_plot_df, x='overdue_bucket', y='failure_within_30d_target', hue='overdue_bucket', palette='Blues', legend=False, ax=axes[2])
axes[2].set_title("Failure Risk vs Maintenance Overdue Days")
axes[2].set_ylabel("30-Day Failure Probability")
axes[2].set_xlabel("Overdue Range")
axes[2].set_ylim(0, 1.0)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2., p.get_height()+0.02), ha='center', fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "counterfactual_risk_curves.png"), dpi=200)
plt.close()

# Figure 5: Maintenance Overdue vs Elapsed Days Temporal Consistency
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df.sample(2000, random_state=42), x='days_since_last_maintenance', y='maintenance_overdue_days', hue='maintenance_interval_days', palette='tab10', alpha=0.6, ax=ax)
ax.set_title("Temporal Relationship: Days Since Last Maintenance vs Overdue Days", fontsize=14, fontweight='bold')
ax.set_xlabel("Days Since Last Maintenance")
ax.set_ylabel("Maintenance Overdue Days")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "maintenance_temporal_consistency.png"), dpi=200)
plt.close()

print("\nValidation computation and artifacts generated successfully!")
