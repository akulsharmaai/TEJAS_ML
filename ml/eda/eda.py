"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Comprehensive Exploratory Data Analysis (EDA) Pipeline

Analyzes data/processed/railway_maintenance_tasks.csv without modifying,
cleaning, or training models. Generates statistical summaries and visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'Arial', 'figure.autolayout': True})

os.makedirs('eda/plots', exist_ok=True)
DATA_PATH = 'data/processed/railway_maintenance_tasks.csv'

def run_eda():
    print("Loading dataset for EDA...")
    df = pd.read_csv(DATA_PATH)
    
    # ---------------------------------------------------------
    # 1. Dataset Overview
    # ---------------------------------------------------------
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
    
    # ---------------------------------------------------------
    # 2. Data Quality Checks
    # ---------------------------------------------------------
    missing_vals = df.isnull().sum()
    duplicates = df.duplicated().sum()
    unique_task_ids = df['task_id'].nunique()
    unique_asset_ids = df['asset_id'].nunique()
    
    # Range / physical validity checks
    invalid_trains = (df['goods_trains_per_day'] > df['trains_per_day']).sum()
    invalid_negatives = (df[num_cols] < 0).sum().to_dict()
    
    # Outlier detection via IQR
    outlier_summary = {}
    for col in [c for c in num_cols if c != 'urgent']:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_summary[col] = {
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "outliers_count": int(outliers),
            "outliers_pct": round(float(outliers / len(df) * 100), 2)
        }
        
    # ---------------------------------------------------------
    # 3. Target Distribution
    # ---------------------------------------------------------
    urgent_counts = df['urgent'].value_counts()
    urgent_pcts = df['urgent'].value_counts(normalize=True) * 100
    
    # ---------------------------------------------------------
    # 4. Target vs Features Breakdown
    # ---------------------------------------------------------
    cat_target_rates = {}
    for col in ['defect_severity', 'asset_criticality', 'operational_impact', 'department', 'maintenance_frequency']:
        rate = df.groupby(col, observed=True)['urgent'].agg(['count', 'mean']).rename(columns={'mean': 'urgency_rate'})
        rate['urgency_rate_pct'] = rate['urgency_rate'] * 100
        cat_target_rates[col] = rate
        
    # Numerical target separation stats
    num_target_comparison = df.groupby('urgent')[num_cols].mean().T
    num_target_comparison.columns = ['urgent_0_mean', 'urgent_1_mean']
    num_target_comparison['relative_diff_pct'] = ((num_target_comparison['urgent_1_mean'] - num_target_comparison['urgent_0_mean']) / num_target_comparison['urgent_0_mean']) * 100
    
    # ---------------------------------------------------------
    # 5. Department Breakdown
    # ---------------------------------------------------------
    dept_stats = df.groupby('department', observed=True).agg({
        'urgent': ['count', 'mean'],
        'trains_per_day': 'mean',
        'previous_failures': 'mean',
        'failures_last_12_months': 'mean',
        'maintenance_duration_hours': 'mean',
        'asset_age_years': 'mean'
    })
    
    # ---------------------------------------------------------
    # 6. Correlation Analysis
    # ---------------------------------------------------------
    corr_matrix = df[num_cols].corr()
    
    # Find top correlation pairs with target
    target_corrs = corr_matrix['urgent'].drop('urgent').sort_values(ascending=False)
    
    # Top inter-feature correlations
    corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    
    # ---------------------------------------------------------
    # 7. Potential Data Leakage Analysis
    # ---------------------------------------------------------
    # Check max individual feature AUC or perfect single-variable separation
    single_rule_severities = df.groupby('defect_severity')['urgent'].value_counts(normalize=True).unstack().fillna(0)
    single_rule_criticalities = df.groupby('asset_criticality')['urgent'].value_counts(normalize=True).unstack().fillna(0)
    
    # ---------------------------------------------------------
    # 8. Visualizations Generation
    # ---------------------------------------------------------
    print("Generating visualizations...")
    
    # Plot 1: Target Distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x='urgent', palette=['#3b82f6', '#ef4444'], ax=ax[0])
    ax[0].set_title('Target Class Counts (urgent: 0 vs 1)', fontsize=13, fontweight='bold')
    ax[0].set_xlabel('Urgent Label')
    ax[0].set_ylabel('Record Count')
    for p in ax[0].patches:
        ax[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
        
    ax[1].pie([urgent_counts[0], urgent_counts[1]], labels=['0: Routine (64.5%)', '1: Urgent (35.5%)'],
              autopct='%1.1f%%', startangle=140, colors=['#3b82f6', '#ef4444'], explode=(0, 0.06))
    ax[1].set_title('Target Class Proportions', fontsize=13, fontweight='bold')
    plt.savefig('eda/plots/01_target_distribution.png', dpi=300)
    plt.close()
    
    # Plot 2: Numerical Distributions
    viz_num_cols = ['asset_age_years', 'num_open_defects', 'days_since_defect', 'days_overdue',
                    'days_since_last_maintenance', 'previous_failures', 'failures_last_12_months',
                    'trains_per_day', 'goods_trains_per_day', 'maintenance_duration_hours']
    
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.flatten()
    for i, col in enumerate(viz_num_cols):
        sns.histplot(data=df, x=col, hue='urgent', bins=30, kde=True, palette=['#3b82f6', '#ef4444'],
                     alpha=0.6, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
    plt.suptitle('Numerical Feature Distributions by Urgency Class', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('eda/plots/02_numerical_feature_distributions.png', dpi=300)
    plt.close()
    
    # Plot 3: Target vs Key Categoricals
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sns.countplot(data=df, x='defect_severity', hue='urgent', order=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                  palette=['#3b82f6', '#ef4444'], ax=axes[0, 0])
    axes[0, 0].set_title('Defect Severity vs Urgency', fontweight='bold')
    
    sns.countplot(data=df, x='asset_criticality', hue='urgent', order=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                  palette=['#3b82f6', '#ef4444'], ax=axes[0, 1])
    axes[0, 1].set_title('Asset Criticality vs Urgency', fontweight='bold')
    
    sns.countplot(data=df, x='operational_impact', hue='urgent', order=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                  palette=['#3b82f6', '#ef4444'], ax=axes[1, 0])
    axes[1, 0].set_title('Operational Impact vs Urgency', fontweight='bold')
    
    sns.countplot(data=df, x='department', hue='urgent', palette=['#3b82f6', '#ef4444'], ax=axes[1, 1])
    axes[1, 1].set_title('Department vs Urgency', fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/plots/03_target_vs_categorical_features.png', dpi=300)
    plt.close()
    
    # Plot 4: Target vs Numerical Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    bp_cols = ['days_overdue', 'days_since_defect', 'failures_last_12_months', 'trains_per_day', 'num_open_defects', 'asset_age_years']
    for idx, col in enumerate(bp_cols):
        sns.boxplot(data=df, x='urgent', y=col, palette=['#3b82f6', '#ef4444'], ax=axes[idx], showmeans=True)
        axes[idx].set_title(f'{col} by Urgency', fontweight='bold')
        axes[idx].set_xticklabels(['0 (Routine)', '1 (Urgent)'])
    plt.suptitle('Key Numerical Drivers Grouped by Urgency Target', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('eda/plots/04_target_vs_numerical_boxplots.png', dpi=300)
    plt.close()
    
    # Plot 5: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.4, vmax=1.0,
                linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax, mask=mask)
    plt.title('Numerical Features Correlation Matrix (Pearson)', fontsize=15, fontweight='bold', pad=15)
    plt.savefig('eda/plots/05_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # Plot 6: Department Level Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Urgency rate by dept
    dept_urg_rate = df.groupby('department')['urgent'].mean().reset_index()
    sns.barplot(data=dept_urg_rate, x='department', y='urgent', palette='Blues_r', ax=axes[0])
    axes[0].set_title('Urgency Rate by Department', fontweight='bold')
    axes[0].set_ylabel('Proportion Urgent (0-1)')
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points')
        
    # Severity by dept
    dept_sev = pd.crosstab(df['department'], df['defect_severity'], normalize='index')
    dept_sev[['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']].plot(kind='bar', stacked=True, colormap='Spectral_r', ax=axes[1])
    axes[1].set_title('Defect Severity Breakdown by Department', fontweight='bold')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Severity', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Operational impact by dept
    dept_ops = pd.crosstab(df['department'], df['operational_impact'], normalize='index')
    dept_ops[['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']].plot(kind='bar', stacked=True, colormap='YlOrRd', ax=axes[2])
    axes[2].set_title('Operational Impact Breakdown by Department', fontweight='bold')
    axes[2].set_ylabel('Proportion')
    axes[2].legend(title='Impact', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('eda/plots/06_department_comparisons.png', dpi=300)
    plt.close()
    
    # Plot 7: Urgency Rate across 23 Asset Types
    fig, ax = plt.subplots(figsize=(12, 10))
    asset_urg_rate = df.groupby('asset_type')['urgent'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=True)
    bars = ax.barh(asset_urg_rate['asset_type'], asset_urg_rate['mean'] * 100, color='#2563eb', alpha=0.85)
    ax.axvline(x=urgent_pcts[1], color='#ef4444', linestyle='--', linewidth=2, label=f'Dataset Average ({urgent_pcts[1]:.1f}%)')
    ax.set_title('Urgency Rate (%) Across Asset Types', fontsize=14, fontweight='bold')
    ax.set_xlabel('Urgency Rate (%)')
    ax.legend(loc='lower right')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/plots/07_asset_type_urgency_rates.png', dpi=300)
    plt.close()
    
    print("Visualizations created successfully.")
    
    return {
        "df": df,
        "missing_vals": missing_vals,
        "duplicates": duplicates,
        "unique_task_ids": unique_task_ids,
        "unique_asset_ids": unique_asset_ids,
        "invalid_trains": invalid_trains,
        "outlier_summary": outlier_summary,
        "urgent_counts": urgent_counts,
        "urgent_pcts": urgent_pcts,
        "cat_target_rates": cat_target_rates,
        "num_target_comparison": num_target_comparison,
        "dept_stats": dept_stats,
        "corr_matrix": corr_matrix,
        "target_corrs": target_corrs,
        "corr_pairs": corr_pairs,
        "single_rule_severities": single_rule_severities,
        "single_rule_criticalities": single_rule_criticalities
    }

if __name__ == '__main__':
    res = run_eda()
    print("EDA execution completed.")
