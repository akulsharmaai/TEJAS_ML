"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Step 11: SHAP-Based Explainability Pipeline

Generates global and local feature explanations using TreeSHAP on the
finalized tuned XGBoost model (models/tuned_models/xgboost_tuned.joblib).
Outputs human-readable risk drivers tailored for Indian Railways operations.
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'figure.autolayout': True})

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "tuned_models", "xgboost_tuned.joblib"),
    os.path.join(BASE_DIR, "models", "baseline_models", "xgboost_weighted.joblib"),
    os.path.join(BASE_DIR, "models", "baseline_models", "xgboost_default.joblib")
]

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "processed", "railway_maintenance_tasks.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "explainability")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

ORDINAL_REVERSE = {
    "defect_severity": {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"},
    "asset_criticality": {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"},
    "operational_impact": {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
}

def get_priority_label(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "CRITICAL"
    elif risk_score >= 0.50:
        return "HIGH"
    elif risk_score >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"

def run_shap_analysis():
    print("="*75)
    print("STEP 11: SHAP EXPLAINABILITY ANALYSIS")
    print("="*75)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load Model & Data
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            model_path = candidate
            break
            
    if model_path is None:
        raise FileNotFoundError(f"No trained XGBoost model found in candidate locations: {MODEL_CANDIDATES}")
        
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print("Loading preprocessed validation dataset...")
    val_df = pd.read_csv(VAL_PATH)
    X_val = val_df.drop(columns=['urgent'])
    y_val = val_df['urgent']
    feature_names = list(X_val.columns)
    
    # Load raw dataset for human-readable raw values
    raw_df = pd.read_csv(RAW_PATH)
    
    # Map validation indices to raw_df rows
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    splits = list(sgkf.split(raw_df.drop(columns=['urgent']), raw_df['urgent'], raw_df['asset_id']))
    val_raw_idx = np.concatenate([splits[i][1] for i in range(14, 17)])
    raw_val_subset = raw_df.iloc[val_raw_idx].reset_index(drop=True)
    
    print(f"Validation records for SHAP: {len(X_val):,}")
    
    # 2. Compute TreeSHAP Values
    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print("Computing SHAP values across validation dataset...")
    shap_values = explainer(X_val)
    
    # Extract values array
    shap_matrix = shap_values.values  # shape: (5997, 137)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]
    print(f"Base expected value (log-odds): {base_value:.4f}")
    
    # 3. Global Feature Importance Calculation
    print("\nComputing global mean absolute SHAP importances...")
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    feat_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    feat_importance_df["rank"] = feat_importance_df.index + 1
    
    importance_csv_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    feat_importance_df.to_csv(importance_csv_path, index=False)
    print(f"Saved feature importance table to: {importance_csv_path}")
    
    print("\nTop 15 Most Influential Features Globally:")
    print(feat_importance_df.head(15).to_string(index=False))
    
    # 4. Generate Global Visualizations
    print("\nGenerating global SHAP summary visualizations...")
    
    # Plot A: Top 20 Global Feature Importance (Bar Chart)
    top20_df = feat_importance_df.head(20).iloc[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top20_df["feature"], top20_df["mean_abs_shap"], color="#2563eb", alpha=0.88)
    ax.set_title("Top 20 Most Influential Features (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Urgency Prediction)", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    bar_plot_path = os.path.join(PLOTS_DIR, "01_shap_global_bar_top20.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_plot_path}")
    
    # Plot B: Top 20 Summary Beeswarm Plot
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_matrix, X_val, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Beeswarm Summary Plot (Top 20 Features)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    beeswarm_plot_path = os.path.join(PLOTS_DIR, "02_shap_summary_beeswarm_top20.png")
    plt.savefig(beeswarm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {beeswarm_plot_path}")
    
    # 5. Local Explanations for Archetypal Cases (High, Medium, Low Risk)
    print("\nGenerating local explanations for representative maintenance cases...")
    
    val_probas_arr = np.asarray(model.predict_proba(X_val))
    val_probas = val_probas_arr[:, 1]
    
    # Identify case indices
    # High risk: proba >= 0.85
    # Medium risk: 0.40 <= proba <= 0.60
    # Low risk: proba <= 0.15
    high_candidates = np.where(val_probas >= 0.88)[0]
    med_candidates = np.where((val_probas >= 0.45) & (val_probas <= 0.55))[0]
    low_candidates = np.where(val_probas <= 0.10)[0]
    
    selected_cases = [
        ("High-Risk Maintenance Task", high_candidates[0], "03_shap_local_waterfall_high_risk.png"),
        ("Medium-Risk Maintenance Task", med_candidates[0], "04_shap_local_waterfall_medium_risk.png"),
        ("Low-Risk Routine Task", low_candidates[0], "05_shap_local_waterfall_low_risk.png")
    ]
    
    example_explanations = []
    
    for label, idx, plot_name in selected_cases:
        p_val = float(val_probas[idx])
        priority = get_priority_label(p_val)
        raw_row = raw_val_subset.iloc[idx].to_dict()
        task_id: str = str(raw_row.get("task_id", f"TASK-{idx:05d}"))
        asset_id: str = str(raw_row.get("asset_id", "UNKNOWN"))
        dept: str = str(raw_row.get("department", "UNKNOWN"))
        asset_type: str = str(raw_row.get("asset_type", "UNKNOWN"))
        
        # Individual SHAP contributions
        row_shap = shap_matrix[idx]
        
        # Sort contributions by absolute magnitude
        contrib_order = np.argsort(np.abs(row_shap))[::-1]
        
        factors = []
        for f_idx in contrib_order[:8]:
            f_name: str = str(feature_names[f_idx])
            f_val_proc = X_val.iloc[idx, f_idx]
            f_shap = float(row_shap[f_idx])
            
            # Format human readable descriptor
            sign = "+" if f_shap > 0 else "-"
            
            # Retrieve original raw value if available
            raw_display = None
            if f_name in raw_row:
                raw_display = f"{raw_row[f_name]}"
            elif f_name == "freight_traffic_ratio":
                raw_display = f"{raw_row.get('goods_trains_per_day', 0)/max(raw_row.get('trains_per_day', 1), 1)*100:.1f}%"
            elif f_name == "failure_rate_indicator":
                raw_display = f"{raw_row.get('failures_last_12_months', 0)/max(raw_row.get('asset_age_years', 1), 0.5):.2f}/yr"
            elif f_name == "maintenance_overdue_ratio":
                raw_display = f"{raw_row.get('days_overdue', 0)} days overdue"
            elif f_name.startswith("department_") or f_name.startswith("asset_type_") or f_name.startswith("defect_type_") or f_name.startswith("maintenance_frequency_"):
                raw_display = "Active (1)" if f_val_proc == 1.0 else "Inactive (0)"
            else:
                raw_display = f"{f_val_proc:.2f}"
                
            factors.append({
                "factor": f_name,
                "raw_value": raw_display,
                "shap_impact": round(f_shap, 4),
                "direction": "INCREASES_RISK (+)" if f_shap > 0 else "DECREASES_RISK (-)"
            })
            
        case_info = {
            "case_type": label,
            "task_id": task_id,
            "asset_id": asset_id,
            "department": dept,
            "asset_type": asset_type,
            "risk_score": round(p_val, 4),
            "priority": priority,
            "raw_attributes": {
                "defect_type": raw_row.get("defect_type"),
                "defect_severity": raw_row.get("defect_severity"),
                "asset_criticality": raw_row.get("asset_criticality"),
                "operational_impact": raw_row.get("operational_impact"),
                "days_overdue": raw_row.get("days_overdue"),
                "days_since_defect": raw_row.get("days_since_defect"),
                "trains_per_day": raw_row.get("trains_per_day"),
                "goods_trains_per_day": raw_row.get("goods_trains_per_day"),
                "failures_last_12_months": raw_row.get("failures_last_12_months"),
                "previous_failures": raw_row.get("previous_failures"),
                "maintenance_duration_hours": raw_row.get("maintenance_duration_hours")
            },
            "top_contributing_factors": factors
        }
        example_explanations.append(case_info)
        
        # Generate Waterfall Plot for this case
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        plt.title(f"Local Explanation: {task_id} ({label})\nRisk Score: {p_val:.2f} [{priority}]", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        case_plot_path = os.path.join(PLOTS_DIR, plot_name)
        plt.savefig(case_plot_path, dpi=300)
        plt.close()
        print(f"Saved local plot: {case_plot_path}")
        
    # Save JSON explanations
    explanations_json_path = os.path.join(OUTPUT_DIR, "example_explanations.json")
    with open(explanations_json_path, "w", encoding="utf-8") as f:
        json.dump(example_explanations, f, indent=2)
    print(f"Saved example explanations JSON to: {explanations_json_path}")
    
    # ---------------------------------------------------------
    # 6. Controlled Perturbation Sensitivity Tests
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("CONTROLLED PERTURBATION SENSITIVITY TESTING")
    print("="*70)
    
    # Choose a median baseline example from validation set
    baseline_idx = med_candidates[0] if len(med_candidates) > 0 else 0
    base_sample = X_val.iloc[[baseline_idx]].copy()
    
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    level_vals = [0, 1, 2, 3]
    
    sensitivity_results = {}
    
    for var in ["asset_criticality", "defect_severity", "operational_impact"]:
        print(f"\n---> Perturbation Test: Varying '{var}' (holding other features constant)...")
        var_tests = []
        for name, val in zip(levels, level_vals):
            temp_df = base_sample.copy()
            temp_df[var] = val
            p = float(model.predict_proba(temp_df)[0, 1])
            priority = get_priority_label(p)
            var_tests.append({
                "level": name,
                "ordinal_value": val,
                "predicted_probability": round(p, 4),
                "risk_percentage": round(p * 100, 2),
                "priority": priority
            })
            print(f"  {var} = {name:<8} ({val}) -> Risk Score: {p:.4f} ({p*100:.2f}%) [{priority}]")
        sensitivity_results[var] = var_tests
        
    sensitivity_json_path = os.path.join(OUTPUT_DIR, "sensitivity_analysis.json")
    with open(sensitivity_json_path, "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)
    print(f"\nSaved sensitivity analysis results to: {sensitivity_json_path}")
    
    return feat_importance_df, example_explanations, sensitivity_results

if __name__ == '__main__':
    feat_importance_df, example_explanations, sensitivity_results = run_shap_analysis()
    print("\nSHAP analysis pipeline execution completed successfully.")
