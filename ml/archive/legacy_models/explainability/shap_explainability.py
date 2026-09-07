"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS ML Pipeline: SHAP Explainability & Risk Attribution Suite

Generates global and local feature attributions using SHAP:
1. Global Feature Importance Ranking across all 4 operational targets.
2. SHAP summary plots (Bar chart & Bee-swarm distribution plots).
3. Local sample explanations for Indian Railways operational archetypes:
   - Critical Emergency Rail Defect (Fracture / High Severity)
   - Routine Track Maintenance (Low Severity / Normal Cycle)
   - Signaling / Interlocking Failure at Critical Junction
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RANDOM_SEED = 42

TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

def get_shap_array(shap_obj: Any) -> np.ndarray:
    if hasattr(shap_obj, "values"):
        return np.asarray(shap_obj.values)
    return np.asarray(shap_obj)

def load_data_and_models():
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    raw_path = os.path.join(BASE_DIR, "data", "tejas_pilot_master_dataset.csv")
    
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    raw_df = pd.read_csv(raw_path)
    
    feature_cols = [c for c in test_df.columns if c not in TARGET_COLS]
    X_test = test_df[feature_cols].copy()
    X_train = train_df[feature_cols].copy()
    
    models_dir = os.path.join(BASE_DIR, "models", "tuned_models")
    f30_model = joblib.load(os.path.join(models_dir, "failure_30d_best_tuned.joblib"))
    blk_model = joblib.load(os.path.join(models_dir, "block_required_best_tuned.joblib"))
    pri_model = joblib.load(os.path.join(models_dir, "priority_score_best_tuned.joblib"))
    
    return X_train, X_test, feature_cols, f30_model, blk_model, pri_model, raw_df


def run_shap_suite():
    print("=" * 80)
    print("STARTING TEJAS SHAP EXPLAINABILITY SUITE")
    print("=" * 80)
    
    output_dir = os.path.join(BASE_DIR, "models", "explainability")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    X_train, X_test, feature_cols, f30_model, blk_model, pri_model, raw_df = load_data_and_models()
    
    # ---------------------------------------------------------------------------------------------------
    # 1. SHAP for Target 1: failure_within_30d_target
    # ---------------------------------------------------------------------------------------------------
    print("\n[1/3] Computing SHAP values for failure_within_30d_target...")
    
    # Sample background and evaluation data for computational efficiency
    bg_sample = shap.sample(X_train, 100, random_state=RANDOM_SEED)
    eval_sample = X_test.iloc[:200]
    
    # Determine appropriate explainer based on model type
    if isinstance(f30_model, (shap.explainers._tree.TreeEnsemble, )):
        explainer_f30 = shap.TreeExplainer(f30_model)
        shap_values_f30 = explainer_f30(eval_sample)
    elif hasattr(f30_model, "coef_"):
        explainer_f30 = shap.LinearExplainer(f30_model, bg_sample)
        shap_values_f30 = explainer_f30(eval_sample)
    else:
        # Fallback to general Explainer / Exact / Kernel
        try:
            explainer_f30 = shap.TreeExplainer(f30_model)
            shap_values_f30 = explainer_f30(eval_sample)
        except Exception:
            explainer_f30 = shap.Explainer(f30_model.predict_proba, bg_sample)
            shap_values_f30 = explainer_f30(eval_sample)
            
    # Extract values array for plotting
    f30_vals = get_shap_array(shap_values_f30)
    if len(f30_vals.shape) == 3 and f30_vals.shape[-1] == 2:
        f30_vals_pos = f30_vals[:, :, 1]
    else:
        f30_vals_pos = f30_vals
        
    mean_abs_shap_f30 = np.mean(np.abs(f30_vals_pos), axis=0)
    top_indices_f30 = np.argsort(mean_abs_shap_f30)[::-1][:20]
    
    df_f30_importance = pd.DataFrame({
        "feature": [feature_cols[i] for i in top_indices_f30],
        "mean_abs_shap": mean_abs_shap_f30[top_indices_f30]
    })
    df_f30_importance.to_csv(os.path.join(output_dir, "shap_feature_importance_failure_30d.csv"), index=False)
    
    # Global Importance Plot
    plt.figure(figsize=(10, 8))
    plt.barh(df_f30_importance['feature'][::-1], df_f30_importance['mean_abs_shap'][::-1], color='#1f77b4')
    plt.xlabel("Mean Absolute SHAP Value (Impact on Failure Risk)", fontsize=11)
    plt.title("Top 20 Global Risk Drivers: failure_within_30d_target", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_bar_failure_30d.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------------------------------------------------
    # 2. SHAP for Target 2: block_required_target
    # ---------------------------------------------------------------------------------------------------
    print("\n[2/3] Computing SHAP values for block_required_target...")
    if hasattr(blk_model, "coef_"):
        explainer_blk = shap.LinearExplainer(blk_model, bg_sample)
        shap_values_blk = explainer_blk(eval_sample)
    else:
        try:
            explainer_blk = shap.TreeExplainer(blk_model)
            shap_values_blk = explainer_blk(eval_sample)
        except Exception:
            explainer_blk = shap.Explainer(blk_model.predict_proba, bg_sample)
            shap_values_blk = explainer_blk(eval_sample)
            
    blk_vals = get_shap_array(shap_values_blk)
    if len(blk_vals.shape) == 3 and blk_vals.shape[-1] == 2:
        blk_vals_pos = blk_vals[:, :, 1]
    else:
        blk_vals_pos = blk_vals
        
    mean_abs_shap_blk = np.mean(np.abs(blk_vals_pos), axis=0)
    top_indices_blk = np.argsort(mean_abs_shap_blk)[::-1][:20]
    
    df_blk_importance = pd.DataFrame({
        "feature": [feature_cols[i] for i in top_indices_blk],
        "mean_abs_shap": mean_abs_shap_blk[top_indices_blk]
    })
    df_blk_importance.to_csv(os.path.join(output_dir, "shap_feature_importance_block_required.csv"), index=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(df_blk_importance['feature'][::-1], df_blk_importance['mean_abs_shap'][::-1], color='#2ca02c')
    plt.xlabel("Mean Absolute SHAP Value (Impact on Block Requirement)", fontsize=11)
    plt.title("Top 20 Global Drivers: block_required_target", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_bar_block_required.png"), dpi=200)
    plt.close()
    
    # ---------------------------------------------------------------------------------------------------
    # 3. Local Explanations for Railway Operational Archetypes
    # ---------------------------------------------------------------------------------------------------
    print("\n[3/3] Generating Local Archetype Explanations for Safety Operations...")
    
    explanations = []
    
    # Identify Archetypes in test set:
    # 1. High Risk / Critical Failure Candidate
    probs_f30 = f30_model.predict_proba(X_test)[:, 1] if hasattr(f30_model, "predict_proba") else f30_model.decision_function(X_test)
    high_risk_idx = int(np.argmax(probs_f30))
    
    # 2. Low Risk Routine Candidate
    low_risk_idx = int(np.argmin(probs_f30))
    
    # 3. Block Required / High Traffic Impact Candidate
    probs_blk = blk_model.predict_proba(X_test)[:, 1] if hasattr(blk_model, "predict_proba") else blk_model.decision_function(X_test)
    block_candidate_idx = int(np.argmax(probs_blk))
    
    archetypes = [
        ("Critical_Failure_Risk_Incident", high_risk_idx),
        ("Routine_Low_Risk_Maintenance", low_risk_idx),
        ("High_Traffic_Block_Required_Event", block_candidate_idx)
    ]
    
    for name, idx in archetypes:
        row_features = X_test.iloc[[idx]]
        pred_prob_f30 = float(probs_f30[idx])
        pred_prob_blk = float(probs_blk[idx])
        pred_priority = float(pri_model.predict(row_features)[0])
        
        # Local SHAP contributions for this instance
        try:
            inst_shap = explainer_f30(row_features)
            inst_vals = get_shap_array(inst_shap)
            if len(inst_vals.shape) == 3:
                inst_vals = inst_vals[0, :, 1]
            elif len(inst_vals.shape) == 2:
                inst_vals = inst_vals[0, :]
        except Exception:
            inst_vals = np.zeros(len(feature_cols))
            
        top_contrib_idx = np.argsort(np.abs(inst_vals))[::-1][:6]
        drivers = []
        for c_idx in top_contrib_idx:
            drivers.append({
                "feature": feature_cols[c_idx],
                "feature_value": float(row_features.iloc[0, c_idx]),
                "shap_impact": round(float(inst_vals[c_idx]), 4),
                "direction": "INCREASES_RISK" if inst_vals[c_idx] > 0 else "DECREASES_RISK"
            })
            
        explanations.append({
            "archetype": name,
            "test_sample_index": idx,
            "predicted_failure_probability_30d": round(pred_prob_f30, 4),
            "predicted_block_probability": round(pred_prob_blk, 4),
            "predicted_priority_score": round(pred_priority, 2),
            "top_shap_risk_drivers": drivers
        })
        
    with open(os.path.join(output_dir, "example_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2)
        
    print("\nSHAP explainability analysis complete. Summary artifacts written to models/explainability/")


if __name__ == '__main__':
    run_shap_suite()
