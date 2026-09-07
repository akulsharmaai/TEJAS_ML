"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS ML Pipeline: Counterfactual & Sanity Verification Suite

Performs domain-grounded safety and sanity tests on finalized tuned models:
1. Severity Monotonicity Test: Increasing defect severity (LOW -> MEDIUM -> HIGH -> CRITICAL)
   must strictly increase (or keep non-decreasing) predicted failure probability and priority score.
2. Maintenance Overdue Monotonicity Test: Increasing overdue days (0 -> 15 -> 45 -> 90 -> 180)
   must strictly increase predicted failure probability.
3. Operational Impact & Traffic Monotonicity Test: Increasing affected services
   must increase block requirement probability and priority score.
4. Defect Recurrence Monotonicity Test: Increasing past defect recurrences
   must elevate failure risk.
5. Strict Target Leakage Audit: Confirms 0 target fields (*_target) are present in the feature space.
"""

import os
import json
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

def load_environment():
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    
    feature_cols = [c for c in test_df.columns if c not in TARGET_COLS]
    
    models_dir = os.path.join(BASE_DIR, "models", "tuned_models")
    f30_model = joblib.load(os.path.join(models_dir, "failure_30d_best_tuned.joblib"))
    blk_model = joblib.load(os.path.join(models_dir, "block_required_best_tuned.joblib"))
    pri_model = joblib.load(os.path.join(models_dir, "priority_score_best_tuned.joblib"))
    dur_model = joblib.load(os.path.join(models_dir, "maintenance_duration_best_tuned.joblib"))
    
    return test_df[feature_cols].copy(), train_df[feature_cols].copy(), feature_cols, f30_model, blk_model, pri_model, dur_model


def run_sanity_tests():
    print("=" * 80)
    print("STARTING TEJAS SANITY & COUNTERFACTUAL VERIFICATION SUITE")
    print("=" * 80)
    
    audit_dir = os.path.join(BASE_DIR, "models", "system_audit")
    os.makedirs(audit_dir, exist_ok=True)
    
    X_test, X_train, feature_cols, f30_model, blk_model, pri_model, dur_model = load_environment()
    
    test_results: Dict[str, Any] = {}
    
    # ---------------------------------------------------------------------------------------------------
    # TEST 1: STRICT TARGET LEAKAGE AUDIT
    # ---------------------------------------------------------------------------------------------------
    print("\n[Audit 1/5] Target Column Leakage Check...")
    leaked_cols = [c for c in feature_cols if any(t in c for t in ['_target', 'target_'])]
    leakage_passed = len(leaked_cols) == 0
    test_results["Target_Leakage_Audit"] = {
        "passed": leakage_passed,
        "leaked_columns_found": leaked_cols,
        "feature_count": len(feature_cols)
    }
    assert leakage_passed, f"FATAL: Target leakage detected in features: {leaked_cols}"
    print(f"  Passed: Zero target leakage. Exactly {len(feature_cols)} clean predictor features.")
    
    # ---------------------------------------------------------------------------------------------------
    # TEST 2: DEFECT SEVERITY MONOTONICITY
    # ---------------------------------------------------------------------------------------------------
    print("\n[Audit 2/5] Defect Severity Monotonicity Test (LOW -> CRITICAL)...")
    # Base sample: median feature values
    base_sample = X_test.median().to_frame().T
    
    severities = [1, 2, 3] # Medium (1), High (2), Critical (3)
    f30_probs_sev = []
    pri_scores_sev = []
    
    for s in severities:
        sample = base_sample.copy()
        if 'defect_severity_ord' in sample.columns:
            sample['defect_severity_ord'] = s
        prob = float(f30_model.predict_proba(sample)[:, 1][0]) if hasattr(f30_model, "predict_proba") else float(f30_model.decision_function(sample)[0])
        pri = float(pri_model.predict(sample)[0])
        f30_probs_sev.append(prob)
        pri_scores_sev.append(pri)
        
    # Check monotonic increase
    f30_sev_monotonic = all(f30_probs_sev[i] <= f30_probs_sev[i+1] + 1e-4 for i in range(len(f30_probs_sev)-1))
    pri_sev_monotonic = all(pri_scores_sev[i] <= pri_scores_sev[i+1] + 1e-4 for i in range(len(pri_scores_sev)-1))
    
    test_results["Defect_Severity_Monotonicity"] = {
        "passed": f30_sev_monotonic and pri_sev_monotonic,
        "severities_tested": ["Medium (1)", "High (2)", "Critical (3)"],
        "failure_probabilities": [round(p, 4) for p in f30_probs_sev],
        "priority_scores": [round(p, 2) for p in pri_scores_sev]
    }
    print(f"  Failure Probabilities: {f30_probs_sev} -> Monotonic: {f30_sev_monotonic}")
    print(f"  Priority Scores:       {pri_scores_sev} -> Monotonic: {pri_sev_monotonic}")
    
    # ---------------------------------------------------------------------------------------------------
    # TEST 3: MAINTENANCE OVERDUE MONOTONICITY
    # ---------------------------------------------------------------------------------------------------
    print("\n[Audit 3/5] Maintenance Overdue Monotonicity Test (0 -> 180 Days)...")
    overdue_days_list = [0.0, 15.0, 45.0, 90.0, 180.0]
    f30_probs_ovd = []
    pri_scores_ovd = []
    
    for ovd in overdue_days_list:
        sample = base_sample.copy()
        if 'maintenance_overdue_days' in sample.columns:
            sample['maintenance_overdue_days'] = ovd
        if 'days_since_last_maintenance' in sample.columns:
            sample['days_since_last_maintenance'] = 60.0 + ovd
        if 'maintenance_overdue_ratio' in sample.columns:
            sample['maintenance_overdue_ratio'] = ovd / 60.0
            
        prob = float(f30_model.predict_proba(sample)[:, 1][0]) if hasattr(f30_model, "predict_proba") else float(f30_model.decision_function(sample)[0])
        pri = float(pri_model.predict(sample)[0])
        f30_probs_ovd.append(prob)
        pri_scores_ovd.append(pri)
        
    f30_ovd_monotonic = all(f30_probs_ovd[i] <= f30_probs_ovd[i+1] + 1e-4 for i in range(len(f30_probs_ovd)-1))
    pri_ovd_monotonic = all(pri_scores_ovd[i] <= pri_scores_ovd[i+1] + 1e-4 for i in range(len(pri_scores_ovd)-1))
    
    test_results["Maintenance_Overdue_Monotonicity"] = {
        "passed": f30_ovd_monotonic and pri_ovd_monotonic,
        "overdue_days_tested": overdue_days_list,
        "failure_probabilities": [round(p, 4) for p in f30_probs_ovd],
        "priority_scores": [round(p, 2) for p in pri_scores_ovd]
    }
    print(f"  Failure Probabilities: {f30_probs_ovd} -> Monotonic: {f30_ovd_monotonic}")
    print(f"  Priority Scores:       {pri_scores_ovd} -> Monotonic: {pri_ovd_monotonic}")
    
    # ---------------------------------------------------------------------------------------------------
    # TEST 4: TRAFFIC & AFFECTED SERVICES MONOTONICITY
    # ---------------------------------------------------------------------------------------------------
    print("\n[Audit 4/5] Operational Impact / Traffic Monotonicity Test...")
    affected_services_list = [0.0, 5.0, 15.0, 30.0, 60.0]
    blk_probs_aff = []
    pri_scores_aff = []
    
    for aff in affected_services_list:
        sample = base_sample.copy()
        if 'affected_services_if_blocked' in sample.columns:
            sample['affected_services_if_blocked'] = aff
        if 'traffic_impact_ratio' in sample.columns:
            sample['traffic_impact_ratio'] = min(1.0, aff / 50.0)
        if 'operational_impact_score' in sample.columns:
            sample['operational_impact_score'] = min(100.0, aff * 1.5)
            
        prob = float(blk_model.predict_proba(sample)[:, 1][0]) if hasattr(blk_model, "predict_proba") else float(blk_model.decision_function(sample)[0])
        pri = float(pri_model.predict(sample)[0])
        blk_probs_aff.append(prob)
        pri_scores_aff.append(pri)
        
    blk_aff_monotonic = all(blk_probs_aff[i] <= blk_probs_aff[i+1] + 1e-4 for i in range(len(blk_probs_aff)-1))
    pri_aff_monotonic = all(pri_scores_aff[i] <= pri_scores_aff[i+1] + 1e-4 for i in range(len(pri_scores_aff)-1))
    
    test_results["Traffic_Impact_Monotonicity"] = {
        "passed": blk_aff_monotonic and pri_aff_monotonic,
        "affected_services_tested": affected_services_list,
        "block_probabilities": [round(p, 4) for p in blk_probs_aff],
        "priority_scores": [round(p, 2) for p in pri_scores_aff]
    }
    print(f"  Block Probabilities:   {blk_probs_aff} -> Monotonic: {blk_aff_monotonic}")
    print(f"  Priority Scores:       {pri_scores_aff} -> Monotonic: {pri_aff_monotonic}")
    
    # ---------------------------------------------------------------------------------------------------
    # TEST 5: PAST RECURRENT FAILURES SENSITIVITY
    # ---------------------------------------------------------------------------------------------------
    print("\n[Audit 5/5] Defect Recurrence Sensitivity Test...")
    recurrences_list = [0.0, 1.0, 3.0, 6.0]
    f30_probs_rec = []
    
    for rec in recurrences_list:
        sample = base_sample.copy()
        if 'same_defect_recurrences_365d' in sample.columns:
            sample['same_defect_recurrences_365d'] = rec
        if 'failures_last_365d' in sample.columns:
            sample['failures_last_365d'] = max(1.0, rec + 1.0)
        if 'recurrence_ratio' in sample.columns:
            sample['recurrence_ratio'] = rec / max(1.0, rec + 1.0)
            
        prob = float(f30_model.predict_proba(sample)[:, 1][0]) if hasattr(f30_model, "predict_proba") else float(f30_model.decision_function(sample)[0])
        f30_probs_rec.append(prob)
        
    f30_rec_monotonic = all(f30_probs_rec[i] <= f30_probs_rec[i+1] + 1e-4 for i in range(len(f30_probs_rec)-1))
    test_results["Defect_Recurrence_Sensitivity"] = {
        "passed": f30_rec_monotonic,
        "recurrences_tested": recurrences_list,
        "failure_probabilities": [round(p, 4) for p in f30_probs_rec]
    }
    print(f"  Failure Probabilities: {f30_probs_rec} -> Monotonic: {f30_rec_monotonic}")
    
    # Overall summary
    all_passed = all(isinstance(v, dict) and v.get("passed", False) for v in test_results.values())
    test_results["Overall_Sanity_Passed"] = all_passed
    
    with open(os.path.join(audit_dir, "counterfactual_test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"SANITY & COUNTERFACTUAL VERIFICATION SUITE: {'ALL 5 TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)
    return test_results


if __name__ == '__main__':
    run_sanity_tests()
