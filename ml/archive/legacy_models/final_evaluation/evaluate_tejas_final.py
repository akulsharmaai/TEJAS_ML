"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS ML Pipeline: Final Model Evaluation on Untouched Test Set

Evaluates tuned champions across all 4 operational targets on the 1,500-sample test set:
1. failure_within_30d_target (Evaluated at both 0.5 and Calibrated Safety Threshold)
2. block_required_target (Evaluated at both 0.5 and Calibrated Safety Threshold)
3. priority_score_target (Continuous regression metrics: R2, MAE, RMSE, Pearson r)
4. maintenance_duration_hours_target (Continuous regression metrics)

Generates:
- Confusion matrices & metric tables.
- Performance plots in models/final_evaluation/
- Comprehensive final evaluation report in markdown.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RANDOM_SEED = 42

TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

def load_datasets():
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
    test_path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    y_train = {t: train_df[t].copy() for t in TARGET_COLS if t in train_df.columns and t != 'priority_class_target'}
    y_val = {t: val_df[t].copy() for t in TARGET_COLS if t in val_df.columns and t != 'priority_class_target'}
    y_test = {t: test_df[t].copy() for t in TARGET_COLS if t in test_df.columns and t != 'priority_class_target'}
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def run_final_evaluation():
    print("=" * 80)
    print("STARTING TEJAS FINAL MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)
    
    eval_dir = os.path.join(BASE_DIR, "models", "final_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_datasets()
    
    # Load calibrated thresholds
    th_path = os.path.join(BASE_DIR, "models", "tuning", "calibrated_thresholds.json")
    with open(th_path, "r") as f:
        thresholds_cfg = json.load(f)
        
    models_dir = os.path.join(BASE_DIR, "models", "tuned_models")
    
    results = []
    
    # ---------------------------------------------------------------------------------------------------
    # 1. EVALUATE TARGET 1: failure_within_30d_target
    # ---------------------------------------------------------------------------------------------------
    f30_model_file = os.path.join(models_dir, "failure_30d_best_tuned.joblib")
    f30_model = joblib.load(f30_model_file)
    f30_th = thresholds_cfg["failure_within_30d_target"]["calibrated_threshold"]
    f30_name = thresholds_cfg["failure_within_30d_target"]["best_model"]
    
    y_tr = y_train['failure_within_30d_target'].to_numpy()
    y_va = y_val['failure_within_30d_target'].to_numpy()
    y_te = y_test['failure_within_30d_target'].to_numpy()
    
    tr_probs = f30_model.predict_proba(X_train)[:, 1] if hasattr(f30_model, "predict_proba") else f30_model.decision_function(X_train)
    va_probs = f30_model.predict_proba(X_val)[:, 1] if hasattr(f30_model, "predict_proba") else f30_model.decision_function(X_val)
    te_probs = f30_model.predict_proba(X_test)[:, 1] if hasattr(f30_model, "predict_proba") else f30_model.decision_function(X_test)
    
    # Default 0.50 Threshold
    te_preds_def = (te_probs >= 0.50).astype(int)
    va_preds_def = (va_probs >= 0.50).astype(int)
    tr_preds_def = (tr_probs >= 0.50).astype(int)
    
    # Calibrated Safety Threshold
    te_preds_cal = (te_probs >= f30_th).astype(int)
    va_preds_cal = (va_probs >= f30_th).astype(int)
    
    cm_def = confusion_matrix(y_te, te_preds_def)
    cm_cal = confusion_matrix(y_te, te_preds_cal)
    
    results.append({
        "target": "failure_within_30d_target",
        "model": f30_name,
        "mode": "Default Threshold (0.50)",
        "threshold": 0.50,
        "val_recall": recall_score(y_va, va_preds_def),
        "val_precision": precision_score(y_va, va_preds_def),
        "val_f1": f1_score(y_va, va_preds_def),
        "val_roc_auc": roc_auc_score(y_va, va_probs),
        "test_accuracy": accuracy_score(y_te, te_preds_def),
        "test_recall": recall_score(y_te, te_preds_def),
        "test_precision": precision_score(y_te, te_preds_def),
        "test_f1": f1_score(y_te, te_preds_def),
        "test_f2": fbeta_score(y_te, te_preds_def, beta=2),
        "test_roc_auc": roc_auc_score(y_te, te_probs),
        "test_pr_auc": average_precision_score(y_te, te_probs),
        "test_TN": int(cm_def[0, 0]),
        "test_FP": int(cm_def[0, 1]),
        "test_FN": int(cm_def[1, 0]),
        "test_TP": int(cm_def[1, 1])
    })
    
    results.append({
        "target": "failure_within_30d_target",
        "model": f30_name,
        "mode": f"Calibrated Safety Threshold ({f30_th:.3f})",
        "threshold": f30_th,
        "val_recall": recall_score(y_va, va_preds_cal),
        "val_precision": precision_score(y_va, va_preds_cal),
        "val_f1": f1_score(y_va, va_preds_cal),
        "val_roc_auc": roc_auc_score(y_va, va_probs),
        "test_accuracy": accuracy_score(y_te, te_preds_cal),
        "test_recall": recall_score(y_te, te_preds_cal),
        "test_precision": precision_score(y_te, te_preds_cal),
        "test_f1": f1_score(y_te, te_preds_cal),
        "test_f2": fbeta_score(y_te, te_preds_cal, beta=2),
        "test_roc_auc": roc_auc_score(y_te, te_probs),
        "test_pr_auc": average_precision_score(y_te, te_probs),
        "test_TN": int(cm_cal[0, 0]),
        "test_FP": int(cm_cal[0, 1]),
        "test_FN": int(cm_cal[1, 0]),
        "test_TP": int(cm_cal[1, 1])
    })
    
    # ---------------------------------------------------------------------------------------------------
    # 2. EVALUATE TARGET 2: block_required_target
    # ---------------------------------------------------------------------------------------------------
    blk_model_file = os.path.join(models_dir, "block_required_best_tuned.joblib")
    blk_model = joblib.load(blk_model_file)
    blk_th = thresholds_cfg["block_required_target"]["calibrated_threshold"]
    blk_name = thresholds_cfg["block_required_target"]["best_model"]
    
    y_tr_b = y_train['block_required_target'].to_numpy()
    y_va_b = y_val['block_required_target'].to_numpy()
    y_te_b = y_test['block_required_target'].to_numpy()
    
    va_probs_b = blk_model.predict_proba(X_val)[:, 1] if hasattr(blk_model, "predict_proba") else blk_model.decision_function(X_val)
    te_probs_b = blk_model.predict_proba(X_test)[:, 1] if hasattr(blk_model, "predict_proba") else blk_model.decision_function(X_test)
    
    te_preds_b_def = (te_probs_b >= 0.50).astype(int)
    va_preds_b_def = (va_probs_b >= 0.50).astype(int)
    te_preds_b_cal = (te_probs_b >= blk_th).astype(int)
    va_preds_b_cal = (va_probs_b >= blk_th).astype(int)
    
    cm_b_def = confusion_matrix(y_te_b, te_preds_b_def)
    cm_b_cal = confusion_matrix(y_te_b, te_preds_b_cal)
    
    results.append({
        "target": "block_required_target",
        "model": blk_name,
        "mode": "Default Threshold (0.50)",
        "threshold": 0.50,
        "val_recall": recall_score(y_va_b, va_preds_b_def),
        "val_precision": precision_score(y_va_b, va_preds_b_def),
        "val_f1": f1_score(y_va_b, va_preds_b_def),
        "val_roc_auc": roc_auc_score(y_va_b, va_probs_b),
        "test_accuracy": accuracy_score(y_te_b, te_preds_b_def),
        "test_recall": recall_score(y_te_b, te_preds_b_def),
        "test_precision": precision_score(y_te_b, te_preds_b_def),
        "test_f1": f1_score(y_te_b, te_preds_b_def),
        "test_f2": fbeta_score(y_te_b, te_preds_b_def, beta=2),
        "test_roc_auc": roc_auc_score(y_te_b, te_probs_b),
        "test_pr_auc": average_precision_score(y_te_b, te_probs_b),
        "test_TN": int(cm_b_def[0, 0]),
        "test_FP": int(cm_b_def[0, 1]),
        "test_FN": int(cm_b_def[1, 0]),
        "test_TP": int(cm_b_def[1, 1])
    })
    
    results.append({
        "target": "block_required_target",
        "model": blk_name,
        "mode": f"Calibrated Safety Threshold ({blk_th:.3f})",
        "threshold": blk_th,
        "val_recall": recall_score(y_va_b, va_preds_b_cal),
        "val_precision": precision_score(y_va_b, va_preds_b_cal),
        "val_f1": f1_score(y_va_b, va_preds_b_cal),
        "val_roc_auc": roc_auc_score(y_va_b, va_probs_b),
        "test_accuracy": accuracy_score(y_te_b, te_preds_b_cal),
        "test_recall": recall_score(y_te_b, te_preds_b_cal),
        "test_precision": precision_score(y_te_b, te_preds_b_cal),
        "test_f1": f1_score(y_te_b, te_preds_b_cal),
        "test_f2": fbeta_score(y_te_b, te_preds_b_cal, beta=2),
        "test_roc_auc": roc_auc_score(y_te_b, te_probs_b),
        "test_pr_auc": average_precision_score(y_te_b, te_probs_b),
        "test_TN": int(cm_b_cal[0, 0]),
        "test_FP": int(cm_b_cal[0, 1]),
        "test_FN": int(cm_b_cal[1, 0]),
        "test_TP": int(cm_b_cal[1, 1])
    })
    
    # ---------------------------------------------------------------------------------------------------
    # 3. EVALUATE TARGET 3 & 4: Regression Targets
    # ---------------------------------------------------------------------------------------------------
    pri_model = joblib.load(os.path.join(models_dir, "priority_score_best_tuned.joblib"))
    dur_model = joblib.load(os.path.join(models_dir, "maintenance_duration_best_tuned.joblib"))
    
    reg_targets = [
        ("priority_score_target", pri_model, y_train['priority_score_target'].to_numpy(), y_val['priority_score_target'].to_numpy(), y_test['priority_score_target'].to_numpy()),
        ("maintenance_duration_hours_target", dur_model, y_train['maintenance_duration_hours_target'].to_numpy(), y_val['maintenance_duration_hours_target'].to_numpy(), y_test['maintenance_duration_hours_target'].to_numpy())
    ]
    
    reg_results = []
    for t_name, model, y_tr_r, y_va_r, y_te_r in reg_targets:
        va_preds = model.predict(X_val)
        te_preds = model.predict(X_test)
        
        r2_va = r2_score(y_va_r, va_preds)
        r2_te = r2_score(y_te_r, te_preds)
        mae_va = mean_absolute_error(y_va_r, va_preds)
        mae_te = mean_absolute_error(y_te_r, te_preds)
        rmse_va = np.sqrt(mean_squared_error(y_va_r, va_preds))
        rmse_te = np.sqrt(mean_squared_error(y_te_r, te_preds))
        r_val, _ = pearsonr(y_te_r, te_preds)
        
        reg_results.append({
            "target": t_name,
            "model": type(model).__name__,
            "val_r2": r2_va,
            "test_r2": r2_te,
            "val_mae": mae_va,
            "test_mae": mae_te,
            "val_rmse": rmse_va,
            "test_rmse": rmse_te,
            "test_pearson_r": r_val
        })
        
    # Save CSVs
    df_cls_results = pd.DataFrame(results)
    df_reg_results = pd.DataFrame(reg_results)
    
    df_cls_results.to_csv(os.path.join(eval_dir, "final_classification_test_results.csv"), index=False)
    df_reg_results.to_csv(os.path.join(eval_dir, "final_regression_test_results.csv"), index=False)
    
    # ---------------------------------------------------------------------------------------------------
    # GENERATE PLOTS: Confusion Matrices & Regression Diagnostic
    # ---------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Failure 30d Confusion Matrix (Calibrated)
    sns.heatmap(cm_cal, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Failure (0)', 'Failure (1)'],
                yticklabels=['Actual No Fail', 'Actual Fail'])
    axes[0, 0].set_title(f"Target 1: failure_within_30d (Threshold = {f30_th:.3f})\nRecall: {recall_score(y_te, te_preds_cal):.1%}, Precision: {precision_score(y_te, te_preds_cal):.1%}", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Subplot 2: Block Required Confusion Matrix (Calibrated)
    sns.heatmap(cm_b_cal, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
                xticklabels=['No Block (0)', 'Block Required (1)'],
                yticklabels=['Actual No Block', 'Actual Block'])
    axes[0, 1].set_title(f"Target 2: block_required (Threshold = {blk_th:.3f})\nRecall: {recall_score(y_te_b, te_preds_b_cal):.1%}, Precision: {precision_score(y_te_b, te_preds_b_cal):.1%}", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Subplot 3: Priority Score Predicted vs Actual
    pri_preds_te = pri_model.predict(X_test)
    y_test_pri_np = y_test['priority_score_target'].to_numpy()
    axes[1, 0].scatter(y_test_pri_np, pri_preds_te, alpha=0.35, color='#2b5c8f', edgecolors='none', s=20)
    axes[1, 0].plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_title(f"Target 3: Priority Score (R² = {r2_score(y_test_pri_np, pri_preds_te):.3f}, MAE = {mean_absolute_error(y_test_pri_np, pri_preds_te):.2f})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Actual Priority Score (0-100)')
    axes[1, 0].set_ylabel('Predicted Priority Score')
    axes[1, 0].legend()
    
    # Subplot 4: Maintenance Duration Predicted vs Actual
    dur_preds_te = dur_model.predict(X_test)
    y_test_dur_np = y_test['maintenance_duration_hours_target'].to_numpy()
    axes[1, 1].scatter(y_test_dur_np, dur_preds_te, alpha=0.35, color='#d95f02', edgecolors='none', s=20)
    axes[1, 1].plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
    axes[1, 1].set_title(f"Target 4: Duration Hours (R² = {r2_score(y_test_dur_np, dur_preds_te):.3f}, MAE = {mean_absolute_error(y_test_dur_np, dur_preds_te):.2f}h)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Actual Duration Hours')
    axes[1, 1].set_ylabel('Predicted Duration Hours')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "tejas_final_evaluation_dashboard.png"), dpi=200)
    plt.close()
    
    print("\nFinal Test Evaluation Complete. Dashboard saved to tejas_final_evaluation_dashboard.png")
    return df_cls_results, df_reg_results


if __name__ == '__main__':
    run_final_evaluation()
