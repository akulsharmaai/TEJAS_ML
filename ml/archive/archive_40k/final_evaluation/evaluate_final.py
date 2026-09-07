"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Step 10: Final Unbiased Test Set Evaluation

Evaluates the tuned XGBoost classifier on the held-out test dataset
(data/processed/test.csv, 5,997 records) across multiple probability thresholds.
Generates confusion matrices, ROC/PR evaluation curves, and comparison metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'figure.autolayout': True})

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_models", "xgboost_tuned.joblib")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "final_evaluation")

def run_final_evaluation():
    print("="*75)
    print("STEP 10: FINAL TEST SET EVALUATION")
    print("="*75)
    
    # 1. Load Model and Untouched Test Set
    print(f"Loading final tuned model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    print(f"Loading test set from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=['urgent'])
    y_test = test_df['urgent']
    
    print(f"Loading validation set for generalization check from: {VAL_DATA_PATH}")
    val_df = pd.read_csv(VAL_DATA_PATH)
    X_val = val_df.drop(columns=['urgent'])
    y_val = val_df['urgent']
    
    print(f"Test Set Dimensions: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    print(f"Test Set Ground Truth: {sum(y_test==0):,} Routine (0), {sum(y_test==1):,} Urgent (1) ({y_test.mean()*100:.2f}% Urgent)")
    
    # 2. Predict Probabilities
    test_proba_arr = np.asarray(model.predict_proba(X_test))
    y_test_proba = test_proba_arr[:, 1]
    
    val_proba_arr = np.asarray(model.predict_proba(X_val))
    y_val_proba = val_proba_arr[:, 1]
    
    # 3. Comprehensive Metric Calculation Across Thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    test_results = []
    
    for th in thresholds:
        y_pred = (y_test_proba >= th).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        roc_auc = roc_auc_score(y_test, y_test_proba)
        pr_auc = average_precision_score(y_test, y_test_proba)
        
        test_results.append({
            "threshold": th,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": round(spec, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })
        
    results_df = pd.DataFrame(test_results)
    csv_output_path = os.path.join(OUTPUT_DIR, "final_test_results.csv")
    results_df.to_csv(csv_output_path, index=False)
    print(f"\nSaved threshold test results to: {csv_output_path}")
    print("\n" + results_df.to_string(index=False))
    
    # 4. Generate Visualizations (Confusion Matrix & Evaluation Curves)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # A. Confusion Matrix at Recommended Threshold 0.45
    y_test_pred_rec = (y_test_proba >= 0.45).astype(int)
    cm_rec = confusion_matrix(y_test, y_test_pred_rec)
    
    sns.heatmap(cm_rec, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=axes[0],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title('Test Confusion Matrix (Threshold = 0.45)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label (0: Routine, 1: Urgent)', fontsize=11)
    axes[0].set_ylabel('Actual Label (0: Routine, 1: Urgent)', fontsize=11)
    axes[0].set_xticklabels(['Routine (0)', 'Urgent (1)'])
    axes[0].set_yticklabels(['Routine (0)', 'Urgent (1)'])
    
    # Annotate TP, TN, FP, FN inside cells
    tn_r, fp_r, fn_r, tp_r = cm_rec.ravel()
    axes[0].text(0.5, 0.25, f"(TN: {tn_r:,})", ha='center', va='center', color='gray', fontsize=10)
    axes[0].text(1.5, 0.25, f"(FP: {fp_r:,})", ha='center', va='center', color='gray', fontsize=10)
    axes[0].text(0.5, 1.25, f"(FN: {fn_r:,})", ha='center', va='center', color='gray', fontsize=10)
    axes[0].text(1.5, 1.25, f"(TP: {tp_r:,})", ha='center', va='center', color='gray', fontsize=10)
    
    # B. ROC Curves (Validation vs Test)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    val_roc = roc_auc_score(y_val, y_val_proba)
    test_roc = roc_auc_score(y_test, y_test_proba)
    
    axes[1].plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {val_roc:.4f})', color='#3b82f6', lw=2, linestyle='--')
    axes[1].plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {test_roc:.4f})', color='#dc2626', lw=2.5)
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':')
    axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].legend(loc='lower right')
    
    # C. Precision-Recall Curves (Validation vs Test)
    prec_val, rec_val, _ = precision_recall_curve(y_val, y_val_proba)
    prec_test, rec_test, _ = precision_recall_curve(y_test, y_test_proba)
    val_pr = average_precision_score(y_val, y_val_proba)
    test_pr = average_precision_score(y_test, y_test_proba)
    
    axes[2].plot(rec_val, prec_val, label=f'Validation PR (AUC = {val_pr:.4f})', color='#3b82f6', lw=2, linestyle='--')
    axes[2].plot(rec_test, prec_test, label=f'Test PR (AUC = {test_pr:.4f})', color='#dc2626', lw=2.5)
    axes[2].set_title('Precision-Recall (PR) Curve', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(loc='lower left')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plots to: {plot_path}")
    
    return results_df, test_roc, test_pr, cm_rec

if __name__ == '__main__':
    results_df, test_roc, test_pr, cm_rec = run_final_evaluation()
    print("\nFinal evaluation completed successfully.")
