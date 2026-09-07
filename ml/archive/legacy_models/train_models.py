"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS Multi-Target Baseline ML Model Training & Evaluation Pipeline

Trains and evaluates baseline models across 4 distinct operational targets:
1. failure_within_30d_target (Binary Classification - Risk of catastrophic in-service failure)
2. block_required_target (Binary Classification - Requirement for traffic block window)
3. priority_score_target (Regression - Granular 0-100 task urgency index)
4. maintenance_duration_hours_target (Regression - Estimated physical maintenance repair hours)

Strict Domain Constraints:
- Zero leakage: NO *_target column is used as a predictor.
- Prioritizes Recall for safety-critical failure & block prediction.
- Saves all models, metrics, confusion matrices, and feature importance rankings.
"""

import os
from typing import Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

RANDOM_SEED = 42
TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

def to_md_table(df_in: pd.DataFrame) -> str:
    headers = [c for c in df_in.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join([":---"] * len(headers)) + " |")
    for _, row in df_in.iterrows():
        vals = []
        for x in row:
            if isinstance(x, (float, np.floating)):
                vals.append(f"{x:.4f}")
            else:
                vals.append(str(x))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def load_data(data_dir: str = "data/processed"):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    targets = {
        'train': {t: train_df[t].copy() for t in TARGET_COLS if t in train_df.columns and t != 'priority_class_target'},
        'val': {t: val_df[t].copy() for t in TARGET_COLS if t in val_df.columns and t != 'priority_class_target'},
        'test': {t: test_df[t].copy() for t in TARGET_COLS if t in test_df.columns and t != 'priority_class_target'}
    }
    
    print(f"Loaded processed features: {X_train.shape[1]} predictors")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    return X_train, X_val, X_test, targets, feature_cols


def evaluate_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name: str, target_name: str):
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict on splits
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_train_prob = model.decision_function(X_train)
        y_val_prob = model.decision_function(X_val)
        y_test_prob = model.decision_function(X_test)
        
    cm_val = confusion_matrix(y_val, y_val_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    metrics = {
        "target": target_name,
        "task_type": "Classification",
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "val_roc_auc": roc_auc_score(y_val, y_val_prob),
        "test_roc_auc": roc_auc_score(y_test, y_test_prob),
        "val_pr_auc": average_precision_score(y_val, y_val_prob),
        "test_pr_auc": average_precision_score(y_test, y_test_prob),
        "val_confusion_matrix": str(cm_val.tolist()),
        "test_confusion_matrix": str(cm_test.tolist()),
        "overfit_gap_acc": accuracy_score(y_train, y_train_pred) - accuracy_score(y_val, y_val_pred)
    }
    
    return model, metrics, cm_test


def evaluate_regressor(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name: str, target_name: str):
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict on splits
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    
    metrics = {
        "target": target_name,
        "task_type": "Regression",
        "model": model_name,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "overfit_gap_r2": train_r2 - val_r2
    }
    
    return model, metrics


def run_training_pipeline():
    print("=" * 80)
    print("STARTING TEJAS BASELINE MULTI-TARGET MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    X_train, X_val, X_test, targets, feature_cols = load_data()
    
    models_dir = "models/baseline_models"
    os.makedirs(models_dir, exist_ok=True)
    
    all_metrics = []
    trained_models = {}
    feature_importances_dict = {}
    
    # =========================================================
    # TARGET 1: failure_within_30d_target (Classification)
    # =========================================================
    t1 = 'failure_within_30d_target'
    print(f"\n---> Training Models for Target 1: {t1}")
    y_train, y_val, y_test = targets['train'][t1], targets['val'][t1], targets['test'][t1]
    
    clf_t1_models = {
        "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting (Balanced)": HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, max_depth=6, random_state=RANDOM_SEED)
    }
    
    best_t1_model: Tuple[str, Any] = ("", None)
    best_t1_f1 = -1.0
    
    for name, clf in clf_t1_models.items():
        fitted_model, m, cm = evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test, name, t1)
        all_metrics.append(m)
        print(f"  [{name}] Val F1: {m['val_f1']:.4f} | Recall: {m['val_recall']:.4f} | ROC-AUC: {m['val_roc_auc']:.4f} | Test Acc: {m['test_accuracy']:.4f}")
        
        if m['val_f1'] > best_t1_f1:
            best_t1_f1 = m['val_f1']
            best_t1_model = (name, fitted_model)
            
        # Extract feature importances if available
        if hasattr(fitted_model, "feature_importances_"):
            feature_importances_dict[f"{t1}_{name}"] = pd.Series(fitted_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        elif hasattr(fitted_model, "coef_"):
            feature_importances_dict[f"{t1}_{name}"] = pd.Series(np.abs(fitted_model.coef_[0]), index=feature_cols).sort_values(ascending=False)
            
    joblib.dump(best_t1_model[1], os.path.join(models_dir, f"{t1}_best_model.joblib"))
    trained_models[t1] = best_t1_model
    
    # =========================================================
    # TARGET 2: block_required_target (Classification)
    # =========================================================
    t2 = 'block_required_target'
    print(f"\n---> Training Models for Target 2: {t2}")
    y_train, y_val, y_test = targets['train'][t2], targets['val'][t2], targets['test'][t2]
    
    clf_t2_models = {
        "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting (Balanced)": HistGradientBoostingClassifier(class_weight='balanced', max_iter=150, max_depth=6, random_state=RANDOM_SEED)
    }
    
    best_t2_model: Tuple[str, Any] = ("", None)
    best_t2_f1 = -1.0
    
    for name, clf in clf_t2_models.items():
        fitted_model, m, cm = evaluate_classifier(clf, X_train, y_train, X_val, y_val, X_test, y_test, name, t2)
        all_metrics.append(m)
        print(f"  [{name}] Val F1: {m['val_f1']:.4f} | Recall: {m['val_recall']:.4f} | ROC-AUC: {m['val_roc_auc']:.4f} | Test Acc: {m['test_accuracy']:.4f}")
        
        if m['val_f1'] > best_t2_f1:
            best_t2_f1 = m['val_f1']
            best_t2_model = (name, fitted_model)
            
        if hasattr(fitted_model, "feature_importances_"):
            feature_importances_dict[f"{t2}_{name}"] = pd.Series(fitted_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        elif hasattr(fitted_model, "coef_"):
            feature_importances_dict[f"{t2}_{name}"] = pd.Series(np.abs(fitted_model.coef_[0]), index=feature_cols).sort_values(ascending=False)
            
    joblib.dump(best_t2_model[1], os.path.join(models_dir, f"{t2}_best_model.joblib"))
    trained_models[t2] = best_t2_model
    
    # =========================================================
    # TARGET 3: priority_score_target (Regression)
    # =========================================================
    t3 = 'priority_score_target'
    print(f"\n---> Training Models for Target 3: {t3}")
    y_train, y_val, y_test = targets['train'][t3], targets['val'][t3], targets['test'][t3]
    
    reg_t3_models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=RANDOM_SEED)
    }
    
    best_t3_model: Tuple[str, Any] = ("", None)
    best_t3_r2 = -999.0
    
    for name, reg in reg_t3_models.items():
        fitted_model, m = evaluate_regressor(reg, X_train, y_train, X_val, y_val, X_test, y_test, name, t3)
        all_metrics.append(m)
        print(f"  [{name}] Val R2: {m['val_r2']:.4f} | Val MAE: {m['val_mae']:.4f} | Val RMSE: {m['val_rmse']:.4f}")
        
        if m['val_r2'] > best_t3_r2:
            best_t3_r2 = m['val_r2']
            best_t3_model = (name, fitted_model)
            
        if hasattr(fitted_model, "feature_importances_"):
            feature_importances_dict[f"{t3}_{name}"] = pd.Series(fitted_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        elif hasattr(fitted_model, "coef_"):
            feature_importances_dict[f"{t3}_{name}"] = pd.Series(np.abs(fitted_model.coef_), index=feature_cols).sort_values(ascending=False)
            
    joblib.dump(best_t3_model[1], os.path.join(models_dir, f"{t3}_best_model.joblib"))
    trained_models[t3] = best_t3_model
    
    # =========================================================
    # TARGET 4: maintenance_duration_hours_target (Regression)
    # =========================================================
    t4 = 'maintenance_duration_hours_target'
    print(f"\n---> Training Models for Target 4: {t4}")
    y_train, y_val, y_test = targets['train'][t4], targets['val'][t4], targets['test'][t4]
    
    reg_t4_models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=RANDOM_SEED)
    }
    
    best_t4_model: Tuple[str, Any] = ("", None)
    best_t4_r2 = -999.0
    
    for name, reg in reg_t4_models.items():
        fitted_model, m = evaluate_regressor(reg, X_train, y_train, X_val, y_val, X_test, y_test, name, t4)
        all_metrics.append(m)
        print(f"  [{name}] Val R2: {m['val_r2']:.4f} | Val MAE: {m['val_mae']:.4f} | Val RMSE: {m['val_rmse']:.4f}")
        
        if m['val_r2'] > best_t4_r2:
            best_t4_r2 = m['val_r2']
            best_t4_model = (name, fitted_model)
            
        if hasattr(fitted_model, "feature_importances_"):
            feature_importances_dict[f"{t4}_{name}"] = pd.Series(fitted_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        elif hasattr(fitted_model, "coef_"):
            feature_importances_dict[f"{t4}_{name}"] = pd.Series(np.abs(fitted_model.coef_), index=feature_cols).sort_values(ascending=False)
            
    joblib.dump(best_t4_model[1], os.path.join(models_dir, f"{t4}_best_model.joblib"))
    trained_models[t4] = best_t4_model
    
    # ---------------------------------------------------------
    # Save Metrics & Comparison CSV
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    comparison_csv_path = "models/model_comparison.csv"
    metrics_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved model evaluation metrics to: {comparison_csv_path}")
    
    # Save top feature importances
    fi_rows = []
    for k, s in feature_importances_dict.items():
        for rank, (feat, val) in enumerate(s.head(15).items(), 1):
            fi_rows.append({"model_target": k, "rank": rank, "feature": feat, "importance": round(float(val), 5)})
    fi_df = pd.DataFrame(fi_rows)
    fi_csv_path = "models/feature_importances.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Saved top feature importances to: {fi_csv_path}")
    
    # ---------------------------------------------------------
    # Generate Comprehensive Markdown Report
    # ---------------------------------------------------------
    report_path = "models/model_training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TEJAS Machine Learning Pipeline: Baseline Model Training Report\n\n")
        f.write(f"**Dataset**: `data/tejas_pilot_master_dataset.csv` (10,000 records × 37 columns)\n")
        f.write(f"**Splitting Strategy**: GroupKFold by `asset_id` (Train: 7,000 | Val: 1,500 | Test: 1,500) — Zero Asset Leakage\n")
        f.write(f"**Predictor Feature Dimensions**: {len(feature_cols)} transformed features (Strict zero target leakage)\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Executive Summary: Best Model Per Target\n\n")
        f.write("| Target | Best Architecture | Primary Metric | Val Performance | Test Performance |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        # T1 best
        t1_best_m = next(m for m in all_metrics if m['target'] == t1 and m['model'] == best_t1_model[0])
        f.write(f"| **{t1}** (Classification) | {best_t1_model[0]} | F1 / Recall / ROC-AUC | F1: {t1_best_m['val_f1']:.4f}, Recall: {t1_best_m['val_recall']:.4f}, AUC: {t1_best_m['val_roc_auc']:.4f} | F1: {t1_best_m['test_f1']:.4f}, Recall: {t1_best_m['test_recall']:.4f}, AUC: {t1_best_m['test_roc_auc']:.4f} |\n")
        
        # T2 best
        t2_best_m = next(m for m in all_metrics if m['target'] == t2 and m['model'] == best_t2_model[0])
        f.write(f"| **{t2}** (Classification) | {best_t2_model[0]} | F1 / Recall / ROC-AUC | F1: {t2_best_m['val_f1']:.4f}, Recall: {t2_best_m['val_recall']:.4f}, AUC: {t2_best_m['val_roc_auc']:.4f} | F1: {t2_best_m['test_f1']:.4f}, Recall: {t2_best_m['test_recall']:.4f}, AUC: {t2_best_m['test_roc_auc']:.4f} |\n")
        
        # T3 best
        t3_best_m = next(m for m in all_metrics if m['target'] == t3 and m['model'] == best_t3_model[0])
        f.write(f"| **{t3}** (Regression) | {best_t3_model[0]} | $R^2$ / MAE / RMSE | $R^2$: {t3_best_m['val_r2']:.4f}, MAE: {t3_best_m['val_mae']:.2f}, RMSE: {t3_best_m['val_rmse']:.2f} | $R^2$: {t3_best_m['test_r2']:.4f}, MAE: {t3_best_m['test_mae']:.2f}, RMSE: {t3_best_m['test_rmse']:.2f} |\n")
        
        # T4 best
        t4_best_m = next(m for m in all_metrics if m['target'] == t4 and m['model'] == best_t4_model[0])
        f.write(f"| **{t4}** (Regression) | {best_t4_model[0]} | $R^2$ / MAE / RMSE | $R^2$: {t4_best_m['val_r2']:.4f}, MAE: {t4_best_m['val_mae']:.2f}h, RMSE: {t4_best_m['val_rmse']:.2f}h | $R^2$: {t4_best_m['test_r2']:.4f}, MAE: {t4_best_m['test_mae']:.2f}h, RMSE: {t4_best_m['test_rmse']:.2f}h |\n")
        
        f.write("\n---\n\n")
        f.write("## 2. Complete Model Comparison Table\n\n")
        f.write(to_md_table(metrics_df))
        f.write("\n\n---\n\n")
        
        f.write("## 3. Strongest Features by Target\n\n")
        for k, s in feature_importances_dict.items():
            f.write(f"### Top Features: `{k}`\n\n")
            top_df = s.head(10).reset_index()
            top_df.columns = ["Feature", "Importance / Weight"]
            f.write(to_md_table(top_df))
            f.write("\n\n")
            
    print(f"Generated comprehensive report at: {report_path}")
    print("=" * 80)
    
    return all_metrics, trained_models, feature_importances_dict

if __name__ == '__main__':
    run_training_pipeline()
