"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS ML Pipeline: Multi-Target Group-Aware Hyperparameter Tuning & Threshold Calibration

Features:
1. GroupKFold Cross-Validation on the 7,000 training records using asset_id to guarantee zero leakage.
2. Multi-target tuning:
   - failure_within_30d_target (Binary Classification)
   - block_required_target (Binary Classification)
   - priority_score_target (Regression)
   - maintenance_duration_hours_target (Regression)
3. Safety-first threshold calibration on the validation set for classification targets (optimizing F2-score & Recall >= 85%).
4. Saves all tuned models, parameter records, and threshold lookup tables.
"""

import os
import sys
import json
import itertools
import warnings
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

def load_data_and_groups():
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
    raw_path = os.path.join(BASE_DIR, "data", "tejas_pilot_master_dataset.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    raw_df = pd.read_csv(raw_path)
    
    # Re-extract asset_id groups for the train split exactly matching preprocess.py
    gkf = GroupKFold(n_splits=20)
    splits = list(gkf.split(raw_df, raw_df['failure_within_30d_target'], raw_df['asset_id']))
    train_idx = np.concatenate([splits[i][1] for i in range(14)])
    val_idx = np.concatenate([splits[i][1] for i in range(14, 17)])
    
    train_groups = raw_df['asset_id'].iloc[train_idx].to_numpy()
    val_groups = raw_df['asset_id'].iloc[val_idx].to_numpy()
    
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS]
    
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    
    y_train_dict = {t: train_df[t] for t in TARGET_COLS if t in train_df.columns and t != 'priority_class_target'}
    y_val_dict = {t: val_df[t] for t in TARGET_COLS if t in val_df.columns and t != 'priority_class_target'}
    
    return X_train, y_train_dict, train_groups, X_val, y_val_dict, val_groups, feature_cols


def run_group_cv_classification(model_cls, param_grid: Dict[str, Any], X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_splits=4) -> Tuple[Dict[str, Any], Dict[str, float]]:
    gkf = GroupKFold(n_splits=n_splits)
    grid = list(ParameterGrid(param_grid))
    
    best_score = -1.0
    best_params: Dict[str, Any] = {}
    best_cv_metrics: Dict[str, float] = {}
    
    for params in grid:
        fold_recalls: List[float] = []
        fold_precisions: List[float] = []
        fold_f1s: List[float] = []
        fold_f2s: List[float] = []
        fold_aucs: List[float] = []
        
        for train_fold_idx, val_fold_idx in gkf.split(X, y, groups):
            X_tr, y_tr = X.iloc[train_fold_idx], y.iloc[train_fold_idx]
            X_va, y_va = X.iloc[val_fold_idx], y.iloc[val_fold_idx]
            
            if model_cls == RandomForestClassifier:
                clf = model_cls(random_state=RANDOM_SEED, n_jobs=-1, **params)
            else:
                clf = model_cls(random_state=RANDOM_SEED, **params)
                    
            clf.fit(X_tr, y_tr)
            
            if hasattr(clf, "predict_proba"):
                y_va_prob = np.asarray(clf.predict_proba(X_va))[:, 1]
                y_va_pred = (y_va_prob >= 0.5).astype(int)
            elif hasattr(clf, "decision_function"):
                y_va_prob = getattr(clf, "decision_function")(X_va)
                y_va_pred = (y_va_prob >= 0.0).astype(int)
            else:
                y_va_pred = clf.predict(X_va)
                y_va_prob = clf.predict(X_va)
                
            fold_recalls.append(float(recall_score(y_va, y_va_pred, zero_division=0)))
            fold_precisions.append(float(precision_score(y_va, y_va_pred, zero_division=0)))
            fold_f1s.append(float(f1_score(y_va, y_va_pred, zero_division=0)))
            fold_f2s.append(float(fbeta_score(y_va, y_va_pred, beta=2, zero_division=0)))
            fold_aucs.append(float(roc_auc_score(y_va, y_va_prob)))
            
        mean_f2 = float(np.mean(fold_f2s))
        mean_auc = float(np.mean(fold_aucs))
        composite_score = 0.6 * mean_f2 + 0.4 * mean_auc
        
        if composite_score > best_score:
            best_score = composite_score
            best_params = params
            best_cv_metrics = {
                "cv_recall": float(np.mean(fold_recalls)),
                "cv_precision": float(np.mean(fold_precisions)),
                "cv_f1": float(np.mean(fold_f1s)),
                "cv_f2": mean_f2,
                "cv_roc_auc": mean_auc,
                "composite_score": composite_score
            }
            
    return best_params, best_cv_metrics


def run_group_cv_regression(model_cls, param_grid: Dict[str, Any], X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_splits=4) -> Tuple[Dict[str, Any], Dict[str, float]]:
    gkf = GroupKFold(n_splits=n_splits)
    grid = list(ParameterGrid(param_grid))
    
    best_score = -999.0
    best_params: Dict[str, Any] = {}
    best_cv_metrics: Dict[str, float] = {}
    
    for params in grid:
        fold_r2s: List[float] = []
        fold_maes: List[float] = []
        fold_rmses: List[float] = []
        
        for train_fold_idx, val_fold_idx in gkf.split(X, y, groups):
            X_tr, y_tr = X.iloc[train_fold_idx], y.iloc[train_fold_idx]
            X_va, y_va = X.iloc[val_fold_idx], y.iloc[val_fold_idx]
            
            if model_cls == RandomForestRegressor:
                reg = model_cls(random_state=RANDOM_SEED, n_jobs=-1, **params)
            elif model_cls == Ridge:
                reg = model_cls(**params)
            else:
                reg = model_cls(random_state=RANDOM_SEED, **params)
                    
            reg.fit(X_tr, y_tr)
            y_va_pred = reg.predict(X_va)
            
            fold_r2s.append(r2_score(y_va, y_va_pred))
            fold_maes.append(mean_absolute_error(y_va, y_va_pred))
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_va, y_va_pred))))
            
        mean_r2 = float(np.mean(fold_r2s))
        mean_mae = float(np.mean(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = params
            best_cv_metrics = {
                "cv_r2": mean_r2,
                "cv_mae": mean_mae,
                "cv_rmse": mean_rmse
            }
            
    return best_params, best_cv_metrics


def calibrate_threshold(y_true: np.ndarray, y_probs: np.ndarray, min_target_recall=0.85) -> Tuple[float, float, pd.DataFrame]:
    """
    Finds optimal safety classification threshold prioritizing high recall (safety-critical constraint)
    and maximal F2 score.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    records = []
    
    best_f2 = -1.0
    best_f2_th = 0.50
    safety_selected_th = 0.50
    best_safety_score = -1.0
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        rec = float(recall_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        f2 = float(fbeta_score(y_true, preds, beta=2, zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        cm = confusion_matrix(y_true, preds)
        fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
        fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
        
        records.append({
            "threshold": round(float(th), 3),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "accuracy": round(acc, 4),
            "false_negatives": fn,
            "false_positives": fp
        })
        
        if f2 > best_f2:
            best_f2 = f2
            best_f2_th = float(th)
            
        if rec >= min_target_recall:
            score = 0.5 * rec + 0.5 * prec
            if score > best_safety_score:
                best_safety_score = score
                safety_selected_th = float(th)
                
    if best_safety_score < 0:
        safety_selected_th = best_f2_th
        
    df_cal = pd.DataFrame(records)
    return safety_selected_th, best_f2_th, df_cal


def run_tuning():
    print("=" * 80, flush=True)
    print("STARTING MULTI-TARGET GROUP-AWARE HYPERPARAMETER TUNING & CALIBRATION", flush=True)
    print("=" * 80, flush=True)
    
    os.makedirs(os.path.join(BASE_DIR, "models", "tuned_models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models", "tuning"), exist_ok=True)
    
    X_train, y_train_dict, train_groups, X_val, y_val_dict, val_groups, feature_cols = load_data_and_groups()
    
    X_tr_np = X_train.to_numpy()
    X_va_np = X_val.to_numpy()
    
    tuning_summary = []
    calibrated_thresholds: Dict[str, Any] = {}
    
    # ---------------------------------------------------------------------------------------------------
    # TARGET 1: failure_within_30d_target
    # ---------------------------------------------------------------------------------------------------
    print("\n" + "#" * 70, flush=True)
    print("TUNING TARGET 1: failure_within_30d_target (Safety Failure Prediction)", flush=True)
    print("#" * 70, flush=True)
    
    y_tr_f30 = y_train_dict['failure_within_30d_target']
    y_va_f30 = y_val_dict['failure_within_30d_target']
    y_tr_f30_np = y_tr_f30.to_numpy()
    y_va_f30_np = y_va_f30.to_numpy()
    
    lr_grid = {
        'C': [0.1, 0.5, 1.0],
        'class_weight': ['balanced', {0: 1.0, 1: 2.5}],
        'max_iter': [1000]
    }
    lr_params, lr_cv = run_group_cv_classification(LogisticRegression, lr_grid, X_train, y_tr_f30, train_groups)
    print(f"  LogisticRegression Best: {lr_params} | F2: {lr_cv['cv_f2']:.4f}, AUC: {lr_cv['cv_roc_auc']:.4f}", flush=True)
    
    rf_grid = {
        'n_estimators': [150],
        'max_depth': [12, 16],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }
    rf_params, rf_cv = run_group_cv_classification(RandomForestClassifier, rf_grid, X_train, y_tr_f30, train_groups)
    print(f"  RandomForest Best: {rf_params} | F2: {rf_cv['cv_f2']:.4f}, AUC: {rf_cv['cv_roc_auc']:.4f}", flush=True)
    
    hgb_grid = {
        'learning_rate': [0.05, 0.08],
        'max_iter': [100],
        'max_depth': [6, 8],
        'min_samples_leaf': [20],
        'l2_regularization': [1.0],
        'class_weight': ['balanced']
    }
    hgb_params, hgb_cv = run_group_cv_classification(HistGradientBoostingClassifier, hgb_grid, X_train, y_tr_f30, train_groups)
    print(f"  HistGradientBoosting Best: {hgb_params} | F2: {hgb_cv['cv_f2']:.4f}, AUC: {hgb_cv['cv_roc_auc']:.4f}", flush=True)
    
    f30_models = {
        "Logistic Regression (Tuned)": LogisticRegression(random_state=RANDOM_SEED, **lr_params),
        "Random Forest (Tuned)": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **rf_params),
        "HistGradientBoosting (Tuned)": HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hgb_params)
    }
    
    best_f30_model_name = ""
    best_f30_model_obj = None
    best_f30_val_auc = -1.0
    best_f30_probs_val = np.zeros(len(y_va_f30_np))
    
    for name, model in f30_models.items():
        model.fit(X_train, y_tr_f30)
        probs_val = np.asarray(model.predict_proba(X_val))[:, 1] if hasattr(model, "predict_proba") else getattr(model, "decision_function")(X_val)
        val_auc = float(roc_auc_score(y_va_f30_np, probs_val))
        val_pr_auc = float(average_precision_score(y_va_f30_np, probs_val))
        
        preds_def = (probs_val >= 0.5).astype(int)
        tuning_summary.append({
            "target": "failure_within_30d_target",
            "model": name,
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr_auc,
            "val_recall_0.5": float(recall_score(y_va_f30_np, preds_def)),
            "val_precision_0.5": float(precision_score(y_va_f30_np, preds_def)),
            "val_f1_0.5": float(f1_score(y_va_f30_np, preds_def))
        })
        
        if val_auc > best_f30_val_auc:
            best_f30_val_auc = val_auc
            best_f30_model_name = name
            best_f30_model_obj = model
            best_f30_probs_val = probs_val
            
    th_safety, th_f2, df_cal_f30 = calibrate_threshold(y_va_f30_np, best_f30_probs_val, min_target_recall=0.85)
    calibrated_thresholds["failure_within_30d_target"] = {
        "best_model": best_f30_model_name,
        "calibrated_threshold": th_safety,
        "f2_threshold": th_f2,
        "min_target_recall": 0.85
    }
    
    preds_cal = (np.asarray(best_f30_probs_val) >= th_safety).astype(int)
    print(f"\nTarget 1 Champion: {best_f30_model_name}", flush=True)
    print(f"  Validation ROC-AUC: {best_f30_val_auc:.4f}", flush=True)
    print(f"  Calibrated Safety Threshold: {th_safety:.3f}", flush=True)
    print(f"  Calibrated Validation -> Recall: {recall_score(y_va_f30_np, preds_cal):.4f}, Precision: {precision_score(y_va_f30_np, preds_cal):.4f}, F1: {f1_score(y_va_f30_np, preds_cal):.4f}", flush=True)
    
    joblib.dump(best_f30_model_obj, os.path.join(BASE_DIR, "models", "tuned_models", "failure_30d_best_tuned.joblib"))
    df_cal_f30.to_csv(os.path.join(BASE_DIR, "models", "tuning", "failure_30d_threshold_calibration.csv"), index=False)
    
    # ---------------------------------------------------------------------------------------------------
    # TARGET 2: block_required_target
    # ---------------------------------------------------------------------------------------------------
    print("\n" + "#" * 70, flush=True)
    print("TUNING TARGET 2: block_required_target (Block Window Requirement)", flush=True)
    print("#" * 70, flush=True)
    
    y_tr_blk = y_train_dict['block_required_target']
    y_va_blk = y_val_dict['block_required_target']
    y_tr_blk_np = y_tr_blk.to_numpy()
    y_va_blk_np = y_va_blk.to_numpy()
    
    lr_blk_grid = {'C': [0.01, 0.1, 1.0, 5.0, 10.0]}
    lr_blk_params, lr_blk_cv = run_group_cv_classification(LogisticRegression, lr_blk_grid, X_train, y_tr_blk, train_groups)
    
    rf_blk_grid = {
        'n_estimators': [150, 250],
        'max_depth': [8, 12],
        'min_samples_leaf': [2, 5],
        'class_weight': ['balanced']
    }
    rf_blk_params, rf_blk_cv = run_group_cv_classification(RandomForestClassifier, rf_blk_grid, X_train, y_tr_blk, train_groups)
    
    hgb_blk_grid = {
        'learning_rate': [0.03, 0.06],
        'max_iter': [120, 200],
        'max_depth': [6, 8],
        'min_samples_leaf': [15, 25],
        'l2_regularization': [1.0, 3.0]
    }
    hgb_blk_params, hgb_blk_cv = run_group_cv_classification(HistGradientBoostingClassifier, hgb_blk_grid, X_train, y_tr_blk, train_groups)
    
    blk_models = {
        "Logistic Regression (Tuned)": LogisticRegression(random_state=RANDOM_SEED, max_iter=2000, **lr_blk_params),
        "Random Forest Classifier (Tuned)": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **rf_blk_params),
        "HistGradientBoosting Classifier (Tuned)": HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hgb_blk_params)
    }
    
    best_blk_model_name = ""
    best_blk_model_obj = None
    best_blk_val_auc = -1.0
    best_blk_probs_val = np.zeros(len(y_va_blk_np))
    
    for name, model in blk_models.items():
        model.fit(X_train, y_tr_blk)
        probs_val = np.asarray(model.predict_proba(X_val))[:, 1] if hasattr(model, "predict_proba") else getattr(model, "decision_function")(X_val)
        val_auc = float(roc_auc_score(y_va_blk_np, probs_val))
        val_pr_auc = float(average_precision_score(y_va_blk_np, probs_val))
        
        preds_def = (probs_val >= 0.5).astype(int)
        tuning_summary.append({
            "target": "block_required_target",
            "model": name,
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr_auc,
            "val_recall_0.5": float(recall_score(y_va_blk_np, preds_def)),
            "val_precision_0.5": float(precision_score(y_va_blk_np, preds_def)),
            "val_f1_0.5": float(f1_score(y_va_blk_np, preds_def))
        })
        
        if val_auc > best_blk_val_auc:
            best_blk_val_auc = val_auc
            best_blk_model_name = name
            best_blk_model_obj = model
            best_blk_probs_val = probs_val
            
    th_blk_safety, th_blk_f2, df_cal_blk = calibrate_threshold(y_va_blk_np, best_blk_probs_val, min_target_recall=0.85)
    calibrated_thresholds["block_required_target"] = {
        "best_model": best_blk_model_name,
        "calibrated_threshold": th_blk_safety,
        "f2_threshold": th_blk_f2,
        "min_target_recall": 0.85
    }
    
    preds_blk_cal = (np.asarray(best_blk_probs_val) >= th_blk_safety).astype(int)
    print(f"\nTarget 2 Champion: {best_blk_model_name}", flush=True)
    print(f"  Validation ROC-AUC: {best_blk_val_auc:.4f}", flush=True)
    print(f"  Calibrated Safety Threshold: {th_blk_safety:.3f}", flush=True)
    print(f"  Calibrated Validation -> Recall: {recall_score(y_va_blk_np, preds_blk_cal):.4f}, Precision: {precision_score(y_va_blk_np, preds_blk_cal):.4f}, F1: {f1_score(y_va_blk_np, preds_blk_cal):.4f}", flush=True)
    
    joblib.dump(best_blk_model_obj, os.path.join(BASE_DIR, "models", "tuned_models", "block_required_best_tuned.joblib"))
    df_cal_blk.to_csv(os.path.join(BASE_DIR, "models", "tuning", "block_required_threshold_calibration.csv"), index=False)
    
    # ---------------------------------------------------------------------------------------------------
    # TARGET 3: priority_score_target
    # ---------------------------------------------------------------------------------------------------
    print("\n" + "#" * 70, flush=True)
    print("TUNING TARGET 3: priority_score_target (Continuous Priority Score)", flush=True)
    print("#" * 70, flush=True)
    
    y_tr_pri = y_train_dict['priority_score_target']
    y_va_pri = y_val_dict['priority_score_target']
    y_tr_pri_np = y_tr_pri.to_numpy()
    y_va_pri_np = y_va_pri.to_numpy()
    
    ridge_grid = {'alpha': [0.5, 1.0, 10.0, 50.0]}
    ridge_pri_params, ridge_pri_cv = run_group_cv_regression(Ridge, ridge_grid, X_train, y_tr_pri, train_groups)
    
    rf_reg_grid = {
        'n_estimators': [150],
        'max_depth': [12, 16],
        'min_samples_leaf': [2, 5]
    }
    rf_pri_params, rf_pri_cv = run_group_cv_regression(RandomForestRegressor, rf_reg_grid, X_train, y_tr_pri, train_groups)
    
    hgb_reg_grid = {
        'learning_rate': [0.05, 0.08],
        'max_iter': [100],
        'max_depth': [6, 8],
        'min_samples_leaf': [20],
        'l2_regularization': [1.0]
    }
    hgb_pri_params, hgb_pri_cv = run_group_cv_regression(HistGradientBoostingRegressor, hgb_reg_grid, X_train, y_tr_pri, train_groups)
    
    pri_models = {
        "Ridge Regression (Tuned)": Ridge(**ridge_pri_params),
        "Random Forest Regressor (Tuned)": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_pri_params),
        "HistGradientBoosting Regressor (Tuned)": HistGradientBoostingRegressor(random_state=RANDOM_SEED, **hgb_pri_params)
    }
    
    best_pri_model_name = ""
    best_pri_model_obj = None
    best_pri_val_r2 = -999.0
    
    for name, model in pri_models.items():
        model.fit(X_train, y_tr_pri)
        preds_val = model.predict(X_val)
        val_r2 = r2_score(y_va_pri_np, preds_val)
        val_mae = mean_absolute_error(y_va_pri_np, preds_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_va_pri_np, preds_val)))
        
        tuning_summary.append({
            "target": "priority_score_target",
            "model": name,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })
        
        if val_r2 > best_pri_val_r2:
            best_pri_val_r2 = val_r2
            best_pri_model_name = name
            best_pri_model_obj = model
            
    print(f"\nTarget 3 Champion: {best_pri_model_name}", flush=True)
    print(f"  Validation R2: {best_pri_val_r2:.4f}", flush=True)
    joblib.dump(best_pri_model_obj, os.path.join(BASE_DIR, "models", "tuned_models", "priority_score_best_tuned.joblib"))
    
    # ---------------------------------------------------------------------------------------------------
    # TARGET 4: maintenance_duration_hours_target
    # ---------------------------------------------------------------------------------------------------
    print("\n" + "#" * 70, flush=True)
    print("TUNING TARGET 4: maintenance_duration_hours_target (Maintenance Duration Hours)", flush=True)
    print("#" * 70, flush=True)
    
    y_tr_dur = y_train_dict['maintenance_duration_hours_target']
    y_va_dur = y_val_dict['maintenance_duration_hours_target']
    y_tr_dur_np = y_tr_dur.to_numpy()
    y_va_dur_np = y_va_dur.to_numpy()
    
    ridge_dur_params, ridge_dur_cv = run_group_cv_regression(Ridge, ridge_grid, X_train, y_tr_dur, train_groups)
    rf_dur_params, rf_dur_cv = run_group_cv_regression(RandomForestRegressor, rf_reg_grid, X_train, y_tr_dur, train_groups)
    hgb_dur_params, hgb_dur_cv = run_group_cv_regression(HistGradientBoostingRegressor, hgb_reg_grid, X_train, y_tr_dur, train_groups)
    
    dur_models = {
        "Ridge Regression (Tuned)": Ridge(**ridge_dur_params),
        "Random Forest Regressor (Tuned)": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_dur_params),
        "HistGradientBoosting Regressor (Tuned)": HistGradientBoostingRegressor(random_state=RANDOM_SEED, **hgb_dur_params)
    }
    
    best_dur_model_name = ""
    best_dur_model_obj = None
    best_dur_val_r2 = -999.0
    
    for name, model in dur_models.items():
        model.fit(X_train, y_tr_dur)
        preds_val = model.predict(X_val)
        val_r2 = r2_score(y_va_dur_np, preds_val)
        val_mae = mean_absolute_error(y_va_dur_np, preds_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_va_dur_np, preds_val)))
        
        tuning_summary.append({
            "target": "maintenance_duration_hours_target",
            "model": name,
            "val_r2": val_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })
        
        if val_r2 > best_dur_val_r2:
            best_dur_val_r2 = val_r2
            best_dur_model_name = name
            best_dur_model_obj = model
            
    print(f"\nTarget 4 Champion: {best_dur_model_name}", flush=True)
    print(f"  Validation R2: {best_dur_val_r2:.4f}", flush=True)
    joblib.dump(best_dur_model_obj, os.path.join(BASE_DIR, "models", "tuned_models", "maintenance_duration_best_tuned.joblib"))
    
    # Save calibrated thresholds config and tuning summary
    with open(os.path.join(BASE_DIR, "models", "tuning", "calibrated_thresholds.json"), "w") as f:
        json.dump(calibrated_thresholds, f, indent=2)
        
    df_tuning_summary = pd.DataFrame(tuning_summary)
    df_tuning_summary.to_csv(os.path.join(BASE_DIR, "models", "tuning", "tejas_tuning_summary.csv"), index=False)
    
    print("\n" + "=" * 80, flush=True)
    print("ALL TARGETS TUNED & CALIBRATED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)
    return calibrated_thresholds


if __name__ == '__main__':
    run_tuning()
