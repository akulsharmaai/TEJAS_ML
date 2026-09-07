"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Step 9: Hyperparameter Tuning & Probability Threshold Calibration

Tunes XGBoost, Random Forest, and Logistic Regression models using
StratifiedGroupKFold Cross-Validation on the training set to prevent asset-level leakage.
Evaluates tuned models on the validation set, compares with baseline performance,
and performs systematic threshold calibration on the best model candidate.
"""

import os
from typing import Any, Dict, List, TypedDict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class TuningExperiment(TypedDict):
    name: str
    estimator: Any
    params: Dict[str, Any]
    n_iter: int

def load_data():
    print("Loading train and validation datasets...")
    train_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
    raw_path = os.path.join(BASE_DIR, "data", "processed", "railway_maintenance_tasks.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Load original raw data to extract asset_id for StratifiedGroupKFold in train set
    raw_df = pd.read_csv(raw_path)
    # Identify training records by index matching or asset grouping
    # In preprocess.py, 20 folds were used with seed 42.
    sgkf_full = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
    splits = list(sgkf_full.split(raw_df.drop(columns=['urgent']), raw_df['urgent'], raw_df['asset_id']))
    train_idx = np.concatenate([splits[i][1] for i in range(14)])
    train_assets = np.asarray(raw_df['asset_id'].iloc[train_idx].to_numpy())
    
    X_train = train_df.drop(columns=['urgent'])
    y_train = train_df['urgent']
    
    X_val = val_df.drop(columns=['urgent'])
    y_val = val_df['urgent']
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"Train asset groups: {len(train_assets)} records across {len(np.unique(train_assets))} unique physical assets.")
    
    return X_train, y_train, X_val, y_val, train_assets

def tune_models():
    X_train, y_train, X_val, y_val, train_assets = load_data()
    
    os.makedirs("models/tuning", exist_ok=True)
    os.makedirs("models/tuned_models", exist_ok=True)
    
    # Stratified Group K-Fold for inner CV during hyperparameter tuning
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    
    # ---------------------------------------------------------
    # 1. Parameter Search Spaces
    # ---------------------------------------------------------
    
    # A. XGBoost Weighted Search Space
    xgb_param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.03, 0.05, 0.08, 0.1, 0.15],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        'scale_pos_weight': [1.5, 1.7, 1.814, 2.0, 2.2]
    }
    
    # B. Random Forest Balanced Search Space
    rf_param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [12, 16, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3, 0.5]
    }
    
    # C. Logistic Regression Balanced Search Space
    lr_param_dist = {
        'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }
    
    tuning_experiments: List[TuningExperiment] = [
        {
            "name": "XGBoost (Tuned)",
            "estimator": XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            "params": xgb_param_dist,
            "n_iter": 30
        },
        {
            "name": "Random Forest (Tuned)",
            "estimator": RandomForestClassifier(
                class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
            ),
            "params": rf_param_dist,
            "n_iter": 15
        },
        {
            "name": "Logistic Regression (Tuned)",
            "estimator": LogisticRegression(
                class_weight='balanced', solver='lbfgs', max_iter=1000,
                random_state=RANDOM_SEED
            ),
            "params": lr_param_dist,
            "n_iter": 9
        }
    ]
    
    tuned_results = []
    best_fitted_models = {}
    
    print("\n" + "="*70)
    print("STARTING HYPERPARAMETER SEARCH (StratifiedGroupKFold on Train)")
    print("="*70)
    
    for exp in tuning_experiments:
        name: str = exp["name"]
        n_iter_val: int = exp["n_iter"]
        print(f"\n---> Searching best parameters for: {name} (Iterations: {n_iter_val}) ...")
        
        search = RandomizedSearchCV(
            estimator=exp["estimator"],
            param_distributions=exp["params"],
            n_iter=n_iter_val,
            scoring='f1',
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train, groups=train_assets)
        
        best_params = search.best_params_
        best_cv_f1 = search.best_score_
        best_model = search.best_estimator_
        best_fitted_models[name] = best_model
        
        print(f"Best CV F1-Score: {best_cv_f1*100:.2f}%")
        print(f"Optimal Parameters: {best_params}")
        
        # Evaluate on validation set
        val_proba_arr = np.asarray(best_model.predict_proba(X_val))
        y_val_proba = val_proba_arr[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, pos_label=1)
        rec = recall_score(y_val, y_val_pred, pos_label=1)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        roc_auc = roc_auc_score(y_val, y_val_proba)
        pr_auc = average_precision_score(y_val, y_val_proba)
        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Save model
        tuned_models_dir = os.path.join(BASE_DIR, "models", "tuned_models")
        os.makedirs(tuned_models_dir, exist_ok=True)
        clean_name: str = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_path = os.path.join(tuned_models_dir, f"{clean_name}.joblib")
        joblib.dump(best_model, model_path)
        
        tuned_results.append({
            "model_name": name,
            "best_cv_f1": round(best_cv_f1, 4),
            "val_accuracy": round(acc, 4),
            "val_precision_urgent": round(prec, 4),
            "val_recall_urgent": round(rec, 4),
            "val_f1_urgent": round(f1, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_roc_auc": round(roc_auc, 4),
            "val_pr_auc": round(pr_auc, 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "best_params": str(best_params),
            "artifact_path": model_path
        })
        
    tuned_df = pd.DataFrame(tuned_results)
    tuning_csv_path = os.path.join(BASE_DIR, "models", "tuning", "tuning_results.csv")
    os.makedirs(os.path.dirname(tuning_csv_path), exist_ok=True)
    tuned_df.to_csv(tuning_csv_path, index=False)
    print(f"\nSaved tuning results to {tuning_csv_path}")
    
    # ---------------------------------------------------------
    # 2. Systematic Probability Threshold Calibration for Best XGBoost
    # ---------------------------------------------------------
    best_xgb = best_fitted_models["XGBoost (Tuned)"]
    best_xgb_probas = np.asarray(best_xgb.predict_proba(X_val))
    xgb_val_probas = best_xgb_probas[:, 1]
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    thresh_records = []
    
    print("\n" + "="*70)
    print("PROBABILITY THRESHOLD CALIBRATION (XGBoost Tuned)")
    print("="*70)
    
    for th in thresholds:
        y_th_pred = (xgb_val_probas >= th).astype(int)
        
        cm = confusion_matrix(y_val, y_th_pred)
        tn, fp, fn, tp = cm.ravel()
        
        p = precision_score(y_val, y_th_pred, pos_label=1, zero_division=0)
        r = recall_score(y_val, y_th_pred, pos_label=1, zero_division=0)
        f = f1_score(y_val, y_th_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_val, y_th_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        thresh_records.append({
            "threshold": th,
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "specificity": round(specificity, 4),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn)
        })
        
    thresh_df = pd.DataFrame(thresh_records)
    thresh_csv_path = "models/tuning/threshold_calibration.csv"
    thresh_df.to_csv(thresh_csv_path, index=False)
    print(thresh_df.to_string(index=False))
    print(f"\nSaved threshold calibration results to {thresh_csv_path}")
    
    return tuned_df, thresh_df, best_fitted_models

if __name__ == '__main__':
    tuned_df, thresh_df, best_fitted_models = tune_models()
    print("\nTuning & calibration completed successfully.")
