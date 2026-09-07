import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import pearsonr

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "tejas_pilot_master_dataset.csv")
MODEL_OUT_PATH = os.path.join(BASE_DIR, "models", "tuned_models", "urgency_model_pipeline.joblib")

def get_severity_bands(sev):
    if sev == 'LOW': return 10, 58
    if sev == 'MEDIUM': return 30, 70
    if sev == 'HIGH': return 55, 90
    if sev == 'CRITICAL': return 78, 100
    return np.nan, np.nan

def generate_target(df):
    df['base_sev'] = df['defect_severity_label'].apply(lambda x: get_severity_bands(x)[0])
    df['ceil_sev'] = df['defect_severity_label'].apply(lambda x: get_severity_bands(x)[1])
    
    overdue_norm = 1 - np.exp(-np.clip(df['maintenance_overdue_days'].fillna(0), 0, None) / 30)
    traffic_norm = 1 - np.exp(-np.clip(df['traffic_volume'], 0, None) / 40)
    crit_norm = np.clip(df['asset_criticality_score'].fillna(50.0), 0, 100) / 100
    fail_norm = 1 - np.exp(-np.clip(df['failures_last_365d'].fillna(0), 0, None) / 2)
    
    context_factor = (0.35 * overdue_norm + 0.25 * traffic_norm + 0.20 * crit_norm + 0.20 * fail_norm)
    urgency = df['base_sev'] + (df['ceil_sev'] - df['base_sev']) * context_factor
    return np.clip(urgency, 0, 100)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("Performing traffic routing...")
    traffic_routed = []
    for _, row in df.iterrows():
        is_real = bool(row.get('is_real_traffic_data', True))
        real_val = row.get('real_daily_train_count')
        proxy_val = row.get('scheduled_services_count_proxy')
        
        val = np.nan
        if is_real:
            if pd.notna(real_val): val = real_val
            elif pd.notna(proxy_val): val = proxy_val
        else:
            if pd.notna(proxy_val): val = proxy_val
            elif pd.notna(real_val): val = real_val
        traffic_routed.append(val)
        
    df['traffic_volume'] = traffic_routed
    df = df.dropna(subset=['traffic_volume'])
    
    # Generate deterministic target
    df['urgency_target_new'] = generate_target(df)
    df = df.dropna(subset=['urgency_target_new'])
    
    # Feature columns
    cat_cols = ['defect_severity_label']
    num_cols = ['maintenance_overdue_days', 'traffic_volume', 'asset_criticality_score', 'failures_last_365d']
    
    X = df[cat_cols + num_cols]
    y = df['urgency_target_new']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='MEDIUM')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols)
        ]
    )
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
        
    results = []
    trained_pipelines = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        preds = np.clip(preds, 0, 100) # Clamp output
        
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        pearson, _ = pearsonr(y_test, preds)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Pearson': pearson
        })
        trained_pipelines[name] = pipeline
        
    results_df = pd.DataFrame(results).sort_values(by='MAE')
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # We will choose HistGradientBoosting because it generally respects monotonic constraints well and is fast/robust.
    best_model_name = 'HistGradientBoosting'
    print(f"\nSelecting model: {best_model_name} because it handles non-linear bounded distributions well and generalizes predictably.")
    
    best_pipeline = trained_pipelines[best_model_name]
    
    # Validation against the problematic cases
    print("\n--- CRITICAL VALIDATION ---")
    print("The model is learning the approved TEJAS urgency-scoring function from synthetic/generated target labels. This does NOT represent validation against historical Indian Railways officer decisions.")
    
    case1 = pd.DataFrame([{
        'defect_severity_label': 'CRITICAL',
        'maintenance_overdue_days': 0,
        'traffic_volume': 73,
        'asset_criticality_score': 83,
        'failures_last_365d': 0
    }])
    case1_target = generate_target(case1).iloc[0]
    case1_pred = np.clip(best_pipeline.predict(case1)[0], 0, 100)
    
    case2 = pd.DataFrame([{
        'defect_severity_label': 'CRITICAL',
        'maintenance_overdue_days': 24,
        'traffic_volume': 18,
        'asset_criticality_score': 65.4,
        'failures_last_365d': 0
    }])
    case2_target = generate_target(case2).iloc[0]
    case2_pred = np.clip(best_pipeline.predict(case2)[0], 0, 100)
    
    print("\nOriginal critical-case predictions:")
    print(f"Case 1 | Deterministic Target: {case1_target:.2f} | ML Prediction: {case1_pred:.2f} | Difference: {abs(case1_target - case1_pred):.2f}")
    print(f"Case 2 | Deterministic Target: {case2_target:.2f} | ML Prediction: {case2_pred:.2f} | Difference: {abs(case2_target - case2_pred):.2f}")
    
    # Monotonicity test
    print("\nMonotonicity checks (Holding medium severity, base values, increasing one by one):")
    base_case = pd.DataFrame([{
        'defect_severity_label': 'MEDIUM',
        'maintenance_overdue_days': 10,
        'traffic_volume': 30,
        'asset_criticality_score': 50,
        'failures_last_365d': 1
    }])
    base_pred = best_pipeline.predict(base_case)[0]
    
    inc_overdue = base_case.copy(); inc_overdue['maintenance_overdue_days'] = 20
    inc_traffic = base_case.copy(); inc_traffic['traffic_volume'] = 40
    inc_crit = base_case.copy(); inc_crit['asset_criticality_score'] = 60
    inc_fail = base_case.copy(); inc_fail['failures_last_365d'] = 2
    inc_sev = base_case.copy(); inc_sev['defect_severity_label'] = 'HIGH'
    
    print(f"Base: {base_pred:.2f}")
    print(f"Inc Overdue ({best_pipeline.predict(inc_overdue)[0]:.2f}) >= Base: {best_pipeline.predict(inc_overdue)[0] >= base_pred - 1e-4}")
    print(f"Inc Traffic ({best_pipeline.predict(inc_traffic)[0]:.2f}) >= Base: {best_pipeline.predict(inc_traffic)[0] >= base_pred - 1e-4}")
    print(f"Inc Criticality ({best_pipeline.predict(inc_crit)[0]:.2f}) >= Base: {best_pipeline.predict(inc_crit)[0] >= base_pred - 1e-4}")
    print(f"Inc Failures ({best_pipeline.predict(inc_fail)[0]:.2f}) >= Base: {best_pipeline.predict(inc_fail)[0] >= base_pred - 1e-4}")
    print(f"Inc Severity ({best_pipeline.predict(inc_sev)[0]:.2f}) >= Base: {best_pipeline.predict(inc_sev)[0] >= base_pred - 1e-4}")
    
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    main()
