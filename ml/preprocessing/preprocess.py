"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS Preprocessing and Feature Engineering Pipeline

Prepares training, validation, and test datasets for multi-target ML models:
1. failure_within_30d_target (Classification)
2. maintenance_duration_hours_target (Regression)
3. priority_score_target (Regression)
4. block_required_target (Classification)

Strictly prevents data leakage:
- Excludes all *_target columns (including priority_class_target) from predictors.
- Performs GroupKFold splitting by asset_id to eliminate asset overlap across train/val/test.
- Fits transformers ONLY on training data.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer

RANDOM_SEED = 42

ORDINAL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3
}

TARGET_COLS = [
    'failure_within_30d_target',
    'maintenance_duration_hours_target',
    'priority_score_target',
    'priority_class_target',
    'block_required_target'
]

ID_COLS = [
    'event_id',
    'event_date',
    'asset_id',
    'station_code',
    'station_name',
    'section_id',
    'officer_observation'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-grounded engineered features safely without data leakage.
    Operates on a copy of the dataframe.
    """
    df_out = df.copy()
    
    # 1. Maintenance Overdue Ratio: overdue days / interval
    interval = df_out['maintenance_interval_days'].clip(lower=1)
    df_out['maintenance_overdue_ratio'] = df_out['maintenance_overdue_days'] / interval
    
    # 2. Traffic Impact Ratio: affected services / scheduled proxy
    sched = df_out['scheduled_services_count_proxy'].clip(lower=1)
    df_out['traffic_impact_ratio'] = (df_out['affected_services_if_blocked'] / sched).clip(lower=0.0, upper=1.0)
    
    # 3. Recent Failure Density: 90d failures normalized by 365d failures
    f365 = df_out['failures_last_365d'].clip(lower=1)
    df_out['recent_failure_ratio'] = df_out['failures_last_90d'] / f365
    
    # 4. Same Defect Recurrence Rate
    df_out['recurrence_ratio'] = df_out['same_defect_recurrences_365d'] / f365
    
    # 5. Ordinal Mappings
    df_out['asset_criticality_ord'] = df_out['asset_criticality'].map(ORDINAL_MAP).fillna(1).astype(int)
    df_out['defect_severity_ord'] = df_out['defect_severity_label'].map(ORDINAL_MAP).fillna(1).astype(int)
    
    # 6. Boolean Mappings
    df_out['alternative_route_available'] = df_out['alternative_route_available'].astype(int)
    df_out['inspection_image_available'] = df_out['inspection_image_available'].astype(int)
    
    return df_out


def create_preprocessed_datasets(raw_data_path: str = "data/tejas_pilot_master_dataset.csv", output_dir: str = "data/processed"):
    print("=" * 70)
    print("STARTING PREPROCESSING PIPELINE FOR TEJAS PILOT MASTER DATASET")
    print(f"Source: {raw_data_path}")
    print("=" * 70)
    
    df_raw = pd.read_csv(raw_data_path)
    n_records = len(df_raw)
    print(f"Loaded raw dataset: {n_records} records, {df_raw.shape[1]} columns")
    
    # Extract targets dictionary and metadata
    targets = {
        'failure_within_30d_target': df_raw['failure_within_30d_target'].astype(int),
        'maintenance_duration_hours_target': df_raw['maintenance_duration_hours_target'].astype(float),
        'priority_score_target': df_raw['priority_score_target'].astype(float),
        'block_required_target': df_raw['block_required_target'].astype(int),
        'priority_class_target': df_raw['priority_class_target']
    }
    
    groups = df_raw['asset_id'].copy()
    
    # Feature engineering
    df_feat = engineer_features(df_raw)
    
    # Drop IDs and all target columns
    drop_cols = ID_COLS + TARGET_COLS + ['asset_criticality', 'defect_severity_label']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])
    
    print(f"Predictor feature matrix shape: {X.shape} ({X.shape[1]} candidate features)")
    print(f"Verified: 0 target columns present in feature matrix -> {any(c in X.columns for c in TARGET_COLS)}")
    
    # ---------------------------------------------------------
    # Train / Val / Test Group Splitting (70% / 15% / 15%)
    # ---------------------------------------------------------
    gkf = GroupKFold(n_splits=20)
    splits = list(gkf.split(X, targets['failure_within_30d_target'], groups))
    
    # 14 splits (70%) train, 3 splits (15%) val, 3 splits (15%) test
    train_idx = np.concatenate([splits[i][1] for i in range(14)])
    val_idx = np.concatenate([splits[i][1] for i in range(14, 17)])
    test_idx = np.concatenate([splits[i][1] for i in range(17, 20)])
    
    X_train: pd.DataFrame = pd.DataFrame(X.iloc[train_idx].copy())
    X_val: pd.DataFrame = pd.DataFrame(X.iloc[val_idx].copy())
    X_test: pd.DataFrame = pd.DataFrame(X.iloc[test_idx].copy())
    
    print(f"\nDataset Splits:")
    print(f"  Train: {len(train_idx)} records ({len(train_idx)/n_records*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} records ({len(val_idx)/n_records*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} records ({len(test_idx)/n_records*100:.1f}%)")
    
    # Verify strict zero asset leakage
    train_assets = set(groups.iloc[train_idx])
    val_assets = set(groups.iloc[val_idx])
    test_assets = set(groups.iloc[test_idx])
    assert len(train_assets.intersection(val_assets)) == 0, "CRITICAL ERROR: Asset leakage between train and val!"
    assert len(train_assets.intersection(test_assets)) == 0, "CRITICAL ERROR: Asset leakage between train and test!"
    assert len(val_assets.intersection(test_assets)) == 0, "CRITICAL ERROR: Asset leakage between val and test!"
    print("Zero-Leakage Verification Passed: No shared asset_ids across train, validation, and test sets.")
    
    # ---------------------------------------------------------
    # Scikit-Learn ColumnTransformer Pipeline
    # ---------------------------------------------------------
    nominal_cols = ['department', 'asset_type', 'safety_function', 'zone', 'state', 'defect_type']
    nominal_cols = [c for c in nominal_cols if c in X.columns]
    
    robust_num_cols = [
        'maintenance_overdue_days', 'days_since_last_maintenance', 'defect_duration_days',
        'failures_last_365d', 'same_defect_recurrences_365d', 'maintenance_overdue_ratio',
        'traffic_impact_ratio', 'recent_failure_ratio', 'recurrence_ratio'
    ]
    robust_num_cols = [c for c in robust_num_cols if c in X.columns]
    
    standard_num_cols = [
        'asset_age_years', 'asset_criticality_score', 'scheduled_services_count_proxy',
        'real_daily_train_count', 'real_criticality_score', 'network_neighbor_degree', 'affected_services_if_blocked',
        'maintenance_interval_days', 'failures_last_30d', 'failures_last_90d'
    ]
    standard_num_cols = [c for c in standard_num_cols if c in X.columns]
    
    quantile_cols = ['real_daily_train_count']
    quantile_cols = [c for c in quantile_cols if c in X.columns]
    
    passthrough_cols = ['asset_criticality_ord', 'defect_severity_ord', 'alternative_route_available', 'inspection_image_available', 'is_real_traffic_data']
    passthrough_cols = [c for c in passthrough_cols if c in X.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_cols),
            ('robust', RobustScaler(), robust_num_cols),
            ('standard', StandardScaler(), standard_num_cols),
            ('quantile', QuantileTransformer(n_quantiles=1000), quantile_cols),
            ('passthrough', 'passthrough', passthrough_cols)
        ],
        remainder='drop'
    )
    
    print("\nFitting ColumnTransformer on X_train...")
    preprocessor.fit(X_train)
    
    ohe_feature_names = list(preprocessor.named_transformers_['ohe'].get_feature_names_out(nominal_cols))
    all_feature_names = ohe_feature_names + robust_num_cols + standard_num_cols + [c + '_quantile' for c in quantile_cols] + passthrough_cols
    print(f"Transformed feature space dimensionality: {len(all_feature_names)} features")
    
    # Helper to convert to dense numpy array
    def to_dense(mat):
        if hasattr(mat, "toarray"):
            return mat.toarray()
        return np.asarray(mat)

    # Transform splits
    X_train_proc = pd.DataFrame(to_dense(preprocessor.transform(X_train)), columns=all_feature_names, index=X_train.index)
    X_val_proc = pd.DataFrame(to_dense(preprocessor.transform(X_val)), columns=all_feature_names, index=X_val.index)
    X_test_proc = pd.DataFrame(to_dense(preprocessor.transform(X_test)), columns=all_feature_names, index=X_test.index)
    
    # Attach all target labels to processed splits
    def attach_targets(df_feat_proc, indices):
        df_full = df_feat_proc.copy()
        for tname, tseries in targets.items():
            df_full[tname] = tseries.iloc[indices].values
        return df_full
    
    train_df = attach_targets(X_train_proc, train_idx)
    val_df = attach_targets(X_val_proc, val_idx)
    test_df = attach_targets(X_test_proc, test_idx)
    
    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("preprocessing", exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    pipeline_path = os.path.join("preprocessing", "preprocessing_pipeline.pkl")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "feature_names": all_feature_names,
        "nominal_cols": nominal_cols,
        "robust_num_cols": robust_num_cols,
        "standard_num_cols": standard_num_cols,
        "passthrough_cols": passthrough_cols,
        "ordinal_mapping": ORDINAL_MAP,
        "target_cols": TARGET_COLS,
        "random_seed": RANDOM_SEED
    }
    
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_artifact, f)
        
    print(f"\nSaved processed datasets:")
    print(f"  {train_path} ({train_df.shape})")
    print(f"  {val_path} ({val_df.shape})")
    print(f"  {test_path} ({test_df.shape})")
    print(f"  {pipeline_path}")
    print("=" * 70)
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_names": all_feature_names,
        "pipeline_artifact": pipeline_artifact
    }

if __name__ == '__main__':
    create_preprocessed_datasets()
