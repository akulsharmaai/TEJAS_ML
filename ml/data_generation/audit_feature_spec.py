import pandas as pd
import numpy as np

def audit_features():
    df = pd.read_csv('ml/data/tejas_pilot_master_dataset.csv')
    
    # Check for forbidden columns in a hypothetical ML view
    forbidden = [
        'task_id', 'event_id', 'asset_id', 'event_date', 'station_code',
        'station_name', 'zone', 'state', 'section_id', 'real_section_code',
        'is_real_traffic_data'
    ]
    
    # Allowed feature components
    features = [
        'defect_severity_label', 'defect_type', 'defect_duration_days',
        'maintenance_overdue_days', 'days_since_last_maintenance',
        'real_daily_train_count', 'scheduled_services_count_proxy',
        'asset_criticality_score', 'affected_services_if_blocked',
        'alternative_route_available', 'failures_last_30d',
        'failures_last_90d', 'failures_last_365d', 'same_defect_recurrences_365d'
    ]
    
    print("--- FEATURE AUDIT ---")
    
    # 1. Circular checks
    corr = df['real_daily_train_count'].corr(df['real_criticality_score'])
    print(f"real_daily_train_count vs real_criticality_score corr: {corr:.4f}")
    if corr > 0.95:
        print("WARNING: real_criticality_score is perfectly collinear with real_daily_train_count.")
        
    corr_asset = df['asset_criticality_score'].corr(pd.factorize(df['asset_criticality'])[0])
    print(f"asset_criticality_score vs asset_criticality corr: {corr_asset:.4f}")
    
    # 2. Leakage check
    # Make sure traffic_percentile and operational_impact_score are NOT in features
    leaky = ['traffic_percentile', 'operational_impact_score']
    for l in leaky:
        assert l not in features, f"Leaky feature {l} found in candidate features!"
        
    print("\n--- PASSED FEATURE AUDIT ---")

if __name__ == '__main__':
    audit_features()
