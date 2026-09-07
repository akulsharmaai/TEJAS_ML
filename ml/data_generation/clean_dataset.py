import pandas as pd
import shutil

def clean():
    file_path = 'ml/data/tejas_pilot_master_dataset.csv'
    shutil.copy(file_path, file_path + '.backup')
    
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Create task_id
    df['task_id'] = df['event_id']
    if not df['task_id'].is_unique:
        raise ValueError("task_id is not unique!")
    
    # Reorder task_id next to event_id
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('task_id')))
    df = df[cols]
    
    # Columns to drop
    leaky_cols = ['traffic_percentile', 'operational_impact_score']
    legacy_targets = [
        'failure_within_30d_target',
        'maintenance_duration_hours_target',
        'priority_score_target',
        'priority_class_target',
        'block_required_target'
    ]
    
    # Optional: save legacy targets to another file just in case
    df[['task_id'] + legacy_targets].to_csv('ml/data/tejas_pilot_legacy_targets.csv', index=False)
    
    # Drop them
    df = df.drop(columns=leaky_cols + legacy_targets)
    
    print(f"New shape: {df.shape}")
    
    df.to_csv(file_path, index=False)
    
    # Write feature table
    with open('ml/data/feature_table_summary.md', 'w') as f:
        f.write("| Feature Name | Source | User/System | Use in Model? | Reason |\n")
        f.write("|--------------|--------|-------------|---------------|--------|\n")
        for c in df.columns:
            source = "REAL" if "real_" in c or "is_real" in c else "SYNTHETIC/MIXED"
            system = "System" if "score" in c or c in ["asset_age_years", "days_since_last_maintenance", "maintenance_overdue_days"] else "Mixed"
            f.write(f"| {c} | {source} | {system} | Yes | Kept after audit |\n")
            
    print("Cleaned dataset saved.")

if __name__ == '__main__':
    clean()
