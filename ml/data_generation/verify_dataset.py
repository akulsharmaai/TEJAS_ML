import pandas as pd

def verify():
    df = pd.read_csv('ml/data/tejas_pilot_master_dataset.csv')
    tests_passed = True
    
    print("--- DATA INTEGRITY TESTS ---")
    
    # 1. task_id unique
    if 'task_id' in df.columns and df['task_id'].is_unique:
        print("PASS: task_id is unique.")
    else:
        print("FAIL: task_id is missing or not unique!")
        tests_passed = False
        
    # 2. Leaky columns removed
    leaky = ['traffic_percentile', 'operational_impact_score']
    for col in leaky:
        if col not in df.columns:
            print(f"PASS: {col} removed successfully.")
        else:
            print(f"FAIL: {col} still present!")
            tests_passed = False
            
    # 3. Legacy targets isolated
    targets = [
        'failure_within_30d_target',
        'maintenance_duration_hours_target',
        'priority_score_target',
        'priority_class_target',
        'block_required_target'
    ]
    for col in targets:
        if col not in df.columns:
            print(f"PASS: legacy target {col} isolated.")
        else:
            print(f"FAIL: legacy target {col} still present!")
            tests_passed = False
            
    # 4. Enriched fields preserved
    enriched = ['real_section_code', 'real_daily_train_count', 'real_criticality_score', 'is_real_traffic_data']
    for col in enriched:
        if col in df.columns:
            print(f"PASS: Enriched feature {col} preserved.")
        else:
            print(f"FAIL: Enriched feature {col} missing!")
            tests_passed = False
            
    # 5. section_id untouched
    if 'section_id' in df.columns:
        print(f"PASS: section_id preserved.")
    else:
        print("FAIL: section_id missing!")
        tests_passed = False
        
    if tests_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")

if __name__ == '__main__':
    verify()
