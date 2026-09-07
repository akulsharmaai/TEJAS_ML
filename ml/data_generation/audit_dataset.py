import pandas as pd
import numpy as np

def audit():
    df = pd.read_csv('ml/data/tejas_pilot_master_dataset.csv')
    print("Columns:", df.columns.tolist())
    
    # 1. Trace provenance (we will analyze values)
    print("\n--- Value Audit ---")
    for col in ['traffic_percentile', 'operational_impact_score', 'asset_criticality_score', 'real_criticality_score']:
        if col in df.columns:
            print(f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.3f}")

    # Inspect operational_impact_score formula
    if 'operational_impact_score' in df.columns:
        corr = df[['operational_impact_score', 'traffic_percentile', 'asset_criticality_score']].corr()
        print("\nCorrelation of operational_impact_score:")
        print(corr)

if __name__ == '__main__':
    audit()
