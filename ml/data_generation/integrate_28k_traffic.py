import pandas as pd
import numpy as np

MASTER_CSV_PATH = "ml/data/tejas_pilot_master_dataset.csv"
TRAFFIC_CSV_PATH = "export_section_traffic_for_ml(2).csv"

print("=" * 80)
print("AUDIT-COMPLIANT INTEGRATION OF 28K SECTION TRAFFIC DATASET")
print("=" * 80)

# Load datasets
master_df = pd.read_csv(MASTER_CSV_PATH)
print(f"Loaded master dataset: {len(master_df)} rows")

traffic_df = pd.read_csv(TRAFFIC_CSV_PATH)
print(f"Loaded traffic dataset: {len(traffic_df)} rows")

# Build a mapping from station_code to section traffic data
station_to_section = {}
for _, row in traffic_df.iterrows():
    section_code = row['section_code']
    daily_train = row['daily_train_count']
    crit_score = row['criticality_score']
    from_st = row['from_station_code']
    to_st = row['to_station_code']
    
    # Prioritize from_station
    if pd.notna(from_st) and from_st not in station_to_section:
        station_to_section[from_st] = {
            'real_section_code': section_code,
            'real_daily_train_count': daily_train,
            'real_criticality_score': crit_score
        }
    
    if pd.notna(to_st) and to_st not in station_to_section:
        station_to_section[to_st] = {
            'real_section_code': section_code,
            'real_daily_train_count': daily_train,
            'real_criticality_score': crit_score
        }

# Arrays for new columns
new_section_codes = []
new_daily_train = []
new_crit_score = []
is_real = []

mapped_count = 0
unmapped_count = 0

for _, row in master_df.iterrows():
    st_code = row['station_code']
    if st_code in station_to_section:
        data = station_to_section[st_code]
        new_section_codes.append(data['real_section_code'])
        new_daily_train.append(data['real_daily_train_count'])
        new_crit_score.append(data['real_criticality_score'])
        is_real.append(True)
        mapped_count += 1
    else:
        # Fallback
        new_section_codes.append('SYNTHETIC_FALLBACK')
        # We fallback to the synthetic proxy
        new_daily_train.append(row.get('scheduled_services_count_proxy', 30))
        # Fallback to asset_criticality_score scaled to 0-1
        new_crit_score.append(row.get('asset_criticality_score', 50.0) / 100.0)
        is_real.append(False)
        unmapped_count += 1

# Add columns WITHOUT destroying section_id or asset_id
master_df['real_section_code'] = new_section_codes
master_df['real_daily_train_count'] = new_daily_train
master_df['real_criticality_score'] = new_crit_score
master_df['is_real_traffic_data'] = is_real

# We do NOT compute traffic_percentile or operational_impact_score here to avoid data leakage.
# The preprocessing pipeline will handle normalizations fitted purely on the train split.

print(f"Mapped {mapped_count} rows successfully to real data.")
print(f"Fell back {unmapped_count} rows to synthetic proxy data.")

master_df.to_csv(MASTER_CSV_PATH, index=False)
print(f"Successfully updated and saved {MASTER_CSV_PATH}")
print("=" * 80)
