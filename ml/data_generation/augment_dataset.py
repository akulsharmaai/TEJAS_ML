"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Dataset Augmentation Script: Generates 20,000 balanced synthetic records with smooth target logic,
producing a high-quality 60,000-row master dataset.
"""

import os
import sys
import csv
import json
import random
import math

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
from collections import Counter
from data_generation.generate_dataset import DOMAIN_SPECS, FREQ_DAYS, poisson_sample, weighted_choice

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CRITICALITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
IMPACT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def compute_smooth_latent_target(
    defect_severity: str,
    asset_criticality: str,
    operational_impact: str,
    days_overdue: int,
    days_since_defect: int,
    failures_last_12_months: int,
    previous_failures: int,
    age: float,
    trains_per_day: int,
    goods_trains_per_day: int,
    num_open_defects: int,
    noise_std: float = 5.5
):
    sev_map = {"LOW": 10.0, "MEDIUM": 35.0, "HIGH": 70.0, "CRITICAL": 95.0}
    crit_map = {"LOW": 10.0, "MEDIUM": 35.0, "HIGH": 70.0, "CRITICAL": 95.0}
    ops_map = {"LOW": 10.0, "MEDIUM": 35.0, "HIGH": 70.0, "CRITICAL": 95.0}

    s_val = sev_map[defect_severity]
    c_val = crit_map[asset_criticality]
    o_val = ops_map[operational_impact]

    s_norm = s_val / 95.0
    c_norm = c_val / 95.0
    o_norm = o_val / 95.0

    base_component = 0.30 * s_val + 0.22 * c_val + 0.16 * o_val

    # Smooth multi-factor interaction without discrete cliff jumps
    synergy_interaction = (s_norm * c_norm * 16.0) + (s_norm * o_norm * 8.0) + (c_norm * o_norm * 4.0)

    temporal_strain = min(45.0, days_overdue * 0.9 + days_since_defect * 0.35)
    history_strain = min(30.0, failures_last_12_months * 5.0 + previous_failures * 1.5 + (age / 35.0) * 10.0)
    traffic_strain = (trains_per_day / 250.0) * 12.0 + (goods_trains_per_day / 120.0) * 8.0
    defect_cluster_strain = (num_open_defects - 1) * 4.0

    stochastic_noise = random.gauss(0.0, noise_std)

    latent_risk_score = (
        base_component +
        synergy_interaction +
        0.40 * temporal_strain +
        0.35 * history_strain +
        0.40 * traffic_strain +
        0.50 * defect_cluster_strain +
        stochastic_noise
    )

    threshold = 82.0
    temperature = 7.5
    logit = (latent_risk_score - threshold) / temperature
    prob_urgent = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
    urgent = 1 if (random.random() < prob_urgent) else 0

    return latent_risk_score, prob_urgent, urgent


def generate_augmented_records(start_task_idx=40001, num_records=20000):
    """
    Generates 20,000 new synthetic records with balanced representation and matched quadruplets.
    """
    records = []
    
    depts = [("Engineering", 0.40), ("S&T", 0.32), ("Traction", 0.28)]
    
    # Asset pool for the augmented dataset
    asset_pool = {
        "Engineering": [f"AST-ENG-{i:05d}" for i in range(4501, 7001)],
        "S&T": [f"AST-SNT-{i:05d}" for i in range(3501, 5501)],
        "Traction": [f"AST-TRD-{i:05d}" for i in range(3001, 5001)],
    }
    
    asset_registry = {}
    for dept, aids in asset_pool.items():
        asset_types = list(DOMAIN_SPECS[dept]["assets"].keys())
        for aid in aids:
            atype = random.choice(asset_types)
            age_min, age_max = DOMAIN_SPECS[dept]["assets"][atype]["age_range"]
            age = round(random.betavariate(2.0, 2.5) * (age_max - age_min) + age_min, 1)
            asset_registry[aid] = {
                "department": dept,
                "asset_type": atype,
                "asset_age_years": age
            }

    # Generate 5,000 quadruplets = 20,000 rows
    num_quadruplets = num_records // 4
    current_task_id = start_task_idx

    # Balanced severity cycling
    sev_cycle = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    impact_cycle = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    for q_idx in range(num_quadruplets):
        dept = weighted_choice(depts)
        aid = random.choice(asset_pool[dept])
        meta = asset_registry[aid]
        atype = str(meta["asset_type"])
        age = float(meta["asset_age_years"])
        aspec = DOMAIN_SPECS[dept]["assets"][atype]

        # Balanced target severity & impact
        target_sev = sev_cycle[q_idx % 4]
        target_impact = impact_cycle[(q_idx // 4) % 4]

        # Select a defect compatible with target_sev if possible
        defect_choices = [(d[0], d[1], d[2]) for d in aspec["defects"]]
        compatible_defects = [d for d in defect_choices if target_sev in d[2]]
        if compatible_defects:
            d_name, _, _ = random.choice(compatible_defects)
        else:
            d_name, _, _ = random.choice(defect_choices)
        defect_severity = target_sev

        # Number of Open Defects
        sev_add = 1.0 if defect_severity in ['HIGH', 'CRITICAL'] else 0.0
        lam_def = 1.2 + (age / 32.0) + sev_add
        num_open_defects = max(1, min(6, poisson_sample(lam_def)))

        # Maintenance Frequency
        maintenance_frequency = random.choice(aspec["maintenance_freq"])
        freq_day = FREQ_DAYS[maintenance_frequency]

        # Days Overdue & Days Since Defect
        if defect_severity == "CRITICAL":
            days_since_defect = int(random.gammavariate(2.0, 3.0) + 1)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 5.0) - 2.0))
        elif defect_severity == "HIGH":
            days_since_defect = int(random.gammavariate(3.0, 5.0) + 2)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 8.0) - 2.0))
        elif defect_severity == "MEDIUM":
            days_since_defect = int(random.gammavariate(4.0, 7.0) + 3)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 12.0) - 3.0))
        else:
            days_since_defect = int(random.gammavariate(4.0, 10.0) + 5)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 15.0) - 4.0))

        days_since_defect = max(1, min(90, days_since_defect))
        days_overdue = max(0, min(65, days_overdue))

        days_since_last_maintenance = max(2, min(450, freq_day + days_overdue + random.randint(-4, 12)))

        # Failures
        fail_lam = 0.4 + (age / 10.0) * 0.8
        previous_failures = max(0, min(15, poisson_sample(fail_lam * 2.5)))
        f12 = sum(1 for _ in range(previous_failures) if random.random() < 0.35) + poisson_sample(0.2)
        failures_last_12_months = max(0, min(min(previous_failures + 1, 6), f12))

        # Traffic
        is_trunk_route = random.random() < 0.65
        if is_trunk_route:
            trains_per_day = int(max(65, min(255, random.gauss(135, 35))))
            goods_trains_per_day = int(max(20, min(120, trains_per_day * random.uniform(0.35, 0.58))))
        else:
            trains_per_day = int(max(10, min(64, random.gauss(38, 14))))
            goods_trains_per_day = int(max(2, min(25, trains_per_day * random.uniform(0.15, 0.40))))

        operational_impact = target_impact

        # Duration
        dur_min, dur_max = aspec["base_duration"]
        dur_modifier = (0.5 if defect_severity in ["HIGH", "CRITICAL"] else 0.0) + (num_open_defects - 1) * 0.2
        maintenance_duration_hours = round(max(0.5, min(7.0, random.uniform(dur_min, dur_max) + dur_modifier + random.gauss(0, 0.2))), 2)

        # Generate matched quadruplet across all 4 criticality levels
        for crit in CRITICALITY_LEVELS:
            _, _, urgent = compute_smooth_latent_target(
                defect_severity=defect_severity,
                asset_criticality=crit,
                operational_impact=operational_impact,
                days_overdue=days_overdue,
                days_since_defect=days_since_defect,
                failures_last_12_months=failures_last_12_months,
                previous_failures=previous_failures,
                age=age,
                trains_per_day=trains_per_day,
                goods_trains_per_day=goods_trains_per_day,
                num_open_defects=num_open_defects
            )

            records.append({
                "task_id": f"TSK-IR-{current_task_id:06d}",
                "asset_id": aid,
                "department": dept,
                "asset_type": atype,
                "asset_age_years": age,
                "asset_criticality": crit,
                "defect_type": d_name,
                "defect_severity": defect_severity,
                "num_open_defects": num_open_defects,
                "days_since_defect": days_since_defect,
                "days_overdue": days_overdue,
                "maintenance_frequency": maintenance_frequency,
                "days_since_last_maintenance": days_since_last_maintenance,
                "previous_failures": previous_failures,
                "failures_last_12_months": failures_last_12_months,
                "trains_per_day": trains_per_day,
                "goods_trains_per_day": goods_trains_per_day,
                "operational_impact": operational_impact,
                "maintenance_duration_hours": maintenance_duration_hours,
                "urgent": urgent
            })
            current_task_id += 1

    return records


def build_augmented_master_dataset():
    original_path = "data/processed/railway_maintenance_tasks_40k_backup.csv"
    output_path = "data/processed/railway_maintenance_tasks.csv"

    print("Loading original 40,000-row dataset...")
    df_original = pd.read_csv(original_path)
    print(f"Original dataset shape: {df_original.shape}")

    print("Generating 20,000 new balanced synthetic records with smooth target logic...")
    new_records = generate_augmented_records(start_task_idx=40001, num_records=20000)
    df_new = pd.DataFrame(new_records)
    print(f"Generated {len(df_new)} new records.")

    print("Concatenating into 60,000-row master dataset...")
    df_combined = pd.concat([df_original, df_new], ignore_index=True)
    print(f"Combined master dataset shape: {df_combined.shape}")

    # Check duplicates
    dups = df_combined.duplicated(subset=[c for c in df_combined.columns if c != 'task_id']).sum()
    print(f"Exact feature duplicate rows: {dups}")

    df_combined.to_csv(output_path, index=False)
    print(f"Saved master dataset to {output_path}")

    return df_original, df_new, df_combined

if __name__ == "__main__":
    build_augmented_master_dataset()
