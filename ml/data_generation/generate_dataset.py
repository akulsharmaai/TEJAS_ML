"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Dataset Generation Pipeline for Railway Maintenance Task Urgency Prediction

Generates a domain-grounded, scientifically defensible dataset of 40,000
maintenance task records across Engineering, S&T, and Traction departments.
"""

import os
import csv
import json
import random
import math
from collections import Counter

# Set deterministic seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DOMAIN_SPECS = {
    "Engineering": {
        "share": 0.40,
        "assets": {
            "Rails (60kg/52kg)": {
                "age_range": (0.5, 20.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.25), ("HIGH", 0.45), ("CRITICAL", 0.25)],
                "maintenance_freq": ["Fortnightly", "Monthly", "Quarterly"],
                "base_duration": (2.0, 4.5),
                "defects": [
                    ("Squat (Rolling Contact Fatigue)", 0.20, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Gauge Corner Cracking (GCC)", 0.18, ["LOW", "MEDIUM", "HIGH"]),
                    ("Thermit Weld Defect", 0.15, ["HIGH", "CRITICAL"]),
                    ("Transverse Rail Fatigue Crack", 0.12, ["HIGH", "CRITICAL"]),
                    ("Rail Corrugation & Wear", 0.15, ["LOW", "MEDIUM", "HIGH"]),
                    ("Wheel Burn & Scabbing", 0.10, ["LOW", "MEDIUM"]),
                    ("Shelling & Spalling", 0.10, ["MEDIUM", "HIGH", "CRITICAL"])
                ]
            },
            "Prestressed Concrete Sleepers (PSC)": {
                "age_range": (1.0, 30.0),
                "criticality_weights": [("LOW", 0.15), ("MEDIUM", 0.45), ("HIGH", 0.30), ("CRITICAL", 0.10)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly"],
                "base_duration": (1.5, 3.5),
                "defects": [
                    ("Longitudinal Sleeper Cracking", 0.30, ["LOW", "MEDIUM", "HIGH"]),
                    ("Rail Seat Abrasion & Deterioration", 0.25, ["MEDIUM", "HIGH"]),
                    ("Insert Loose / Corroded", 0.25, ["LOW", "MEDIUM", "HIGH"]),
                    ("Center Bound Cracking Under Dynamic Load", 0.20, ["HIGH", "CRITICAL"])
                ]
            },
            "Turnout / Points & Crossing (1 in 12 / 1 in 8.5)": {
                "age_range": (0.5, 15.0),
                "criticality_weights": [("LOW", 0.02), ("MEDIUM", 0.18), ("HIGH", 0.45), ("CRITICAL", 0.35)],
                "maintenance_freq": ["Weekly", "Fortnightly", "Monthly"],
                "base_duration": (2.5, 5.0),
                "defects": [
                    ("Tongue Rail Chipping / Wear", 0.30, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Nose of Crossing Severe Wear", 0.25, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Stock Rail Gauge Corner Cut", 0.20, ["LOW", "MEDIUM", "HIGH"]),
                    ("Stretcher Bar Distortion", 0.15, ["HIGH", "CRITICAL"]),
                    ("Check Rail Clearance Deviation", 0.10, ["MEDIUM", "HIGH"])
                ]
            },
            "Elastic Rail Clip & Fastenings": {
                "age_range": (0.5, 12.0),
                "criticality_weights": [("LOW", 0.20), ("MEDIUM", 0.45), ("HIGH", 0.25), ("CRITICAL", 0.10)],
                "maintenance_freq": ["Fortnightly", "Monthly", "Quarterly"],
                "base_duration": (1.0, 2.5),
                "defects": [
                    ("Missing / Broken Elastic Rail Clips (ERC)", 0.40, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Rubber Pad Perished / Displaced", 0.30, ["LOW", "MEDIUM", "HIGH"]),
                    ("Liner Missing / Crushed", 0.20, ["LOW", "MEDIUM"]),
                    ("Fastening Bolt Sheared", 0.10, ["HIGH", "CRITICAL"])
                ]
            },
            "Ballast Cushion & Formation": {
                "age_range": (1.0, 25.0),
                "criticality_weights": [("LOW", 0.15), ("MEDIUM", 0.40), ("HIGH", 0.35), ("CRITICAL", 0.10)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (2.5, 6.0),
                "defects": [
                    ("Ballast Deficiency & Caking", 0.35, ["LOW", "MEDIUM", "HIGH"]),
                    ("Mud Pumping & Foul Ballast in Monsoon", 0.30, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Formation Subsidence / Settlement", 0.20, ["HIGH", "CRITICAL"]),
                    ("Slack Packing & Void Under Sleepers", 0.15, ["MEDIUM", "HIGH"])
                ]
            },
            "Glued Insulated Rail Joint (GJ)": {
                "age_range": (0.5, 10.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.25), ("HIGH", 0.50), ("CRITICAL", 0.20)],
                "maintenance_freq": ["Fortnightly", "Monthly", "Quarterly"],
                "base_duration": (1.5, 3.5),
                "defects": [
                    ("Insulating End Post Crushed", 0.40, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Fishplate Bolt Loose / Sheared", 0.30, ["HIGH", "CRITICAL"]),
                    ("Electrical Insulation Resistance Drop", 0.30, ["MEDIUM", "HIGH", "CRITICAL"])
                ]
            },
            "Switch Expansion Joint (SEJ)": {
                "age_range": (0.5, 15.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.30), ("HIGH", 0.45), ("CRITICAL", 0.20)],
                "maintenance_freq": ["Fortnightly", "Monthly"],
                "base_duration": (1.5, 3.0),
                "defects": [
                    ("SEJ Gap Beyond Permissible Thermal Limit", 0.50, ["HIGH", "CRITICAL"]),
                    ("Roller Plate Seized / Jammed", 0.30, ["MEDIUM", "HIGH"]),
                    ("Central Sleeper Spacing Fault", 0.20, ["LOW", "MEDIUM"])
                ]
            },
            "Girder Bridge / Culvert Bearing": {
                "age_range": (2.0, 45.0),
                "criticality_weights": [("LOW", 0.02), ("MEDIUM", 0.18), ("HIGH", 0.50), ("CRITICAL", 0.30)],
                "maintenance_freq": ["Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (3.0, 6.5),
                "defects": [
                    ("Rocker / Roller Bearing Seizure", 0.35, ["HIGH", "CRITICAL"]),
                    ("Bed Block Structural Crack", 0.30, ["HIGH", "CRITICAL"]),
                    ("Expansion Joint Choked with Debris", 0.20, ["LOW", "MEDIUM", "HIGH"]),
                    ("Bearing Greasing Depletion & Tilt", 0.15, ["MEDIUM", "HIGH"])
                ]
            }
        }
    },
    "S&T": {
        "share": 0.32,
        "assets": {
            "Electric Point Machine (IRS/Siemens)": {
                "age_range": (0.5, 15.0),
                "criticality_weights": [("LOW", 0.01), ("MEDIUM", 0.14), ("HIGH", 0.50), ("CRITICAL", 0.35)],
                "maintenance_freq": ["Weekly", "Fortnightly", "Monthly"],
                "base_duration": (1.0, 2.5),
                "defects": [
                    ("Point Detection Contact Carbonization", 0.30, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Friction Clutch Slippage / Over-Torque", 0.25, ["MEDIUM", "HIGH"]),
                    ("Drive Motor High Current Draw / Stall", 0.20, ["HIGH", "CRITICAL"]),
                    ("Lock Bar & Split Pin Obstruction", 0.15, ["HIGH", "CRITICAL"]),
                    ("Ground Connection Rod Bush Wear", 0.10, ["LOW", "MEDIUM"])
                ]
            },
            "Digital Axle Counter (MSDAC/SSDAC)": {
                "age_range": (0.5, 12.0),
                "criticality_weights": [("LOW", 0.01), ("MEDIUM", 0.15), ("HIGH", 0.44), ("CRITICAL", 0.40)],
                "maintenance_freq": ["Fortnightly", "Monthly", "Quarterly"],
                "base_duration": (1.0, 3.0),
                "defects": [
                    ("Wheel Sensor Detection Coil Misalignment", 0.35, ["HIGH", "CRITICAL"]),
                    ("Telegram Transmission Bit Error / Reset Trip", 0.30, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Evaluator Card Processing Fault", 0.20, ["HIGH", "CRITICAL"]),
                    ("Outdoor Sensor Clamp Loose Due to Vibration", 0.15, ["MEDIUM", "HIGH"])
                ]
            },
            "DC Track Circuit (TC)": {
                "age_range": (0.5, 20.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.30), ("HIGH", 0.45), ("CRITICAL", 0.20)],
                "maintenance_freq": ["Weekly", "Fortnightly", "Monthly"],
                "base_duration": (1.0, 2.0),
                "defects": [
                    ("Track Relay Drop / False Occupancy", 0.35, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Bond Wire Severed / High Resistance", 0.25, ["HIGH", "CRITICAL"]),
                    ("Ballast Resistance Low in Rain", 0.25, ["MEDIUM", "HIGH"]),
                    ("Feed End Resistance Drift", 0.15, ["LOW", "MEDIUM"])
                ]
            },
            "Electronic Interlocking (EI Rack)": {
                "age_range": (0.5, 12.0),
                "criticality_weights": [("LOW", 0.00), ("MEDIUM", 0.05), ("HIGH", 0.45), ("CRITICAL", 0.50)],
                "maintenance_freq": ["Monthly", "Quarterly"],
                "base_duration": (1.5, 4.0),
                "defects": [
                    ("Standby CPU Card Health Mismatch", 0.35, ["HIGH", "CRITICAL"]),
                    ("Vital Input/Output Card Communication Error", 0.30, ["HIGH", "CRITICAL"]),
                    ("Optical Fiber Communication Loss", 0.20, ["HIGH", "CRITICAL"]),
                    ("Cabinet Overheating & Fan Filter Choke", 0.15, ["LOW", "MEDIUM", "HIGH"])
                ]
            },
            "Color Light Signal (LED Unit)": {
                "age_range": (0.5, 10.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.35), ("HIGH", 0.45), ("CRITICAL", 0.15)],
                "maintenance_freq": ["Monthly", "Quarterly"],
                "base_duration": (0.75, 2.0),
                "defects": [
                    ("Current Sensor Threshold Mismatch / Blank Signal", 0.40, ["HIGH", "CRITICAL"]),
                    ("LED Aspect Cluster Partial Failure (>20%)", 0.30, ["MEDIUM", "HIGH"]),
                    ("Signal Post Alignment Deviation", 0.15, ["LOW", "MEDIUM"]),
                    ("Transformer ECR Current Drop", 0.15, ["MEDIUM", "HIGH"])
                ]
            },
            "Interlocked Level Crossing (LC Gate)": {
                "age_range": (0.5, 18.0),
                "criticality_weights": [("LOW", 0.02), ("MEDIUM", 0.20), ("HIGH", 0.50), ("CRITICAL", 0.28)],
                "maintenance_freq": ["Weekly", "Fortnightly", "Monthly"],
                "base_duration": (1.0, 2.5),
                "defects": [
                    ("Boom Locking Plunger Misaligned", 0.40, ["HIGH", "CRITICAL"]),
                    ("Circuit Closer Contact Chattering", 0.25, ["MEDIUM", "HIGH"]),
                    ("Winch Wire Rope Frayed", 0.20, ["MEDIUM", "HIGH"]),
                    ("Warning Hooter & Flashers Disconnected", 0.15, ["HIGH", "CRITICAL"])
                ]
            },
            "Signalling Power Supply / IPS": {
                "age_range": (0.5, 15.0),
                "criticality_weights": [("LOW", 0.02), ("MEDIUM", 0.18), ("HIGH", 0.50), ("CRITICAL", 0.30)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly"],
                "base_duration": (1.5, 3.5),
                "defects": [
                    ("Inverter Module Fault / Load Transfer", 0.35, ["HIGH", "CRITICAL"]),
                    ("Battery Bank Individual Cell Voltage Low", 0.30, ["MEDIUM", "HIGH"]),
                    ("Surge Protection Device (SPD) Blown", 0.20, ["HIGH", "CRITICAL"]),
                    ("DC-DC Converter Ripple Exceeds Limit", 0.15, ["LOW", "MEDIUM"])
                ]
            }
        }
    },
    "Traction": {
        "share": 0.28,
        "assets": {
            "Contact Wire (107/150 sq mm Cu)": {
                "age_range": (0.5, 20.0),
                "criticality_weights": [("LOW", 0.02), ("MEDIUM", 0.20), ("HIGH", 0.48), ("CRITICAL", 0.30)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly"],
                "base_duration": (2.0, 4.5),
                "defects": [
                    ("Severe Contact Wire Wear (>20% cross-section)", 0.35, ["HIGH", "CRITICAL"]),
                    ("Local Kink / Arc Blemish", 0.25, ["MEDIUM", "HIGH", "CRITICAL"]),
                    ("Contact Wire Height & Stagger Deviation", 0.25, ["LOW", "MEDIUM", "HIGH"]),
                    ("Dropper Clip Loose / Damaged", 0.15, ["LOW", "MEDIUM"])
                ]
            },
            "Catenary Wire (65/125 sq mm)": {
                "age_range": (0.5, 25.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.35), ("HIGH", 0.45), ("CRITICAL", 0.15)],
                "maintenance_freq": ["Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (2.0, 4.0),
                "defects": [
                    ("Strand Severed / Loose Outer Strand", 0.40, ["HIGH", "CRITICAL"]),
                    ("Corrosive Pitting Near Industrial Area", 0.30, ["LOW", "MEDIUM", "HIGH"]),
                    ("Bridle Wire Tension Discrepancy", 0.30, ["MEDIUM", "HIGH"])
                ]
            },
            "Section Insulator Assembly": {
                "age_range": (0.5, 12.0),
                "criticality_weights": [("LOW", 0.01), ("MEDIUM", 0.15), ("HIGH", 0.50), ("CRITICAL", 0.34)],
                "maintenance_freq": ["Monthly", "Quarterly"],
                "base_duration": (1.5, 3.5),
                "defects": [
                    ("Runner Strip Erosion & Arcing Grooves", 0.40, ["HIGH", "CRITICAL"]),
                    ("Insulator Glaze Puncture / Flashover Mark", 0.30, ["HIGH", "CRITICAL"]),
                    ("Section Insulator Sag / Level Defect", 0.30, ["MEDIUM", "HIGH"])
                ]
            },
            "25kV Vacuum Circuit Breaker (VCB)": {
                "age_range": (0.5, 20.0),
                "criticality_weights": [("LOW", 0.01), ("MEDIUM", 0.10), ("HIGH", 0.45), ("CRITICAL", 0.44)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (2.0, 4.0),
                "defects": [
                    ("Vacuum Bottle Contact Resistance High", 0.35, ["HIGH", "CRITICAL"]),
                    ("Operating Mechanism Pneumatic / SF6 Pressure Low", 0.30, ["HIGH", "CRITICAL"]),
                    ("Trip Coil Timing Exceeds Tolerance", 0.20, ["HIGH", "CRITICAL"]),
                    ("Auxiliary Switch Chatter / Oxidation", 0.15, ["LOW", "MEDIUM"])
                ]
            },
            "Cantilever Assembly & Regulating Bracket": {
                "age_range": (0.5, 25.0),
                "criticality_weights": [("LOW", 0.10), ("MEDIUM", 0.40), ("HIGH", 0.38), ("CRITICAL", 0.12)],
                "maintenance_freq": ["Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (1.5, 3.0),
                "defects": [
                    ("Stay Arm / Bracket Tube Corrosion", 0.35, ["LOW", "MEDIUM", "HIGH"]),
                    ("Porcelain / Composite Insulator Flashover Tracking", 0.35, ["HIGH", "CRITICAL"]),
                    ("Swivelling Clip Stiff / Jammed", 0.30, ["MEDIUM", "HIGH"])
                ]
            },
            "Auto Tensioning Device (ATD 3:1 Pulley)": {
                "age_range": (0.5, 20.0),
                "criticality_weights": [("LOW", 0.05), ("MEDIUM", 0.25), ("HIGH", 0.45), ("CRITICAL", 0.25)],
                "maintenance_freq": ["Monthly", "Quarterly", "Half-Yearly"],
                "base_duration": (1.5, 3.0),
                "defects": [
                    ("X-Y Dimension Out of Temperature Chart Limit", 0.45, ["HIGH", "CRITICAL"]),
                    ("Stainless Steel Wire Rope Strands Snapped", 0.30, ["HIGH", "CRITICAL"]),
                    ("Pulley Bearing Seizure / Stiff Movement", 0.25, ["MEDIUM", "HIGH"])
                ]
            },
            "Traction Power Transformer (25kV/21.6MVA)": {
                "age_range": (1.0, 35.0),
                "criticality_weights": [("LOW", 0.00), ("MEDIUM", 0.08), ("HIGH", 0.42), ("CRITICAL", 0.50)],
                "maintenance_freq": ["Quarterly", "Half-Yearly", "Annual"],
                "base_duration": (3.5, 6.5),
                "defects": [
                    ("Dissolved Gas Analysis (DGA) Acetylene Surge", 0.40, ["HIGH", "CRITICAL"]),
                    ("Bushing Oil Leakage & Thermal Hotspot", 0.25, ["HIGH", "CRITICAL"]),
                    ("Winding Temperature Alarm Trip", 0.20, ["HIGH", "CRITICAL"]),
                    ("Silica Gel Breather Saturated & Moisture Ingress", 0.15, ["LOW", "MEDIUM", "HIGH"])
                ]
            },
            "Neutral Section (PTFE type)": {
                "age_range": (0.5, 12.0),
                "criticality_weights": [("LOW", 0.01), ("MEDIUM", 0.15), ("HIGH", 0.44), ("CRITICAL", 0.40)],
                "maintenance_freq": ["Fortnightly", "Monthly", "Quarterly"],
                "base_duration": (2.0, 4.0),
                "defects": [
                    ("PTFE Insulating Rod Tracking & Surface Arcing", 0.45, ["HIGH", "CRITICAL"]),
                    ("Arcing Horn Burnout & Gap Excessive", 0.35, ["HIGH", "CRITICAL"]),
                    ("Transition Joint Loose / Uneven", 0.20, ["MEDIUM", "HIGH"])
                ]
            }
        }
    }
}

FREQ_DAYS = {
    "Daily": 1,
    "Weekly": 7,
    "Fortnightly": 15,
    "Monthly": 30,
    "Quarterly": 90,
    "Half-Yearly": 180,
    "Annual": 365
}

def weighted_choice(choices):
    # choices is list of (item, weight)
    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for item, w in choices:
        if upto + w >= r:
            return item
        upto += w
    return choices[-1][0]

def poisson_sample(lam):
    # Knuth algorithm for poisson
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def generate_dataset(num_records=40000):
    records = []
    
    depts = [("Engineering", 0.40), ("S&T", 0.32), ("Traction", 0.28)]
    
    asset_pool = {
        "Engineering": [f"AST-ENG-{i:05d}" for i in range(1, 4501)],
        "S&T": [f"AST-SNT-{i:05d}" for i in range(1, 3501)],
        "Traction": [f"AST-TRD-{i:05d}" for i in range(1, 3001)],
    }
    
    asset_registry = {}
    for dept, aids in asset_pool.items():
        asset_types = list(DOMAIN_SPECS[dept]["assets"].keys())
        for aid in aids:
            atype = random.choice(asset_types)
            age_min, age_max = DOMAIN_SPECS[dept]["assets"][atype]["age_range"]
            # Beta-like distribution using betavariate
            age = round(random.betavariate(2.0, 2.5) * (age_max - age_min) + age_min, 1)
            asset_registry[aid] = {
                "department": dept,
                "asset_type": atype,
                "asset_age_years": age
            }
            
    print(f"Initialized asset registry with {len(asset_registry)} physical assets.")

    for i in range(1, num_records + 1):
        task_id = f"TSK-IR-{i:06d}"
        dept = weighted_choice(depts)
        
        aid = random.choice(asset_pool[dept])
        meta = asset_registry[aid]
        atype = str(meta["asset_type"])
        age = float(meta["asset_age_years"])
        
        aspec = DOMAIN_SPECS[dept]["assets"][atype]
        
        # 1. Asset Criticality
        asset_criticality = weighted_choice(aspec["criticality_weights"])
        
        # 2. Defect Type and Defect Severity
        defect_choices = [(d[0], d[1], d[2]) for d in aspec["defects"]]
        d_name, _, d_sevs = weighted_choice([(d, d[1]) for d in defect_choices])
        defect_severity = random.choice(d_sevs)
        
        # 3. Number of Open Defects
        sev_add = 1.0 if defect_severity in ['HIGH', 'CRITICAL'] else 0.0
        lam_def = 1.2 + (age / 32.0) + sev_add
        num_open_defects = max(1, min(6, poisson_sample(lam_def)))
        
        # 4. Maintenance Frequency
        maintenance_frequency = random.choice(aspec["maintenance_freq"])
        freq_day = FREQ_DAYS[maintenance_frequency]
        
        # 5. Days Overdue and Days Since Defect
        if defect_severity == "CRITICAL":
            days_since_defect = int(random.gammavariate(2.0, 3.0) + 1)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 5.0) - 2.0))
        elif defect_severity == "HIGH":
            days_since_defect = int(random.gammavariate(3.0, 5.0) + 2)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 8.0) - 2.0))
        elif defect_severity == "MEDIUM":
            days_since_defect = int(random.gammavariate(4.0, 7.0) + 3)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 12.0) - 3.0))
        else: # LOW
            days_since_defect = int(random.gammavariate(4.0, 10.0) + 5)
            days_overdue = int(max(0.0, random.expovariate(1.0 / 15.0) - 4.0))
            
        days_since_defect = max(1, min(90, days_since_defect))
        days_overdue = max(0, min(65, days_overdue))
        
        # 6. Days Since Last Maintenance
        days_since_last_maintenance = max(2, min(450, freq_day + days_overdue + random.randint(-4, 12)))
        
        # 7. Previous Failures and Failures in Last 12 Months
        fail_lam = 0.4 + (age / 10.0) * 0.8 + (0.5 if asset_criticality == "CRITICAL" else 0.0)
        previous_failures = max(0, min(15, poisson_sample(fail_lam * 2.5)))
        # Binomial sample
        f12 = sum(1 for _ in range(previous_failures) if random.random() < 0.35) + poisson_sample(0.2)
        failures_last_12_months = max(0, min(min(previous_failures + 1, 6), f12))
        
        # 8. Train Traffic
        is_trunk_route = random.random() < 0.65
        if is_trunk_route:
            trains_per_day = int(max(65, min(255, random.gauss(135, 35))))
            goods_trains_per_day = int(max(20, min(120, trains_per_day * random.uniform(0.35, 0.58))))
        else:
            trains_per_day = int(max(10, min(64, random.gauss(38, 14))))
            goods_trains_per_day = int(max(2, min(25, trains_per_day * random.uniform(0.15, 0.40))))
            
        # 9. Operational Impact
        crit_factor = 0.3 if asset_criticality in ['HIGH', 'CRITICAL'] else 0.0
        impact_score = (trains_per_day / 240.0) * 0.5 + crit_factor + (goods_trains_per_day / 110.0) * 0.2 + random.gauss(0, 0.08)
        if impact_score > 0.68:
            operational_impact = "CRITICAL"
        elif impact_score > 0.46:
            operational_impact = "HIGH"
        elif impact_score > 0.26:
            operational_impact = "MEDIUM"
        else:
            operational_impact = "LOW"
            
        # 10. Maintenance Duration Hours
        dur_min, dur_max = aspec["base_duration"]
        dur_modifier = (0.5 if defect_severity in ["HIGH", "CRITICAL"] else 0.0) + (num_open_defects - 1) * 0.2
        maintenance_duration_hours = round(max(0.5, min(7.0, random.uniform(dur_min, dur_max) + dur_modifier + random.gauss(0, 0.2))), 2)
        
        # 11. Target Generation: Multi-Factor Latent Risk Score
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
        temporal_strain = min(45.0, days_overdue * 0.9 + days_since_defect * 0.35)
        history_strain = min(30.0, failures_last_12_months * 5.0 + previous_failures * 1.5 + (age / 35.0) * 10.0)
        traffic_strain = (trains_per_day / 250.0) * 12.0 + (goods_trains_per_day / 120.0) * 8.0
        defect_cluster_strain = (num_open_defects - 1) * 4.0
        
        # Smooth continuous multi-factor interaction (no artificial step-cliffs)
        synergy_interaction = (s_norm * c_norm * 16.0) + (s_norm * o_norm * 8.0) + (c_norm * o_norm * 4.0)
        stochastic_noise = random.gauss(0.0, 5.5)
        
        latent_risk_score = (
            base_component +
            synergy_interaction +
            0.40 * temporal_strain +
            0.35 * history_strain +
            0.40 * traffic_strain +
            0.50 * defect_cluster_strain +
            stochastic_noise
        )
        
        # Logistic probability calibration for thresholding
        # Calibrated such that ~30-32% of maintenance tasks are classified as urgent
        threshold = 82.0
        temperature = 7.5
        logit = (latent_risk_score - threshold) / temperature
        # Sigmoid probability
        prob_urgent = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        
        urgent = 1 if (random.random() < prob_urgent) else 0
        
        records.append({
            "task_id": task_id,
            "asset_id": aid,
            "department": dept,
            "asset_type": atype,
            "asset_age_years": age,
            "asset_criticality": asset_criticality,
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
        
    # Write to CSV
    output_path = os.path.join("data", "processed", "railway_maintenance_tasks.csv")
    fieldnames = [
        "task_id", "asset_id", "department", "asset_type", "asset_age_years",
        "asset_criticality", "defect_type", "defect_severity", "num_open_defects",
        "days_since_defect", "days_overdue", "maintenance_frequency",
        "days_since_last_maintenance", "previous_failures", "failures_last_12_months",
        "trains_per_day", "goods_trains_per_day", "operational_impact",
        "maintenance_duration_hours", "urgent"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Generated and saved {len(records)} records to {output_path}")
    return records

def print_summary_stats(records):
    total = len(records)
    print("\n" + "="*50)
    print(f"DATASET VALIDATION REPORT (Total Records: {total})")
    print("="*50)
    
    urgent_counts = Counter(r["urgent"] for r in records)
    print("\nUrgent Distribution (0/1):")
    for k, v in sorted(urgent_counts.items()):
        print(f"  {k}: {v:,} ({v/total*100:.2f}%)")
        
    dept_counts = Counter(r["department"] for r in records)
    print("\nDepartment Distribution:")
    for k, v in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,} ({v/total*100:.2f}%)")
        
    print("\nAsset Type Distribution (Top 10):")
    asset_counts = Counter(r["asset_type"] for r in records)
    for k, v in asset_counts.most_common(10):
        print(f"  {k}: {v:,} ({v/total*100:.2f}%)")
        
    sev_counts = Counter(r["defect_severity"] for r in records)
    print("\nDefect Severity Distribution:")
    for k, v in sorted(sev_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,} ({v/total*100:.2f}%)")
        
    crit_counts = Counter(r["asset_criticality"] for r in records)
    print("\nAsset Criticality Distribution:")
    for k, v in sorted(crit_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,} ({v/total*100:.2f}%)")
        
    # Check nulls or missing fields
    missing_counts = 0
    for r in records:
        for k, v in r.items():
            if v is None or v == "":
                missing_counts += 1
    print(f"\nMissing/Null Field Count across all cells: {missing_counts}")

if __name__ == '__main__':
    records = generate_dataset(40000)
    print_summary_stats(records)
