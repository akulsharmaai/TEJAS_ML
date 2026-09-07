# Data Synthesis Methodology: Combining Empirical Data & Domain Rules

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Document**: Data Synthesis & Engineering Specification

---

## 1. Executive Rationale: Why Synthetic Task-Level Data?

Indian Railways operational databases (e.g. TMS, ICMS, COA, FMS) contain restricted safety and security data that are not publicly released at individual track-task granularity with unified multi-department defect logs. Available public datasets either provide:
1. **Raw Timetable Schedules** (Train numbers, stations, timings), or
2. **Computer Vision Images** (Rail surface cracks, bounding boxes), or
3. **IoT Sensor Streams** (Synthetic accelerometer/temperature readings).

However, **SIH26027 requires maintenance tasks as the core entity**—comprising cross-departmental coordination (Engineering, S&T, Traction), defect severity, historical asset failure metrics, traffic density, and required block maintenance duration.

To bridge this gap without fabricating ungrounded data, we formulated a **Domain-Grounded Hybrid Synthesis Pipeline**:
- Real traffic and network structure parameterize operational load (`trains_per_day`, `goods_trains_per_day`, `operational_impact`).
- Mendeley and RDSO engineering norms define defect taxonomies and severity hierarchies.
- Lifetime asset pools model realistic physical asset re-inspection over time.
- Inter-feature correlation modeling ensures statistical and physical coherence.

```
+-----------------------------------------------------------------------------------+
|                            PUBLIC INPUT SOURCES                                   |
|                                                                                   |
|  [Real Timetable & Routes]     [Mendeley Track Faults]     [RDSO / IRPWM / SEM]   |
|   (Traffic Density & Ratios)    (7 Surface Fault Classes)   (Asset Types & Manuals)|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                           DOMAIN SYNTHESIS ENGINE                                 |
|                                                                                   |
|  1. Asset Registry (11,000 physical assets across 3 departments with aging curves) |
|  2. Cross-Departmental Fault Engine (40+ specific failure modes across 24 assets)  |
|  3. Traffic Load & Operational Impact Mapping (Trunk vs. Branch distributions)    |
|  4. Work-Order Temporal Progression (Days overdue, open defect clusters, history)  |
|  5. Non-Linear Multi-Factor Latent Risk Model with Stochastic Noise Injection      |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                              FINAL SIH DATASET                                    |
|             40,000 Maintenance Tasks | 20 Schema-Compliant Columns                |
+-----------------------------------------------------------------------------------+
```

---

## 2. Departmental & Asset Architecture

The dataset covers the three core technical departments in Indian Railways:

### A. Engineering Department (Civil / Track & Structures) ~ 40% Share
* **Physical Assets**: Rails (60kg/52kg), Prestressed Concrete Sleepers (PSC), Turnouts / Points & Crossing (1 in 12 / 1 in 8.5), Elastic Rail Clip & Fastenings, Ballast Cushion & Formation, Glued Insulated Rail Joints (GJ), Switch Expansion Joints (SEJ), Girder Bridges / Culvert Bearings.
* **Aging Distribution**: Rails (0.5–20 yrs), PSC Sleepers (1–30 yrs), Bridges (2–45 yrs). Modeled using Beta distributions to reflect network life-cycle profiles.
* **Maintenance Durations**: 1.5 to 6.5 hours depending on tamping, welding, rail renewal, or bearing replacement.

### B. Signal & Telecommunication (S&T) ~ 32% Share
* **Physical Assets**: Electric Point Machine (IRS/Siemens), Digital Axle Counter (MSDAC/SSDAC), DC Track Circuit, Electronic Interlocking (EI Rack), Color Light Signal (LED Unit), Interlocked Level Crossing (LC Gate), Signalling Power Supply / IPS.
* **Aging Distribution**: Electronics (0.5–12 yrs), Point Machines (0.5–15 yrs), LC Gates (0.5–18 yrs).
* **Maintenance Durations**: 0.75 to 4.0 hours for card replacement, contact burnishing, point detection adjustment, or cable re-termination.

### C. Traction Distribution (TRD / OHE & Substation) ~ 28% Share
* **Physical Assets**: Contact Wire (107/150 sq mm), Catenary Wire (65/125 sq mm), Section Insulator Assembly, 25kV Vacuum Circuit Breaker (VCB), Cantilever Assembly & Regulating Bracket, Auto Tensioning Device (ATD 3:1 Pulley), Traction Power Transformer (25kV/21.6MVA), Neutral Section (PTFE type).
* **Aging Distribution**: OHE Conductors (0.5–20 yrs), Substation Transformers (1–35 yrs), ATD Pulleys (0.5–20 yrs).
* **Maintenance Durations**: 1.5 to 6.5 hours for power block isolation, OHE wire restringing, DGA oil filtration, or insulator wash.

---

## 3. Physical Asset Pooling & Historical Tracking

Rather than generating 40,000 isolated independent rows, we instantiated an **Asset Registry of 11,000 persistent physical assets**:
- `AST-ENG-00001` to `AST-ENG-04500` (4,500 Engineering assets)
- `AST-SNT-00001` to `AST-SNT-03500` (3,500 S&T assets)
- `AST-TRD-00001` to `AST-TRD-03000` (3,000 Traction assets)

Each maintenance task links to an asset ID, preserving consistent asset properties (`asset_type`, baseline `asset_age_years`) while capturing sequential maintenance states (`days_since_last_maintenance`, `days_overdue`, `previous_failures`).

---

## 4. Modeling Inter-Feature Correlations

To ensure physical realism, variables are statistically and conditionally correlated:

1. **Defect Severity & Reporting Latency**:
   - `CRITICAL` defects are logged rapidly (mean `days_since_defect` ~ 2–8 days) and exhibit shorter overdue tolerances before urgent intervention.
   - `LOW` severity defects can remain in backlog longer (mean `days_since_defect` ~ 15–50 days).

2. **Traffic & Operational Impact**:
   - Assets situated on heavy trunk routes (`trains_per_day` 100–255, `goods_trains_per_day` 35–120) carry higher `operational_impact` (`HIGH`/`CRITICAL`) due to the absence of alternate diversion paths and the risk of severe cascading delays.

3. **Asset Age & Failure Progression**:
   - Older assets (`asset_age_years` > 15) have higher Poisson rates of `previous_failures` and multiple open defects (`num_open_defects`).

4. **Maintenance Frequency & Days Since Last Maintenance**:
   - `days_since_last_maintenance` equals the manual's mandated cycle (`maintenance_frequency` in days) plus `days_overdue`, adjusted with realistic operational jitter ($\pm 10$ days).

---

## 5. Summary of Synthesis Integrity

* **Total Records**: 40,000
* **Completeness**: 100% (0 missing or null values).
* **Reproducibility**: Deterministic generation with fixed random seed `RANDOM_SEED = 42`.
* **Zero Contamination**: Cleanly separated raw downloads (`ml/data/raw/`), extracted sources (`ml/data/sources/`), generation scripts (`ml/data_generation/`), and output datasets (`ml/data/processed/`).
