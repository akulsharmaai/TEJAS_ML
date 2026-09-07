# Source Attribution & Provenance File

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Task**: Predictive Maintenance Task Urgency & Risk Scoring Dataset

---

## 1. Provenance Taxonomy

To maintain strict scientific integrity and transparency, all data components are categorized under three distinct tiers:

1. **PUBLIC REAL DATA**: Directly extracted or empirically fitted from public operational Indian Railways databases and government repositories.
2. **PUBLIC SYNTHETIC DATA**: Referenced and benchmarked from peer-reviewed or community-published synthetic railway datasets.
3. **OUR SYNTHETIC SIH DATA**: Domain-modeled, multi-factor generated synthetic maintenance task records developed specifically to bridge missing operational fields according to Indian Railways technical codes (RDSO / IRPWM / SEM / ACTM).

---

## 2. Public Real Data Sources

### Source A: Indian Railways Train Timetable & Operational Network Data
* **Repository / Source**: Government of India Open Government Data (OGD) / DataMeet Indian Railways Project
* **URLs**:
  - https://www.data.gov.in/resource/indian-railways-time-table-trains-available-reservation-03082015
  - https://github.com/datameet/railways (National Train Enquiry System - NTES extract)
* **Local Raw Files**: 
  - `ml/data/raw/datameet_trains.json` (5,208 train route features)
  - `ml/data/raw/datameet_stations.json` (747 stations)
* **Extracted Parameters (`ml/data/sources/indian_railways_traffic_stats.json`)**:
  - Daily passenger train frequencies (median: 6-15 trains/day on branch lines, 75-140 on standard lines, 140-255 on High-Density Corridors / Golden Quadrilateral).
  - Freight vs. passenger traffic ratio splits (35%–60% freight on dedicated corridors vs. 15%–35% on branch lines).
* **Role in Final Dataset**: Directly parameterized `trains_per_day` and `goods_trains_per_day` distributions and the real-world scale of `operational_impact`.

---

## 3. Public Synthetic Data Sources

### Source B: Indian Railway Failure Detection & Maintenance (100K)
* **Repository**: Hugging Face Hub (`shambhuraje/Indian_Railway_maintance`)
* **URL**: https://huggingface.co/datasets/shambhuraje/Indian_Railway_maintance
* **License**: Creative Commons Attribution 4.0 (CC BY 4.0)
* **Status**: **PUBLIC SYNTHETIC DATA** (Authored synthetically by Shambhuraje for IoT/predictive maintenance experimentation).
* **Local Raw Files**: 
  - `ml/data/raw/hf_indian_railway_maintenance_sample.csv`
  - `ml/data/raw/hf_indian_railway_maintenance_readme.md`
* **Role in Final Dataset**: Inspected and referenced to understand failure type categories, sensor health correlation concepts, and inspection score ranges. Was **NOT** blindly merged because it represents IoT sensor time-series rather than railway maintenance block tasks.

---

## 4. Academic & Domain Technical References

### Source C: Railway Track Surface Faults Dataset
* **Publisher / Host**: Mendeley Data / *Data in Brief* (Elsevier, DOI: [10.17632/8hxtgyyxrw.2](https://doi.org/10.17632/8hxtgyyxrw.2))
* **Authors**: Asfar Arain, Sanaullah Mehran, Muhammad Zakir Shaikh, Dileep Kumar, Tanweer Hussain, Bhawani Shankar Chowdhry (MUET)
* **License**: Creative Commons Attribution 4.0 (CC BY 4.0)
* **Local Metadata**: `ml/data/sources/mendeley_track_faults_meta.json`
* **Role in Final Dataset**: Grounded track surface defect nomenclature into 7 realistic physical categories: Squats (Rolling Contact Fatigue), Cracks (Transverse/Fatigue), Flakings, Shellings, Spallings, Grooves, and Insulated Rail Joint Faults.

### Source D: Indian Railways Technical Manuals & RDSO Norms
* **Organizations**: Research Designs and Standards Organisation (RDSO), Ministry of Railways, Government of India.
* **Referenced Codes**:
  1. *Indian Railways Permanent Way Manual (IRPWM)* – Track assets, tolerances, LWR/CWR breathing, USFD testing schedules, turnout wear tolerances.
  2. *Signal Engineering Manual (SEM Part I & II)* – Point machines, track circuits, digital axle counters, electronic interlocking maintenance cycles.
  3. *AC Traction Manual (ACTM Vol II)* – 25kV OHE, contact wire wear limits (>20%), ATD pulley tensioning, traction substations, vacuum circuit breakers.
* **Role in Final Dataset**: Provided authoritative domain basis for asset classification across 3 departments (`Engineering`, `S&T`, `Traction`), maintenance intervals (`maintenance_frequency`), standard block maintenance durations (`maintenance_duration_hours`), and asset lifecycle profiles (`asset_age_years`).

---

## 5. Our Synthetic SIH Data Generation

* **Generator Script**: `ml/data_generation/generate_dataset.py`
* **Output File**: `ml/data/processed/railway_maintenance_tasks.csv`
* **Total Records**: 40,000 maintenance tasks
* **Design Philosophy**:
  1. Real-world Indian Railways maintenance task schema (20 variables).
  2. Strict alignment with RDSO operational rules and traffic distributions.
  3. Continuous latent multi-factor risk index preventing single-feature shortcuts.
  4. Fully reproducible with fixed random seed (`RANDOM_SEED = 42`).
