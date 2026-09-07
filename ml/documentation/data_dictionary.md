# Data Dictionary: SIH26027 Railway Maintenance Task Urgency Dataset

**Problem Statement**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Task**: Maintenance Task Urgency & Risk Scoring  
**Target Variable**: `urgent` (Binary: `0` = Routine/Scheduled Maintenance, `1` = Urgent / Priority Block Allocation Required)  
**Total Records**: 40,000  
**Total Columns**: 20  
**File Location**: `ml/data/processed/railway_maintenance_tasks.csv`

---

## Column Specifications

| Column Name | Data Type | Permissible Values / Range | Description | Domain Context & Source |
|:---|:---|:---|:---|:---|
| `task_id` | String (Identifier) | `TSK-IR-000001` to `TSK-IR-040000` | Unique alphanumeric identifier assigned to each maintenance work order or inspection task. | Indian Railways Computerized Track Management System (TMS) work order convention. |
| `asset_id` | String (Identifier) | `AST-ENG-XXXXX`, `AST-SNT-XXXXX`, `AST-TRD-XXXXX` | Unique identifier of the physical railway asset being inspected or serviced. Multiple historical tasks can link to the same asset ID. | Represents physical asset tag in Indian Railways Asset Management System (IR-AMS / FMS). |
| `department` | String (Categorical) | `Engineering`, `S&T`, `Traction` | The railway department responsible for maintaining the asset and executing the block maintenance. | 3 Core Technical Departments in Indian Railways (Track/Civil, Signal & Telecom, Traction Distribution). |
| `asset_type` | String (Categorical) | 24 domain asset classes across 3 departments (see breakdown below) | Specific technical classification of the railway asset. | Indian Railways Permanent Way Manual (IRPWM), Signal Engineering Manual (SEM), and AC Traction Manual (ACTM). |
| `asset_age_years` | Float (Continuous) | `0.5` to `45.0` years | Age of the asset in operational service since installation or last capital overhaul/replacement. | Distributed according to asset-specific design lifespans (e.g. Rails 10-20 yrs, Bridges 45+ yrs, Electronics 10-15 yrs). |
| `asset_criticality` | String (Ordinal) | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Strategic and operational importance of the asset within the railway network hierarchy. | Based on Indian Railways route classification (High Density Network 'HDN', Highly Utilized Network 'HUN', Class A/B/C/D lines). |
| `defect_type` | String (Categorical) | 40+ domain-specific defect types | The observed or diagnosed failure mode, degradation symptom, or defect. | Mendeley Track Surface Faults (8hxtgyyxrw/2) + RDSO inspection guidelines. |
| `defect_severity` | String (Ordinal) | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Qualitative engineering grading of the defect's current physical deterioration. | RDSO / IRPWM Track Condition Index & defect classification norms. |
| `num_open_defects` | Integer (Discrete) | `1` to `6` | Number of unresolved or concurrent defects recorded on the same asset or cluster. | Field inspection logs tracking composite wear. |
| `days_since_defect` | Integer (Discrete) | `1` to `90` days | Number of days elapsed since the defect was first logged by inspection or IoT sensor. | Inspection lag and work-order backlog indicator. |
| `days_overdue` | Integer (Discrete) | `0` to `65` days | Number of days by which scheduled maintenance exceeds the mandatory periodic SLA. `0` indicates maintenance is within window. | IRPWM / SEM periodic maintenance tolerance tracking. |
| `maintenance_frequency` | String (Categorical) | `Weekly`, `Fortnightly`, `Monthly`, `Quarterly`, `Half-Yearly`, `Annual` | Mandatory scheduled maintenance cadence specified in Indian Railways maintenance manuals. | RDSO maintenance schedules. |
| `days_since_last_maintenance` | Integer (Discrete) | `2` to `450` days | Actual elapsed days since the last successful routine maintenance or overhaul was executed. | Logged maintenance history. |
| `previous_failures` | Integer (Discrete) | `0` to `15` | Total lifetime historical breakdowns or unscheduled failure incidents recorded on this asset. | Asset lifecycle reliability history (Mean Time Between Failures). |
| `failures_last_12_months` | Integer (Discrete) | `0` to `6` | Unscheduled failures or signal/track interruptions caused by this asset in the preceding 12 months. | Key short-term reliability and recurrent failure indicator. |
| `trains_per_day` | Integer (Discrete) | `10` to `255` trains/day | Total average daily train traffic (passenger + freight) passing over this asset's section. | Derived from Indian Railways Timetable & NTES route density data. |
| `goods_trains_per_day` | Integer (Discrete) | `2` to `120` trains/day | Average daily freight/goods train traffic (heavy axle loads > 22.82t / 25t). | Derived from DFC and Indian Railways freight corridor operational statistics. |
| `operational_impact` | String (Ordinal) | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Potential disruption level on train punctuality, cascading delay, and revenue if asset fails. | Composite index reflecting route traffic density, alternative loop availability, and bottleneck severity. |
| `maintenance_duration_hours` | Float (Continuous) | `0.50` to `7.00` hours | Estimated shadow or active traffic block window required by maintenance crew to rectify the defect. | Standard Indian Railways block duration norms for track tamping, OHE cut-and-splice, point motor replacement, etc. |
| `urgent` | Integer (Binary Target) | `0` or `1` | Target classification label: `1` if maintenance task requires urgent priority block allocation (< 24-72 hrs); `0` if routine/schedulable. | Multi-factor risk synthesis function combining severity, criticality, overdue strain, traffic, and history. |

---

## Breakdown of Asset Types by Department

### 1. Engineering (Civil / Track, P-Way & Structures) - 8 Asset Classes
1. `Rails (60kg/52kg)`: Mainline running rails subject to dynamic contact stresses, wheel burns, and fatigue.
2. `Prestressed Concrete Sleepers (PSC)`: Monoblock PSC sleepers supporting rails and maintaining gauge.
3. `Turnout / Points & Crossing (1 in 12 / 1 in 8.5)`: High-wear track switching installations.
4. `Elastic Rail Clip & Fastenings`: Elastic rail clips (ERC Mk-III/V), liners, and rubber pads.
5. `Ballast Cushion & Formation`: 300mm/350mm crushed stone ballast bed and subgrade formation.
6. `Glued Insulated Rail Joint (GJ)`: G3(L) glued joints providing electrical isolation for track circuits.
7. `Switch Expansion Joint (SEJ)`: Thermal breathing joints at ends of Long Welded Rails (LWR/CWR).
8. `Girder Bridge / Culvert Bearing`: Steel rocker/roller bearings and elastomeric pads on railway bridges.

### 2. S&T (Signal & Telecommunication) - 7 Asset Classes
1. `Electric Point Machine (IRS/Siemens)`: Electro-mechanical actuator operating track switch blades.
2. `Digital Axle Counter (MSDAC/SSDAC)`: Multi-section / single-section wheel counting safety systems.
3. `DC Track Circuit (TC)`: Closed-loop electrical rail circuit for train presence detection.
4. `Electronic Interlocking (EI Rack)`: Solid-state computer-based failsafe interlocking logic.
5. `Color Light Signal (LED Unit)`: Long-range multi-aspect LED railway signals.
6. `Interlocked Level Crossing (LC Gate)`: Electrically interlocked lifting barriers with road-rail interlocking.
7. `Signalling Power Supply / IPS`: Integrated Power Supply system with battery backup and solar inverter.

### 3. Traction (TRD / OHE & PSI) - 8 Asset Classes
1. `Contact Wire (107/150 sq mm Cu)`: Hard-drawn copper overhead contact wire delivering 25kV AC to pantographs.
2. `Catenary Wire (65/125 sq mm)`: Cadmium copper / all-aluminium alloy supporting stranded conductor.
3. `Section Insulator Assembly`: Air-gap or composite insulating assembly separating OHE electrical sections.
4. `25kV Vacuum Circuit Breaker (VCB)`: High-voltage substation circuit breaker and interrupter.
5. `Cantilever Assembly & Regulating Bracket`: Galvanized steel mast cantilever and composite stay insulators.
6. `Auto Tensioning Device (ATD 3:1 Pulley)`: Weight-and-pulley mechanical tensioning mechanism.
7. `Traction Power Transformer (25kV/21.6MVA)`: 132kV/25kV or 220kV/25kV railway traction substation transformer.
8. `Neutral Section (PTFE type)`: Dead-zone phase break assembly separating adjacent traction substations.
