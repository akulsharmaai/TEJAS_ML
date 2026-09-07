# SHAP Explainability & Risk Factor Attribution Report

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Phase**: Step 11 – Model Explainability & Interpretability  
**Model Analyzed**: Tuned XGBoost Classifier (`models/tuned_models/xgboost_tuned.joblib`)  
**Methodology**: TreeSHAP (Lundberg et al.) with exact Shapley values on 137 input features  
**Evaluation Scope**: 5,997 Validation Task Records  

---

## 1. Executive Summary & Explainability Objective

In safety-critical Indian Railways operations, automatic block planning algorithms must provide transparent, actionable explanations to Chief Traction Engineers, Senior Section Engineers (P-Way), and Section Signal Officers. Machine learning predictions cannot remain "black-box" outputs.

By applying **TreeSHAP (SHapley Additive exPlanations)**, every risk prediction is decomposed into additive feature attributions:

$$g(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$

Where:
- $\phi_0 = -0.1990$ is the baseline log-odds expected value across the railway network.
- $\phi_j(x)$ is the precise mathematical contribution of feature $j$ for task $x$.
- $\phi_j > 0$ denotes a **Risk-Increasing Factor (+)** that drives the task towards urgent block prioritization.
- $\phi_j < 0$ denotes a **Risk-Mitigating Factor (-)** that supports routine maintenance scheduling.

---

## 2. Top 15 Most Influential Features Globally

Computed as the mean absolute SHAP value across all 5,997 validation tasks:

$$\text{Importance}_j = \frac{1}{N}\sum_{i=1}^N |\phi_{i, j}|$$

| Rank | Feature Name | Mean \|SHAP Value\| | Domain Operational Interpretation |
|:---:|:---|:---:|:---|
| **1** | `defect_severity` | **1.2145** | Primary physical driver: severe or critical rail/wire wear immediately escalates intervention urgency. |
| **2** | `asset_criticality` | **1.0190** | Strategic asset hierarchy: points, interlocking, and mainline OHE carry higher default baseline urgency. |
| **3** | `operational_impact` | **0.7164** | Route delay risk: assets located on high-density corridors (HDN/HUN) where failure halts cascading trains. |
| **4** | `num_open_defects` | **0.2045** | Composite degradation: multiple concurrent open defects signify accelerating failure probability. |
| **5** | `trains_per_day` | **0.1546** | Line saturation stress: heavy daily train traffic accelerates fatigue and limits maintenance buffer windows. |
| **6** | `days_overdue` | **0.1348** | Maintenance tolerance strain: exceeding mandated RDSO maintenance intervals compounds failure risk. |
| **7** | `failures_last_12_months` | **0.0990** | Recent breakdown velocity: assets with frequent recent trips indicate active unaddressed instability. |
| **8** | `previous_failures` | **0.0871** | Lifetime fatigue: cumulative historical breakdown count over asset lifecycle. |
| **9** | `failure_rate_indicator` | **0.0526** | Engineered rate ($\text{failures}/\text{age}$): highlights early infant mortality on young infrastructure. |
| **10** | `asset_age_years` | **0.0365** | Physical age wear: older assets require closer monitoring. |
| **11** | `maintenance_overdue_ratio` | **0.0292** | Overdue days normalized by maintenance cycle (e.g. Weekly vs Annual). |
| **12** | `maintenance_duration_hours` | **0.0253** | Complex repair jobs requiring longer shadow blocks. |
| **13** | `freight_traffic_ratio` | **0.0244** | Heavy axle load ratio ($\ge 22.82\text{t} / 25\text{t}$). |
| **14** | `days_since_defect` | **0.0233** | Reporting latency: rapidly logged critical defects vs older backlogged items. |
| **15** | `days_since_last_maintenance` | **0.0191** | Elapsed days since last overhaul. |

---

## 3. Directional SHAP Insights (Summary Beeswarm Analysis)

Analysis of `plots/02_shap_summary_beeswarm_top20.png` confirms domain validity:
1. **Defect Severity**: High values (`CRITICAL`/`HIGH`, red) push SHAP values strongly positive ($+1.0$ to $+1.8$), whereas low severity (`LOW`/`MEDIUM`, blue) pulls SHAP strongly negative ($-1.2$ to $-2.2$).
2. **Asset Criticality**: High asset criticality (red) consistently elevates risk; low/medium criticality (blue) acts as a powerful dampener, ensuring minor sidings do not receive emergency block allocations.
3. **Operational Impact**: High impact (red) yields large positive SHAP values, reflecting the imperative to prevent cascading mainline delays.
4. **Traffic & Overdue Days**: Increasing traffic density and overdue days monotonically elevate risk score.

---

## 4. Local Explanations for Railway Section Engineers

Below are three representative archetypal cases converted into actionable operational formats:

### Case 1: High-Risk Urgent Task (Priority: CRITICAL)
- **Task ID**: `TSK-IR-000367` | **Asset ID**: `AST-SNT-03365`
- **Department**: `S&T` | **Asset Type**: `Digital Axle Counter (MSDAC/SSDAC)`
- **Specific Defect**: `Evaluator Card Processing Fault`
- **Model Risk Score**: **0.91 (90.63%)** $\rightarrow$ **Priority: `CRITICAL`**
- **Decision Recommendation**: **Immediate Priority Block Required (< 24–48 Hours)**.

```
Main Contributing Factors:
[+] defect_severity = CRITICAL                     (SHAP: +1.097) -> Acute electronic card processing trip
[+] asset_criticality = CRITICAL                   (SHAP: +0.934) -> Mainline fail-safe detection asset
[+] operational_impact = HIGH                      (SHAP: +0.375) -> High cascade signal delay potential
[+] failures_last_12_months = 5                    (SHAP: +0.266) -> Unstable recurrent failure history
[+] previous_failures = 5                          (SHAP: +0.128) -> High historical breakdown count
[+] num_open_defects = 3                           (SHAP: +0.077) -> Multiple open defects on section
[-] trains_per_day = 41                            (SHAP: -0.234) -> Moderate branch traffic dampens impact
[-] days_overdue = 0                               (SHAP: -0.075) -> Within scheduled maintenance cycle
```

---

### Case 2: Medium-Risk Transition Task (Priority: MEDIUM)
- **Task ID**: `TSK-IR-000475` | **Asset ID**: `AST-TRD-01879`
- **Department**: `Traction` | **Asset Type**: `Traction Power Transformer (25kV/21.6MVA)`
- **Specific Defect**: `Dissolved Gas Analysis (DGA) Acetylene Surge`
- **Model Risk Score**: **0.48 (48.10%)** $\rightarrow$ **Priority: `MEDIUM`**
- **Decision Recommendation**: **Schedule during Next Standard Power Block (within 7–14 days)**.

```
Main Contributing Factors:
[-] operational_impact = MEDIUM                    (SHAP: -1.123) -> Alternate substation feed available
[+] defect_severity = CRITICAL                     (SHAP: +1.004) -> Acetylene gas surge indicates thermal arcing
[+] asset_criticality = HIGH                       (SHAP: +0.321) -> Primary substation power transformer
[-] num_open_defects = 1                           (SHAP: -0.281) -> Isolated single defect
[+] failures_last_12_months = 3                    (SHAP: +0.205) -> Elevated recent trip frequency
[-] trains_per_day = 34                            (SHAP: -0.152) -> Moderate feeder line traffic
[+] previous_failures = 5                          (SHAP: +0.116) -> Aging substation equipment
[+] asset_age_years = 14.7 yrs                     (SHAP: +0.073) -> Mid-life transformer wear
```

---

### Case 3: Low-Risk Routine Task (Priority: LOW)
- **Task ID**: `TSK-IR-000141` | **Asset ID**: `AST-SNT-03363`
- **Department**: `S&T` | **Asset Type**: `Interlocked Level Crossing (LC Gate)`
- **Specific Defect**: `Boom Locking Plunger Misaligned`
- **Model Risk Score**: **0.06 (5.88%)** $\rightarrow$ **Priority: `LOW`**
- **Decision Recommendation**: **Routine Bi-Weekly / Monthly Maintenance Schedule**.

```
Main Contributing Factors:
[-] asset_criticality = MEDIUM                     (SHAP: -2.310) -> Secondary rural road crossing
[-] operational_impact = LOW                       (SHAP: -0.999) -> Minimal mainline speed restriction impact
[+] defect_severity = CRITICAL                     (SHAP: +0.795) -> Mechanical boom locking misalignment
[-] failure_rate_indicator = 0.00/yr               (SHAP: -0.163) -> Zero recent breakdowns in last 12 months
[+] num_open_defects = 3                           (SHAP: +0.163) -> Minor linkage wear
[-] trains_per_day = 70                            (SHAP: -0.128) -> Moderate traffic density
[+] freight_traffic_ratio = 52.9%                  (SHAP: +0.082) -> Heavy road/rail freight movements
```

---

## 5. Summary of Saved Artifacts

All SHAP analysis code, data, and charts are saved in `ml/models/explainability/`:
1. **Explainability Pipeline Script**: [models/explainability/shap_analysis.py](file:///c:/Users/acous/Desktop/sih2026/ml/models/explainability/shap_analysis.py)
2. **Global Feature Importance CSV**: [models/explainability/shap_feature_importance.csv](file:///c:/Users/acous/Desktop/sih2026/ml/models/explainability/shap_feature_importance.csv)
3. **Local Explanations JSON**: [models/explainability/example_explanations.json](file:///c:/Users/acous/Desktop/sih2026/ml/models/explainability/example_explanations.json)
4. **Visualizations Generated** (in [models/explainability/plots/](file:///c:/Users/acous/Desktop/sih2026/ml/models/explainability/plots/)):
   - `01_shap_global_bar_top20.png` (Top 20 global feature importance bar chart)
   - `02_shap_summary_beeswarm_top20.png` (Top 20 directional beeswarm plot)
   - `03_shap_local_waterfall_high_risk.png` (Local waterfall plot for high-risk task)
   - `04_shap_local_waterfall_medium_risk.png` (Local waterfall plot for medium-risk task)
   - `05_shap_local_waterfall_low_risk.png` (Local waterfall plot for low-risk task)
