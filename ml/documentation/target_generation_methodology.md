# Target Generation Methodology: Multi-Factor Urgency Risk Model

**Project**: SIH26027 – AI-Powered Automatic Block Planning for Indian Railways  
**Target Variable**: `urgent` ($\in \{0, 1\}$)  
**Location in Pipeline**: `ml/data_generation/generate_dataset.py`

---

## 1. Problem Formulation & Design Objectives

In Indian Railways maintenance management, determining whether a task is **URGENT** (requiring immediate priority block clearance within 24–72 hours) is **never a single-variable threshold**. 

For example:
- A `CRITICAL` severity crack on an abandoned siding with 2 trains/week is **NOT** urgent enough to stop mainline express trains.
- A `MEDIUM` severity contact wire wear that is **35 days overdue** on a high-density trunk route with 180 trains/day and 3 recent failures **IS acutely urgent**.

Therefore, the target generation process must satisfy three core criteria:
1. **Multi-Factor Synergy**: Combine severity, asset criticality, overdue duration, traffic intensity, failure history, and defect cluster size.
2. **Non-Trivial Predictability**: Prevent 100% correlation with any single feature (e.g. `defect_severity == 'CRITICAL'`), forcing machine learning models to discover realistic non-linear multivariate interactions.
3. **Realistic Class Imbalance**: Ensure urgent tasks represent approximately 30%–36% of all active maintenance work orders, reflecting real-world priority scheduling backlogs.

---

## 2. Mathematical Formulation

The generation of `urgent` follows a two-stage process:
1. Computation of a **Continuous Latent Urgency Risk Index** ($R_{latent} \in \mathbb{R}$).
2. Conversion to a **Stochastic Bernouilli Target** via a calibrated Sigmoidal Logistic Link Function.

### Step 1: Continuous Latent Risk Index Equation

$$R_{latent} = R_{base} + 0.40 \cdot T_{temporal} + 0.35 \cdot H_{history} + 0.40 \cdot \Lambda_{traffic} + 0.50 \cdot D_{cluster} + \Psi_{synergy} + \epsilon$$

Where the individual constituent terms are defined as:

#### 1. Baseline Severity & Criticality ($R_{base}$)
Let $S_{defect}, C_{asset}, O_{impact} \in \{10.0, 35.0, 70.0, 95.0\}$ correspond to ordinal grades `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`:

$$R_{base} = 0.32 \cdot S_{defect} + 0.22 \cdot C_{asset} + 0.16 \cdot O_{impact}$$

#### 2. Temporal Strain ($T_{temporal}$)
Captures the cumulative penalty of delayed maintenance and unaddressed defect age:

$$T_{temporal} = \min\Big(45.0, \; 0.90 \cdot \text{days\_overdue} + 0.35 \cdot \text{days\_since\_defect}\Big)$$

#### 3. Historical Failure & Aging Strain ($H_{history}$)
Reflects recurrent breakdown risk and physical fatigue:

$$H_{history} = \min\Big(30.0, \; 5.0 \cdot \text{failures\_last\_12\_months} + 1.5 \cdot \text{previous\_failures} + 10.0 \cdot \frac{\text{asset\_age\_years}}{35.0}\Big)$$

#### 4. Traffic Load Intensity ($\Lambda_{traffic}$)
Dynamic stress exerted by high-speed passenger and heavy-axle freight traffic:

$$\Lambda_{traffic} = 12.0 \cdot \left(\frac{\text{trains\_per\_day}}{250.0}\right) + 8.0 \cdot \left(\frac{\text{goods\_trains\_per\_day}}{120.0}\right)$$

#### 5. Open Defect Cluster Strain ($D_{cluster}$)
Penalty for multiple concurrent unresolved defects on the same asset:

$$D_{cluster} = (\text{num\_open\_defects} - 1) \cdot 4.0$$

#### 6. Non-Linear Synergy Term ($\Psi_{synergy}$)
Represents acute engineering danger when high severity co-occurs with high criticality on busy corridors, as well as de-escalation for minor defects on low-risk lines:

$$\Psi_{synergy} = \begin{cases}
+24.0, & \text{if } S_{defect} \ge 70 \land C_{asset} \ge 70 \land O_{impact} \ge 70 \\
+14.0, & \text{if } S_{defect} \ge 70 \land C_{asset} \ge 70 \land O_{impact} < 70 \\
-12.0, & \text{if } S_{defect} \le 10 \land C_{asset} \le 35 \land \text{days\_overdue} < 10 \\
0.0, & \text{otherwise}
\end{cases}$$

#### 7. Stochastic Unobserved Variance ($\epsilon$)
Models unmeasured field conditions (e.g. ambient monsoon rain, qualitative inspector commentary, local track curvature):

$$\epsilon \sim \mathcal{N}(\mu=0, \, \sigma^2 = 6.5^2)$$

---

### Step 2: Probabilistic Target Mapping

To map the continuous latent risk score $R_{latent}$ into the binary ground-truth label `urgent` $\in \{0, 1\}$, we apply a calibrated logistic sigmoid function:

$$z = \frac{R_{latent} - \tau}{\theta}$$

Where:
* **Threshold ($\tau$)**: `82.0` (calibrated to anchor the decision boundary)
* **Temperature parameter ($\theta$)**: `7.5` (controls the smoothness of transition across borderline cases)

The conditional probability of urgency is:

$$P(\text{urgent} = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The final binary assignment is sampled stochastically from the Bernoulli distribution:

$$\text{urgent} \sim \text{Bernoulli}\Big(P(\text{urgent} = 1 \mid \mathbf{x})\Big)$$

---

## 3. Empirical Target Properties in the Dataset

In the generated 40,000 records:
* **Class 0 (Routine / Non-Urgent)**: 25,787 records (**64.47%**)
* **Class 1 (Urgent Block Required)**: 14,213 records (**35.53%**)
* **Multi-variable distribution**:
  - `CRITICAL` severity tasks are urgent ~72% of the time (not 100%, because an un-overdue critical defect on a low-traffic branch line may have a short scheduled buffer).
  - `MEDIUM` severity tasks are urgent ~22% of the time (when high overdue days and heavy traffic compound the risk).
  - `LOW` severity tasks are urgent ~3% of the time (when severely overdue with multiple open defects on critical infrastructure).

This guarantees that future ML models (Random Forests, Gradient Boosting, Multi-Layer Perceptrons) will learn rich, robust decision boundaries.
