# TEJAS Urgency ML Module - Integration Contract

This document outlines the frozen API contract for the TEJAS Maintenance Task Urgency ML Module.

## WHAT I PROVIDE
This module provides a single endpoint that generates a calibrated Urgency Score (0-100) for a given maintenance task. 
A higher score indicates a more urgent task that should be prioritized for line block allocation.

The output will ONLY contain:
- `task_id`
- `urgency_score`

*Note: All legacy prediction targets (failure risk, maintenance duration, block requirement, priority classes, and CV labels) have been permanently removed from the payload.*

## HOW TO CALL
Make a `POST` request to the `/predict` endpoint:

**Endpoint:** `POST /predict`
**Content-Type:** `application/json`

### Example Request Body
```json
{
  "task_id": "TSK-IR-000367",
  "defect_severity_label": "CRITICAL",
  "days_overdue": 5,
  "trains_per_day": 55,
  "asset_criticality_score": 75.0,
  "failures_last_365d": 0.0,
  "is_real_traffic_data": true
}
```

### Input Fields Dictionary
- `task_id` **(Required, string)**: Unique identifier for the maintenance task.
- `defect_severity_label` *(Optional, string)*: Expected values: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`. Defaults to `"MEDIUM"`.
- `days_overdue` *(Optional, int)*: Number of days the maintenance is overdue. Defaults to `0`.
- `trains_per_day` *(Optional, int)*: The scheduled trains per day. Defaults to `30`.
- `asset_criticality_score` *(Optional, float)*: Criticality score (0-100) of the physical asset. Defaults to `50.0`.
- `failures_last_365d` *(Optional, float)*: Number of failures for this asset in the past year. Defaults to `0.0`.
- `is_real_traffic_data` *(Optional, bool)*: Indicates if the traffic volume is real or proxy data. Defaults to `true`.

## EXACT API OUTPUT CONTRACT
The response is guaranteed to be a JSON object containing exactly the task ID and a float `urgency_score` between 0.0 and 100.0.

### Example Response Body
```json
{
  "task_id": "TSK-IR-000367",
  "urgency_score": 68.32
}
```

## WHAT THE SCORE MEANS
The `urgency_score` operates on a 0-100 scale:
- **0-100 scale**: Higher values equal greater operational urgency.
- **Monotonicity**: The score strictly increases as severity, traffic, overdue days, and asset criticality increase.
- **Usage**: Consume this single value directly into your CP-SAT optimization model to rank and select tasks for line block planning.
