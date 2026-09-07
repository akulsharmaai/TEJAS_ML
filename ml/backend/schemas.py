"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
Pydantic Data Schemas & Enums for Multi-Target API Request/Response Validation
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SeverityLevelEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MaintenanceTaskInput(BaseModel):
    defect_severity_label: Optional[SeverityLevelEnum] = None
    days_overdue: Optional[int] = 0
    trains_per_day: Optional[int] = 30
    scheduled_services_count_proxy: Optional[float] = 30.0
    real_daily_train_count: Optional[float] = 30.0
    is_real_traffic_data: Optional[bool] = True
    asset_criticality_score: Optional[float] = 50.0
    failures_last_365d: Optional[float] = 0.0
    days_since_defect: Optional[float] = 3.0
    task_id: str
    asset_id: Optional[str] = None

class OfficerIncidentInput(BaseModel):
    asset_id: str
    defect_type: Optional[str] = None
    defect_severity: Optional[SeverityLevelEnum] = None
    officer_observation: Optional[str] = None
    days_since_defect: Optional[int] = 3
    inspection_image_available: Optional[int] = None
    task_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "defect_severity_label": "CRITICAL",
                "days_overdue": 5,
                "trains_per_day": 55,
                "asset_criticality_score": 75.0,
                "task_id": "TSK-IR-000367",
                "asset_id": "AST-SNT-03365"
            }
        }
    )


class PredictionResponse(BaseModel):
    task_id: str
    urgency_score: float


class HealthResponse(BaseModel):
    status: str = "ok"
    system: str = "TEJAS ML Pipeline"
    version: str = "2.0.0"


class RootResponse(BaseModel):
    message: str
    project: str = "SIH26027"
    system: str = "TEJAS ML Engine"
    docs_url: str = "/docs"
    health_url: str = "/health"
    predict_url: str = "/predict"
