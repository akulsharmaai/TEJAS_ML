"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
FastAPI REST API Application
"""

import os
import sys

# Ensure ml root is in pythonpath
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.predictor import predictor
from backend.schemas import (
    HealthResponse,
    MaintenanceTaskInput,
    PredictionResponse,
    RootResponse,
    OfficerIncidentInput
)
from backend.asset_service import asset_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("railway_api")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing TEJAS ML API...")
    # predictor is initialized as a singleton
    yield
    logger.info("Shutting down Railway Maintenance ML API.")


app = FastAPI(
    title="Indian Railways AI Automatic Block Planning API (TEJAS)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Input Validation Error"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP Error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error"},
    )


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="TEJAS API is running.",
        project="SIH26027",
        system="TEJAS ML Engine",
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        system="TEJAS ML Pipeline",
        version="2.0.0",
    )


@app.get("/assets")
async def search_assets(query: str = "", limit: int = 10):
    return asset_service.search_assets(query=query, limit=limit)


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    asset = asset_service.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@app.post("/predict/officer")
async def predict_officer(report: OfficerIncidentInput):
    try:
        # Resolve the asset to full features
        _, retrieved_context, _ = asset_service.resolve_officer_report(
            asset_id=report.asset_id,
            defect_type=report.defect_type,
            defect_severity=report.defect_severity.value if report.defect_severity else None,
            officer_observation=report.officer_observation,
            days_since_defect=report.days_since_defect,
            inspection_image_available=report.inspection_image_available,
            task_id=report.task_id
        )
        
        # Build the MaintenanceTaskInput using retrieved context
        task_input = MaintenanceTaskInput(
            task_id=report.task_id or f"TSK-{report.asset_id}",
            asset_id=report.asset_id,
            defect_severity_label=report.defect_severity,
            days_overdue=int(report.days_since_defect) if report.days_since_defect is not None else 0,
            trains_per_day=int(retrieved_context.get("scheduled_daily_trains", 30)),
            scheduled_services_count_proxy=float(retrieved_context.get("scheduled_daily_trains", 30.0)),
            real_daily_train_count=float(retrieved_context.get("scheduled_daily_trains", 30.0)),
            is_real_traffic_data=False,
            asset_criticality_score=float(retrieved_context.get("asset_criticality_score", 50.0)),
            failures_last_365d=float(retrieved_context.get("historical_failures_365d", 0.0)),
            days_since_defect=float(report.days_since_defect) if report.days_since_defect else 3.0
        )
        
        # Predict using predictor
        prediction = predictor.predict(task_input)
        
        return {
            "task_id": prediction.task_id,
            "urgency_score": prediction.urgency_score,
            "retrieved_asset_context": retrieved_context
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failure in /predict/officer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )



@app.post("/predict", response_model=PredictionResponse)
async def predict_task_urgency(task: MaintenanceTaskInput) -> PredictionResponse:
    try:
        response = predictor.predict(task)
        return response
    except Exception as e:
        logger.error(f"Inference failure in /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
