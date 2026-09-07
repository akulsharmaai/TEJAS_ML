import os
import math
import logging
import joblib
import pandas as pd
from backend.schemas import MaintenanceTaskInput, PredictionResponse

logger = logging.getLogger("railway_api.predictor")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "tuned_models", "urgency_model_pipeline.joblib")

class Predictor:
    """
    Implements the TEJAS Urgency Engine.
    Transforms raw incoming maintenance task requests into a single Urgency Score
    using the approved HistGBM machine learning pipeline.
    
    The deterministic mathematical formula is preserved as a reference target generator.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Predictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            self.pipeline = joblib.load(PIPELINE_PATH)
            logger.info(f"Successfully loaded TEJAS Urgency ML pipeline from {PIPELINE_PATH}")
        except Exception as e:
            logger.error(f"Failed to load TEJAS Urgency pipeline: {e}")
            self.pipeline = None

    def calculate_deterministic_reference(self, task: MaintenanceTaskInput) -> float:
        """
        The frozen mathematical deterministic formula, preserved as a reference target generator.
        This represents the ground-truth function the ML pipeline attempts to learn.
        """
        severity = task.defect_severity_label
        if not severity:
            raise ValueError("Severity is missing or invalid")
            
        severity_value = severity.value if hasattr(severity, 'value') else severity
        
        severity_bands = {
            "LOW": (10, 58),
            "MEDIUM": (30, 70),
            "HIGH": (55, 90),
            "CRITICAL": (78, 100)
        }
        
        if severity_value not in severity_bands:
            raise ValueError(f"Invalid severity: {severity_value}")
            
        base, ceiling = severity_bands[severity_value]
        
        # 1. OVERDUE
        overdue = getattr(task, 'days_overdue', 0)
        overdue = max(0.0, float(overdue) if overdue is not None else 0.0)
        overdue_norm = 1 - math.exp(-overdue / 30)
        overdue_norm = max(0.0, min(1.0, overdue_norm))
        
        # 2. TRAFFIC
        is_real = getattr(task, 'is_real_traffic_data', True)
        real_traffic = getattr(task, 'real_daily_train_count', None)
        proxy_traffic = getattr(task, 'scheduled_services_count_proxy', None)
        
        traffic_volume = None
        if is_real:
            if real_traffic is not None:
                traffic_volume = float(real_traffic)
            else:
                if proxy_traffic is not None:
                    traffic_volume = float(proxy_traffic)
                else:
                    raise ValueError("Traffic coverage is missing (both real and proxy are null)")
        else:
            if proxy_traffic is not None:
                traffic_volume = float(proxy_traffic)
            elif real_traffic is not None:
                traffic_volume = float(real_traffic)
            else:
                raise ValueError("Traffic coverage is missing")
                
        traffic_volume = max(0.0, float(traffic_volume))
        traffic_norm = 1 - math.exp(-traffic_volume / 40)
        traffic_norm = max(0.0, min(1.0, traffic_norm))
        
        # 3. ASSET CRITICALITY
        criticality = getattr(task, 'asset_criticality_score', None)
        if criticality is None:
            criticality = 50.0
            
        criticality = max(0.0, min(100.0, float(criticality)))
        criticality_norm = criticality / 100.0
        criticality_norm = max(0.0, min(1.0, criticality_norm))
        
        # 4. FAILURE HISTORY
        failures = getattr(task, 'failures_last_365d', 0)
        failures = max(0.0, float(failures) if failures is not None else 0.0)
        failures_norm = 1 - math.exp(-failures / 2)
        failures_norm = max(0.0, min(1.0, failures_norm))
        
        # CONTEXT FACTOR
        context_factor = (
            0.35 * overdue_norm +
            0.25 * traffic_norm +
            0.20 * criticality_norm +
            0.20 * failures_norm
        )
        context_factor = max(0.0, min(1.0, context_factor))
        
        # FINAL URGENCY
        urgency = base + (ceiling - base) * context_factor
        urgency = max(0.0, min(100.0, urgency))
        
        return urgency

    def predict(self, task: MaintenanceTaskInput) -> PredictionResponse:
        """
        Extracts features, resolves traffic routing, and evaluates the trained ML Pipeline.
        """
        if not self.pipeline:
            raise RuntimeError("TEJAS Urgency ML pipeline is not loaded.")
            
        severity = task.defect_severity_label
        if not severity:
            raise ValueError("Severity is missing or invalid")
            
        severity_value = severity.value if hasattr(severity, 'value') else severity
        
        # Validate severity
        if severity_value not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            raise ValueError(f"Invalid severity: {severity_value}")
        
        # Extract features & Traffic Routing
        overdue = getattr(task, 'days_overdue', 0)
        overdue = max(0.0, float(overdue) if overdue is not None else 0.0)
        
        is_real = getattr(task, 'is_real_traffic_data', True)
        real_traffic = getattr(task, 'real_daily_train_count', None)
        proxy_traffic = getattr(task, 'scheduled_services_count_proxy', None)
        
        traffic_volume = None
        if is_real:
            if real_traffic is not None:
                traffic_volume = float(real_traffic)
            else:
                if proxy_traffic is not None:
                    traffic_volume = float(proxy_traffic)
                    logger.warning("real_daily_train_count is missing, fallback to scheduled_services_count_proxy used.")
                else:
                    raise ValueError("Traffic coverage is missing (both real and proxy are null)")
        else:
            if proxy_traffic is not None:
                traffic_volume = float(proxy_traffic)
            elif real_traffic is not None:
                traffic_volume = float(real_traffic)
                logger.warning("is_real_traffic_data is false but proxy missing, fallback to real_daily_train_count used.")
            else:
                raise ValueError("Traffic coverage is missing")
                
        traffic_volume = max(0.0, float(traffic_volume))
        
        criticality = getattr(task, 'asset_criticality_score', None)
        if criticality is None:
            criticality = 50.0
            logger.info("asset_criticality_score is missing, imputed neutral value 50.")
        criticality = max(0.0, min(100.0, float(criticality)))
        
        failures = getattr(task, 'failures_last_365d', 0)
        failures = max(0.0, float(failures) if failures is not None else 0.0)

        # Create DataFrame matching pipeline requirements
        df_features = pd.DataFrame([{
            'defect_severity_label': severity_value,
            'maintenance_overdue_days': overdue,
            'traffic_volume': traffic_volume,
            'asset_criticality_score': criticality,
            'failures_last_365d': failures
        }])
        
        # Predict using the trained ML pipeline
        urgency_pred = self.pipeline.predict(df_features)[0]
        
        # Final safety clamp
        urgency_score = max(0.0, min(100.0, float(urgency_pred)))
        
        return PredictionResponse(
            task_id=task.task_id,
            urgency_score=round(urgency_score, 1)
        )

# Create singleton
predictor = Predictor()
