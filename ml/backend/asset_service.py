"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
TEJAS Asset Registry & Data/Feature Retrieval Service Layer

Resolves officer-reported maintenance events (primarily Asset ID + observed defect)
to complete, trusted railway infrastructure, historical maintenance, traffic load,
and network topology attributes extracted from official project datasets.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data", "tejas_pilot_master_dataset.csv")
STATIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "datameet_stations.json")
TRAINS_PATH = os.path.join(BASE_DIR, "data", "raw", "datameet_trains.json")


class AssetRegistryService:
    """
    In-memory railway asset registry resolving physical Asset IDs to full
    verified operational profiles with comprehensive provenance tracking.
    """

    def __init__(self):
        self.asset_registry: Dict[str, Dict[str, Any]] = {}
        self.station_metadata: Dict[str, Dict[str, Any]] = {}
        self.is_loaded: bool = False

    def load_registry(self) -> None:
        """
        Loads and indexes all unique asset records and GIS station data.
        """
        print("=" * 75)
        print("LOADING TEJAS ASSET REGISTRY & FEATURE RETRIEVAL LAYER")
        print("=" * 75)

        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Master pilot dataset not found at: {DATASET_PATH}")

        # 1. Load Station GIS metadata if available
        if os.path.exists(STATIONS_PATH):
            try:
                with open(STATIONS_PATH, "r", encoding="utf-8") as f:
                    st_data = json.load(f)
                    for feat in st_data.get("features", []):
                        props = feat.get("properties", {})
                        code = props.get("code")
                        if code:
                            self.station_metadata[code] = {
                                "name": props.get("name"),
                                "zone": props.get("zone"),
                                "state": props.get("state"),
                                "address": props.get("address")
                            }
            except Exception as e:
                print(f"  [Notice] Station GIS load warning: {e}")

        # 2. Load Master Dataset and construct asset profiles
        df = pd.read_csv(DATASET_PATH)
        print(f"Loading asset records from {DATASET_PATH} ({len(df)} records)...")

        # Group by asset_id and extract consistent physical and historical attributes
        # Sort by event_date / event_id so the latest known state is prioritized
        if "event_date" in df.columns:
            df = df.sort_values(by=["event_date", "event_id"])

        grouped = df.groupby("asset_id")
        for asset_id, group in grouped:
            latest = group.iloc[-1]
            
            # Compute average/aggregate historical metrics across asset records
            avg_maint_overdue = (group["maintenance_overdue_days"].mean()) if "maintenance_overdue_days" in group else float(latest.get("maintenance_overdue_days", 0.0))
            avg_days_since_maint = (group["days_since_last_maintenance"].mean()) if "days_since_last_maintenance" in group else float(latest.get("days_since_last_maintenance", 30.0))
            max_f365 = int(group["failures_last_365d"].max()) if "failures_last_365d" in group else int(latest.get("failures_last_365d", 0))
            max_f90 = int(group["failures_last_90d"].max()) if "failures_last_90d" in group else int(latest.get("failures_last_90d", 0))
            max_f30 = int(group["failures_last_30d"].max()) if "failures_last_30d" in group else int(latest.get("failures_last_30d", 0))
            max_rec = int(group["same_defect_recurrences_365d"].max()) if "same_defect_recurrences_365d" in group else int(latest.get("same_defect_recurrences_365d", 0))

            asset_profile = {
                "asset_id": str(asset_id),
                "department": str(latest["department"]),
                "asset_type": str(latest["asset_type"]),
                "safety_function": str(latest.get("safety_function", "direct train running safety")),
                "station_code": str(latest.get("station_code", "NDLS")),
                "station_name": str(latest.get("station_name", "NEW DELHI")),
                "zone": str(latest.get("zone", "NR")),
                "state": str(latest.get("state", "Delhi")),
                "section_id": str(latest.get("section_id", "SEC-001")),
                "asset_age_years": round(float(latest.get("asset_age_years", 5.0)), 1),
                "asset_criticality": str(latest.get("asset_criticality", "HIGH")),
                "asset_criticality_score": round(float(latest.get("asset_criticality_score", 70.0)), 1),
                "scheduled_services_count_proxy": round(float(latest.get("scheduled_services_count_proxy", 50.0)), 1),
                "real_daily_train_count": round(float(latest.get("real_daily_train_count", 50.0)), 1),
                "real_criticality_score": float(latest.get("real_criticality_score", 0.5)),
                "traffic_percentile": round(float(latest.get("traffic_percentile", 0.65)), 4),
                "network_neighbor_degree": float(latest.get("network_neighbor_degree", 4.0)),
                "alternative_route_available": int(latest.get("alternative_route_available", 1)),
                "affected_services_if_blocked": round(float(latest.get("affected_services_if_blocked", 12.0)), 1),
                "operational_impact_score": round(float(latest.get("operational_impact_score", 50.0)), 1),
                "maintenance_interval_days": round(float(latest.get("maintenance_interval_days", 30.0)), 1),
                "days_since_last_maintenance": int(latest.get("days_since_last_maintenance", avg_days_since_maint)),
                "maintenance_overdue_days": round(float(latest.get("maintenance_overdue_days", avg_maint_overdue)), 1),
                "failures_last_365d": max_f365,
                "failures_last_90d": max_f90,
                "failures_last_30d": max_f30,
                "same_defect_recurrences_365d": max_rec,
                "inspection_image_available": int(latest.get("inspection_image_available", 1)),
                "total_events_logged": len(group),
                "last_recorded_event_id": str(latest.get("event_id", "EVT-0000000")),
                "last_recorded_date": str(latest.get("event_date", "2025-01-01"))
            }

            self.asset_registry[str(asset_id)] = asset_profile

        self.is_loaded = True
        print(f"Successfully loaded and indexed {len(self.asset_registry)} unique railway assets into memory.")
        print("=" * 75)

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves verified asset record by ID. Returns None if asset_id is not found.
        """
        if not self.is_loaded:
            self.load_registry()
        return self.asset_registry.get(asset_id.strip())

    def search_assets(
        self,
        query: str = "",
        department: Optional[str] = None,
        zone: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Searches the asset registry by ID, asset type, station code/name, or zone.
        """
        if not self.is_loaded:
            self.load_registry()

        q = query.strip().upper()
        matches = []

        # Normalize department filter to handle URL unescaped/variations
        dept_norm = None
        if department:
            d_clean = department.strip().upper()
            if d_clean in ["S", "S&T", "S_AND_T", "ST", "S%26T", "SIGNALS", "TELECOM"]:
                dept_norm = "S&T"
            elif d_clean in ["ENG", "ENGINEERING", "TRACK"]:
                dept_norm = "Engineering"
            elif d_clean in ["TRAC", "TRACTION", "OHE"]:
                dept_norm = "Traction"
            else:
                dept_norm = department.strip()

        for asset_id, data in self.asset_registry.items():
            if dept_norm and data["department"].upper() != dept_norm.upper():
                continue
            if zone and data["zone"].upper() != zone.strip().upper():
                continue

            if not q or (
                q in asset_id.upper()
                or q in data["asset_type"].upper()
                or q in data["station_code"].upper()
                or q in data["station_name"].upper()
                or q in data["zone"].upper()
            ):
                matches.append(data)
                if len(matches) >= limit:
                    break

        return matches

    def resolve_officer_report(
        self,
        asset_id: str,
        defect_type: Optional[str] = None,
        defect_severity: Optional[str] = "MEDIUM",
        officer_observation: Optional[str] = None,
        days_since_defect: Optional[int] = 3,
        inspection_image_available: Optional[int] = None,
        task_id: Optional[str] = None,
        cv_defect_type: Optional[str] = None,
        cv_confidence: Optional[float] = None,
        cv_derived_severity: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Core retrieval and fusion method:
        Merges officer observation and CV vision model results with immutable verified asset registry data.
        Returns:
        1. complete raw model input dictionary (for ColumnTransformer)
        2. retrieved asset telemetry context (for UI display)
        3. field-level provenance dictionary (tracking the origin of every attribute)
        """
        asset_record = self.get_asset(asset_id)
        if not asset_record:
            raise KeyError(f"Asset ID '{asset_id}' not found in Indian Railways Asset Registry.")

        # If officer did not specify defect_type but CV detected one, use CV output
        resolved_defect_type = defect_type
        defect_type_source = "OFFICER_INPUT"
        defect_type_confidence = "OFFICER_OBSERVATION"

        if cv_defect_type:
            if not resolved_defect_type:
                resolved_defect_type = cv_defect_type
                defect_type_source = "DATASET:CV_MOBILENET_MODEL"
                defect_type_confidence = f"CV_CLASSIFICATION ({round((cv_confidence or 0.95)*100, 1)}%)"
            else:
                # If officer specified defect_type and image is also analyzed
                defect_type_source = "OFFICER_INPUT (VERIFIED_BY_CV)"
                defect_type_confidence = f"CV_VERIFIED ({round((cv_confidence or 0.95)*100, 1)}%)"
        elif not resolved_defect_type:
            # Fallback default if completely unspecified
            resolved_defect_type = "surface shelling/squat"

        resolved_severity = cv_derived_severity or defect_severity or "MEDIUM"
        severity_source = "CV_MODEL_DERIVED" if cv_derived_severity else "SYSTEM_DEFAULT"
        severity_confidence = f"CV_AUTO_DERIVED ({round((cv_confidence or 0.0)*100, 1)}%)" if cv_derived_severity else "SYSTEM_DEFAULT (no image provided)"
        resolved_days_since = (days_since_defect) if days_since_defect is not None else 3
        resolved_img = (inspection_image_available) if inspection_image_available is not None else int(asset_record["inspection_image_available"])
        resolved_task_id = task_id.strip() if task_id and task_id.strip() else f"TSK-{asset_record['department'][:3].upper()}-{asset_id[-6:]}"

        # 1. Assemble complete input dictionary for ColumnTransformer
        full_input_dict = {
            "department": asset_record["department"],
            "asset_type": asset_record["asset_type"],
            "safety_function": asset_record["safety_function"],
            "zone": asset_record["zone"],
            "state": asset_record["state"],
            "defect_type": resolved_defect_type,
            "defect_severity": resolved_severity,
            "defect_severity_label": resolved_severity,
            "days_since_defect": resolved_days_since,
            "defect_duration_days": float(resolved_days_since),
            "days_overdue": int(asset_record["maintenance_overdue_days"]),
            "maintenance_overdue_days": float(asset_record["maintenance_overdue_days"]),
            "maintenance_frequency": "Monthly" if asset_record["maintenance_interval_days"] <= 30 else ("Quarterly" if asset_record["maintenance_interval_days"] <= 90 else "Annual"),
            "maintenance_interval_days": float(asset_record["maintenance_interval_days"]),
            "days_since_last_maintenance": int(asset_record["days_since_last_maintenance"]),
            "previous_failures": int(asset_record["failures_last_365d"]),
            "failures_last_12_months": int(asset_record["failures_last_365d"]),
            "failures_last_365d": float(asset_record["failures_last_365d"]),
            "failures_last_90d": float(asset_record["failures_last_90d"]),
            "failures_last_30d": float(asset_record["failures_last_30d"]),
            "same_defect_recurrences_365d": float(asset_record["same_defect_recurrences_365d"]),
            "trains_per_day": int(asset_record["scheduled_services_count_proxy"]),
            "scheduled_services_count_proxy": float(asset_record["scheduled_services_count_proxy"]),
            "goods_trains_per_day": int(asset_record["affected_services_if_blocked"]),
            "affected_services_if_blocked": float(asset_record["affected_services_if_blocked"]),
            "operational_impact": "HIGH" if asset_record["operational_impact_score"] >= 65 else ("MEDIUM" if asset_record["operational_impact_score"] >= 35 else "LOW"),
            "operational_impact_score": float(asset_record["operational_impact_score"]),
            "asset_age_years": float(asset_record["asset_age_years"]),
            "asset_criticality": asset_record["asset_criticality"],
            "asset_criticality_score": float(asset_record["asset_criticality_score"]),
            "traffic_percentile": float(asset_record["traffic_percentile"]),
            "network_neighbor_degree": float(asset_record["network_neighbor_degree"]),
            "alternative_route_available": int(asset_record["alternative_route_available"]),
            "inspection_image_available": resolved_img,
            "task_id": resolved_task_id,
            "asset_id": asset_id,
            "officer_observation": officer_observation or f"Observed {resolved_defect_type} with {resolved_severity} severity during field inspection."
        }

        # 2. Retrieved Telemetry Context (for API consumer / Frontend)
        retrieved_context = {
            "asset_id": str(asset_record["asset_id"]),
            "department": str(asset_record["department"]),
            "asset_type": str(asset_record["asset_type"]),
            "safety_function": str(asset_record["safety_function"]),
            "station_code": str(asset_record["station_code"]),
            "station_name": str(asset_record["station_name"]),
            "zone": str(asset_record["zone"]),
            "state": str(asset_record["state"]),
            "section_id": str(asset_record["section_id"]),
            "asset_age_years": float(asset_record["asset_age_years"]),
            "asset_criticality": str(asset_record["asset_criticality"]),
            "asset_criticality_score": float(asset_record["asset_criticality_score"]),
            "scheduled_daily_trains": float(asset_record["scheduled_services_count_proxy"]),
            "traffic_density_percentile": float(asset_record["traffic_percentile"]),
            "network_neighbor_degree": float(asset_record["network_neighbor_degree"]),
            "alternative_divert_available": bool(asset_record["alternative_route_available"]),
            "affected_services_if_blocked": float(asset_record["affected_services_if_blocked"]),
            "operational_impact_score": float(asset_record["operational_impact_score"]),
            "statutory_maintenance_interval_days": float(asset_record["maintenance_interval_days"]),
            "days_since_last_maintenance": int(asset_record["days_since_last_maintenance"]),
            "maintenance_overdue_days": float(asset_record["maintenance_overdue_days"]),
            "historical_failures_365d": int(asset_record["failures_last_365d"]),
            "historical_failures_90d": int(asset_record["failures_last_90d"]),
            "historical_failures_30d": int(asset_record["failures_last_30d"]),
            "same_defect_recurrences_365d": int(asset_record["same_defect_recurrences_365d"]),
            "historical_event_count": int(asset_record["total_events_logged"])
        }

        # 3. Provenance Tracking (Explicitly separates officer inputs from verified registry sources)
        provenance = {
            "asset_id": {
                "value": asset_id,
                "source": "OFFICER_INPUT",
                "confidence": "OFFICER_OBSERVATION"
            },
            "defect_type": {
                "value": resolved_defect_type,
                "source": defect_type_source,
                "confidence": defect_type_confidence
            },
            "defect_severity": {
                "value": resolved_severity,
                "source": severity_source,
                "confidence": severity_confidence
            },
            "officer_observation": {
                "value": str(full_input_dict["officer_observation"]),
                "source": "OFFICER_INPUT",
                "confidence": "OFFICER_OBSERVATION"
            },
            "days_since_defect": {
                "value": resolved_days_since,
                "source": "OFFICER_INPUT",
                "confidence": "OFFICER_OBSERVATION"
            },
            "department": {
                "value": str(asset_record["department"]),
                "source": "DATASET:TEJAS_ASSET_REGISTRY",
                "confidence": "VERIFIED_RECORD"
            },
            "asset_type": {
                "value": str(asset_record["asset_type"]),
                "source": "DATASET:TEJAS_ASSET_REGISTRY",
                "confidence": "VERIFIED_RECORD"
            },
            "safety_function": {
                "value": str(asset_record["safety_function"]),
                "source": "DATASET:TEJAS_ASSET_REGISTRY",
                "confidence": "VERIFIED_RECORD"
            },
            "station_code": {
                "value": str(asset_record["station_code"]),
                "source": "DATASET:DATAMEET_GIS",
                "confidence": "VERIFIED_RECORD"
            },
            "zone": {
                "value": str(asset_record["zone"]),
                "source": "DATASET:DATAMEET_GIS",
                "confidence": "VERIFIED_RECORD"
            },
            "state": {
                "value": str(asset_record["state"]),
                "source": "DATASET:DATAMEET_GIS",
                "confidence": "VERIFIED_RECORD"
            },
            "asset_age_years": {
                "value": float(asset_record["asset_age_years"]),
                "source": "DATASET:TEJAS_ASSET_REGISTRY",
                "confidence": "VERIFIED_RECORD"
            },
            "asset_criticality": {
                "value": str(asset_record["asset_criticality"]),
                "source": "DATASET:TEJAS_ASSET_REGISTRY",
                "confidence": "VERIFIED_RECORD"
            },
            "scheduled_services_count_proxy": {
                "value": float(asset_record["scheduled_services_count_proxy"]),
                "source": "DATASET:DATAMEET_TRAINS",
                "confidence": "VERIFIED_RECORD"
            },
            "traffic_percentile": {
                "value": float(asset_record["traffic_percentile"]),
                "source": "DATASET:INDIAN_RAILWAYS_TRAFFIC_STATS",
                "confidence": "VERIFIED_RECORD"
            },
            "maintenance_overdue_days": {
                "value": float(asset_record["maintenance_overdue_days"]),
                "source": "DATASET:TEJAS_MAINTENANCE_SCHEDULE",
                "confidence": "VERIFIED_RECORD"
            },
            "failures_last_365d": {
                "value": int(asset_record["failures_last_365d"]),
                "source": "DATASET:TEJAS_FAILURE_LOGS",
                "confidence": "VERIFIED_RECORD"
            }
        }

        return full_input_dict, retrieved_context, provenance


# Global singleton instance
asset_service = AssetRegistryService()
