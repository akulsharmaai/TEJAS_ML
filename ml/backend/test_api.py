"""
SIH26027: AI-Powered Automatic Block Planning for Indian Railways
FastAPI Backend Integration Test Suite
"""

import os
import sys
import json
import pandas as pd

# Ensure ml root is in pythonpath
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from backend.predictor import predictor

def test_api_suite():
    with TestClient(app) as client:
        print("\n" + "=" * 75)
        print("RUNNING TEJAS FASTAPI BACKEND TEST SUITE")
        print("=" * 75)

        # -----------------------------------------------------
        # 1. Test Root Endpoint (GET /)
        # -----------------------------------------------------
        print("\n[1] Testing GET / ...")
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "running" in resp_root.json()["message"]

        # -----------------------------------------------------
        # 2. Test Health Endpoint (GET /health)
        # -----------------------------------------------------
        print("\n[2] Testing GET /health ...")
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json()["status"] == "ok"

        # -----------------------------------------------------
        # 3. Test Valid Prediction (POST /predict)
        # -----------------------------------------------------
        print("\n[3] Testing POST /predict (Valid Input) ...")
        payload = {
            "defect_severity_label": "CRITICAL",
            "days_overdue": 5,
            "trains_per_day": 55,
            "asset_criticality_score": 75.0,
            "task_id": "TSK-IR-000367",
            "asset_id": "AST-SNT-03365"
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "urgency_score" in data
        assert "task_id" in data
        assert data["task_id"] == "TSK-IR-000367"
        assert 0.0 <= data["urgency_score"] <= 100.0

        # -----------------------------------------------------
        # 4. Test Missing Required Input
        # -----------------------------------------------------
        print("\n[4] Testing POST /predict (Missing Required Input) ...")
        # task_id is required
        bad_payload = {
            "defect_severity_label": "HIGH"
        }
        resp_bad = client.post("/predict", json=bad_payload)
        assert resp_bad.status_code == 422 # Validation Error

        # -----------------------------------------------------
        # 5. Test Invalid Severity
        # -----------------------------------------------------
        print("\n[5] Testing POST /predict (Invalid Severity) ...")
        # invalid severity string
        invalid_sev_payload = {
            "defect_severity_label": "NOT_A_REAL_SEVERITY",
            "task_id": "TSK-IR-000999"
        }
        resp_inv = client.post("/predict", json=invalid_sev_payload)
        assert resp_inv.status_code == 422 # Enum validation should fail


        # -----------------------------------------------------
        # 7. Test Real Sample Prediction from Master Dataset
        # -----------------------------------------------------
        print("\n[7] Testing Real Sample from 10,000-record Master Dataset...")
        data_path = os.path.join(BASE_DIR, "data", "tejas_pilot_master_dataset.csv")
        assert os.path.exists(data_path), f"Master dataset not found at: {data_path}"
        df_sample = pd.read_csv(data_path, nrows=5)
        
        for idx, (_, row) in enumerate(df_sample.iterrows(), start=1):
            row_payload = {
                "defect_severity_label": str(row["defect_severity_label"]),
                "days_overdue": int(row["maintenance_overdue_days"]),
                "trains_per_day": int(row["scheduled_services_count_proxy"]),
                "asset_criticality_score": float(row["asset_criticality_score"]),
                "task_id": str(row["event_id"]),
                "asset_id": str(row["asset_id"])
            }
            resp_sample = client.post("/predict", json=row_payload)
            assert resp_sample.status_code == 200, f"Failed on sample row {idx}: {resp_sample.text}"
            res_json = resp_sample.json()
            assert 0.0 <= res_json['urgency_score'] <= 100.0
            print(f"  Sample {idx} [{row_payload['asset_id']}] -> Urgency Score: {res_json['urgency_score']}")

        print("\n" + "=" * 75)
        print("ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY!")
        print("=" * 75)

if __name__ == "__main__":
    test_api_suite()
