"""
Unit and Integration tests for WebSocket Telemetry & Emergency Broadcast Engine.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.app.routers.telemetry import ConnectionManager

client = TestClient(app)

def test_telemetry_websocket_connection():
    """
    Test WebSocket connection handshake and welcome payload.
    """
    with client.websocket_connect("/ws/telemetry") as websocket:
        data = websocket.receive_json()
        assert data["event_type"] == "CONNECTED"
        assert "Connected to TEJAS" in data["message"]


def test_broadcast_endpoint():
    """
    Test /broadcast/telemetry HTTP endpoint.
    """
    payload = {
        "event_type": "DEFECT_REPORTED",
        "id": "EMG-109",
        "title": "USFD Rail Flaw & Weld Fracture",
        "urgencyScore": 96.7,
        "type": "EMERGENCY"
    }
    response = client.post("/broadcast/telemetry", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "broadcasted"
    assert response.json()["payload_type"] == "DEFECT_REPORTED"
