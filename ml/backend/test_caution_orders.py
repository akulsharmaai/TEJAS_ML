import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)

def test_caution_orders_flow():
    # 1. Fetch active caution orders (should seed default sections)
    response = client.get("/caution-orders/active")
    assert response.status_code == 200
    initial_orders = response.json()

    # 2. Fetch sections
    response = client.get("/caution-orders/sections")
    assert response.status_code == 200
    sections = response.json()
    assert len(sections) > 0
    sec_id = sections[0]["section_id"]

    # 3. Issue a new Caution Order
    issue_payload = {
        "section_id": sec_id,
        "km_from": 120.5,
        "km_to": 122.0,
        "max_speed_kmh": 30,
        "reason": "Test Transverse Rail Crack",
        "issued_by_officer": "Test Officer"
    }
    response = client.post("/caution-orders/issue", json=issue_payload)
    assert response.status_code == 201
    created_order = response.json()
    assert created_order["status"] == "ACTIVE"
    assert created_order["max_speed_kmh"] == 30
    order_id = created_order["order_id"]

    # 4. Fetch Form T/409 PDF
    response = client.get(f"/caution-orders/{order_id}/pdf")
    assert response.status_code == 200

    # 5. Revoke Caution Order
    response = client.patch(f"/caution-orders/{order_id}/revoke")
    assert response.status_code == 200
    revoked_order = response.json()
    assert revoked_order["status"] == "REVOKED"
    assert revoked_order["revoked_at"] is not None
    print("ALL CAUTION ORDER API ENDPOINTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_caution_orders_flow()

