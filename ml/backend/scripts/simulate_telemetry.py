"""
Async IoT Sensor Telemetry & ML Defect Emergency Alert Simulator for TEJAS.
Simulates live feeds from:
- USFD (Ultrasonic Flaw Detection) Track Flaw Cars
- Point Machine voltage microswitches
- OHE (Overhead Equipment) catenary wire tension sensors
"""

import asyncio
import datetime
import json
import random
import logging
from typing import Dict, Any

try:
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulator")

API_BROADCAST_URL = "http://localhost:8000/broadcast/telemetry"
WS_URL = "ws://localhost:8000/ws/telemetry"

DEFECT_TEMPLATES = [
    {
        "id": "EMG-109",
        "title": "USFD Rail Flaw & Weld Fracture",
        "message": "Severe 4.8mm rail fracture detected under dynamic freight load.",
        "type": "EMERGENCY",
        "department": "ENGINEERING",
        "subsystem": "P.Way (USFD Rail Inspection)",
        "urgencyScore": 96.7,
        "sectionCode": "VAR-LKO-SEC1",
        "routeLocation": "Varanasi - Lucknow Mainline (KM 42.4, Track T1)",
        "detailedObservations": "USFD waveform registered 4.8mm transverse crack at thermit weld joint #214.",
        "recommendedAction": "Impose 20 km/h emergency speed restriction (PSR). Dispatch SSE/P.Way immediately."
    },
    {
        "id": "EMG-204",
        "title": "Point Machine Voltage Spike & Overload",
        "message": "Point 102B microswitch voltage spike (38.4V vs 24V normal limit) during crossover setting.",
        "type": "EMERGENCY",
        "department": "SIGNAL & TELECOM",
        "subsystem": "Point Machine & Detection Circuit",
        "urgencyScore": 93.4,
        "sectionCode": "NDLS-CNB-SEC3",
        "routeLocation": "Kanpur Central West Yard (Point 102B)",
        "detailedObservations": "Operating current exceeded 6.2A for >4.5 sec. High friction or mechanical obstruction suspected.",
        "recommendedAction": "Hold route locking signal. Inspect Point 102B switch rail and throw bar immediately."
    },
    {
        "id": "EMG-305",
        "title": "OHE Catenary Wire Drop & High Tension Anomaly",
        "message": "Catenary wire tension dropped to 8.2 kN (Normal: 12.0 kN) with pantograph spark burst.",
        "type": "EMERGENCY",
        "department": "ELECTRICAL (TRD)",
        "subsystem": "OHE Catenary & Dropper Assembly",
        "urgencyScore": 98.1,
        "sectionCode": "HWH-NDLS-SEC7",
        "routeLocation": "Howrah - Dhanbad Mainline (Mast #412/18)",
        "detailedObservations": "ATD (Auto Tensioning Device) counterweight stuck. Dropper #4 snapped under thermal expansion.",
        "recommendedAction": "De-energize OHE Section 412. Issue emergency power block and dispatch TRD tower wagon."
    }
]


def get_formatted_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%d %b %Y, %H:%M:%S IST")


def generate_routine_telemetry() -> Dict[str, Any]:
    return {
        "event_type": "TELEMETRY_METRIC",
        "timestamp": get_formatted_time(),
        "sensors": {
            "usfd_track_flaw_car": {
                "car_id": "TRC-04",
                "speed_kmh": round(random.uniform(70.0, 110.0), 1),
                "vertical_acceleration_g": round(random.uniform(0.12, 0.45), 2),
                "gauge_deviation_mm": round(random.uniform(-1.5, 2.8), 1)
            },
            "point_machine": {
                "point_id": "PT-102B",
                "voltage_v": round(random.uniform(23.5, 25.2), 1),
                "operating_current_a": round(random.uniform(2.1, 3.4), 2),
                "microswitch_status": "NORMAL"
            },
            "ohe_catenary": {
                "mast_id": "MST-412/18",
                "catenary_tension_kn": round(random.uniform(11.8, 12.4), 2),
                "contact_wire_wear_mm": round(random.uniform(0.8, 1.4), 2),
                "temperature_c": round(random.uniform(28.0, 36.5), 1)
            }
        }
    }


def generate_emergency_defect() -> Dict[str, Any]:
    template = random.choice(DEFECT_TEMPLATES).copy()
    template["event_type"] = "DEFECT_REPORTED"
    template["reportedExactTime"] = get_formatted_time()
    # Ensure urgency score > 90%
    template["urgencyScore"] = round(random.uniform(91.5, 99.2), 1)
    return template


async def send_payload_http(payload: Dict[str, Any]):
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            API_BROADCAST_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            logger.info(f"Broadcasted payload ({payload.get('event_type')}) HTTP {resp.status}")
    except Exception as e:
        logger.error(f"HTTP Broadcast failed: {e}. Is FastAPI running on http://localhost:8000?")


async def run_simulation_loop(interval_sec: float = 5.0):
    logger.info("Starting TEJAS Low-Latency Telemetry & Defect Simulator...")
    logger.info(f"Broadcasting to {API_BROADCAST_URL} every {interval_sec} seconds...")
    
    step = 0
    while True:
        try:
            step += 1
            # Every 7th cycle (35 seconds), broadcast a new >90% urgency emergency defect
            if step % 7 == 0:
                defect_payload = generate_emergency_defect()
                logger.warning(
                    f"🚨 [EMERGENCY ALERT BROADCAST] Defect {defect_payload['id']} - "
                    f"Urgency: {defect_payload['urgencyScore']}% - {defect_payload['title']}"
                )
                await send_payload_http(defect_payload)
            else:
                telemetry_payload = generate_routine_telemetry()
                logger.info(f"📡 [LIVE TELEMETRY STREAM] Updated sensor metrics at {telemetry_payload['timestamp']}")
                await send_payload_http(telemetry_payload)

            await asyncio.sleep(interval_sec)



        except asyncio.CancelledError:
            logger.info("Simulation loop stopped.")
            break
        except Exception as e:
            logger.error(f"Simulation iteration error: {e}")
            await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    try:
        asyncio.run(run_simulation_loop())
    except KeyboardInterrupt:
        logger.info("Simulator terminated by user.")
