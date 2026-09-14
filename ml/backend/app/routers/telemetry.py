"""
FastAPI Connection Manager and WebSocket Telemetry Router for TEJAS.
Provides sub-second live telemetry streaming and emergency defect broadcasts.
"""

from typing import List, Dict, Any
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("telemetry_websocket")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast JSON payload to all active WebSocket clients.
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()

router = APIRouter(tags=["telemetry"])

@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sensor telemetry and emergency defect alerts.
    Clients connect to ws://<host>:<port>/ws/telemetry
    """
    await manager.connect(websocket)
    try:
        # Send initial handshake / welcome message
        await manager.send_personal_message(
            {
                "event_type": "CONNECTED",
                "message": "Connected to TEJAS Sub-Second Telemetry Stream",
                "status": "ACTIVE"
            },
            websocket
        )
        
        # Keep connection open and listen for incoming client frames if any
        while True:
            data = await websocket.receive_json()
            # If client sends telemetry command or heartbeats
            logger.debug(f"Received WebSocket message from client: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)
