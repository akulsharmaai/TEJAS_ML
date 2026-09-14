"""
Re-export backend.app.routers.telemetry for import compatibility.
"""
from backend.app.routers.telemetry import router, manager, ConnectionManager

__all__ = ["router", "manager", "ConnectionManager"]
