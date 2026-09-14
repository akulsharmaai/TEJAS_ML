from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import enum

from backend.app.database import Base


class CautionOrderStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class Section(Base):
    __tablename__ = "sections"

    section_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    section_code = Column(String, unique=True, index=True, nullable=False)
    section_name = Column(String, nullable=False)
    start_station = Column(String, nullable=False)
    end_station = Column(String, nullable=False)
    total_length_km = Column(Float, nullable=False, default=100.0)

    caution_orders = relationship("CautionOrder", back_populates="section")


class CautionOrder(Base):
    __tablename__ = "caution_orders"

    order_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_number = Column(String, unique=True, index=True, nullable=False)
    section_id = Column(Integer, ForeignKey("sections.section_id"), nullable=False)
    km_from = Column(Float, nullable=False)
    km_to = Column(Float, nullable=False)
    max_speed_kmh = Column(Integer, nullable=False)
    reason = Column(String, nullable=False)
    issued_by_officer = Column(String, nullable=False)
    status = Column(String, default=CautionOrderStatus.ACTIVE.value, nullable=False)
    issued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    section = relationship("Section", back_populates="caution_orders")
