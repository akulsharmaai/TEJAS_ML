import io
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db, engine
from backend.app.models import Base, CautionOrder, Section, CautionOrderStatus
from backend.app.routers.telemetry import manager as telemetry_manager

# Ensure tables exist
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/caution-orders", tags=["Caution Orders (TSR)"])


# Pydantic Schemas
class CautionOrderCreate(BaseModel):
    section_id: int
    km_from: float = Field(..., description="Start kilometer post, e.g. 412.5")
    km_to: float = Field(..., description="End kilometer post, e.g. 415.0")
    max_speed_kmh: int = Field(..., description="Restricted max speed in km/h (20, 30, 45, 60, 75)")
    reason: str = Field(..., description="Reason for caution order e.g. USFD Transverse Rail Crack")
    issued_by_officer: str = Field(..., description="Officer name/designation issuing order")


class CautionOrderResponse(BaseModel):
    order_id: int
    order_number: str
    section_id: int
    section_code: Optional[str] = None
    section_name: Optional[str] = None
    start_station: Optional[str] = None
    end_station: Optional[str] = None
    km_from: float
    km_to: float
    max_speed_kmh: int
    reason: str
    issued_by_officer: str
    status: str
    issued_at: datetime
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


def seed_default_sections_if_empty(db: Session):
    if db.query(Section).count() == 0:
        default_sections = [
            Section(section_id=1, section_code="NDLS-CNB", section_name="New Delhi - Kanpur Central Main Corridor", start_station="New Delhi (NDLS)", end_station="Kanpur Central (CNB)", total_length_km=440.0),
            Section(section_id=2, section_code="CNB-PRYJ", section_name="Kanpur - Prayagraj Fast Line", start_station="Kanpur Central (CNB)", end_station="Prayagraj Junction (PRYJ)", total_length_km=194.0),
            Section(section_id=3, section_code="PRYJ-DDU", section_name="Prayagraj - Pt. Deen Dayal Upadhyaya Trunk Corridor", start_station="Prayagraj Junction (PRYJ)", end_station="Pt. Deen Dayal Upadhyaya (DDU)", total_length_km=153.0),
            Section(section_id=4, section_code="HWH-DDU", section_name="Howrah - DDU Grand Chord", start_station="Howrah (HWH)", end_station="Pt. Deen Dayal Upadhyaya (DDU)", total_length_km=668.0),
        ]
        db.add_all(default_sections)
        db.commit()


@router.post("/issue", response_model=CautionOrderResponse, status_code=status.HTTP_201_CREATED)
async def issue_caution_order(payload: CautionOrderCreate, db: Session = Depends(get_db)):
    """
    Issue a new Caution Order (Temporary Speed Restriction - TSR) Form T/409.
    Triggers a real-time WebSocket alert over telemetry manager.
    """
    seed_default_sections_if_empty(db)

    section = db.query(Section).filter(Section.section_id == payload.section_id).first()
    if not section:
        raise HTTPException(status_code=404, detail=f"Section ID {payload.section_id} not found.")

    count = db.query(CautionOrder).count() + 1
    order_number = f"TSR-{section.section_code[:3]}-2026-{count:03d}"

    new_order = CautionOrder(
        order_number=order_number,
        section_id=payload.section_id,
        km_from=payload.km_from,
        km_to=payload.km_to,
        max_speed_kmh=payload.max_speed_kmh,
        reason=payload.reason,
        issued_by_officer=payload.issued_by_officer,
        status=CautionOrderStatus.ACTIVE.value,
        issued_at=datetime.utcnow()
    )

    db.add(new_order)
    db.commit()
    db.refresh(new_order)

    # Broadcast WebSocket alert
    ws_event = {
        "event_type": "CAUTION_ORDER_ISSUED",
        "timestamp": new_order.issued_at.isoformat(),
        "data": {
            "order_id": new_order.order_id,
            "order_number": new_order.order_number,
            "section_code": section.section_code,
            "km_range": f"{new_order.km_from} - {new_order.km_to} KM",
            "max_speed_kmh": new_order.max_speed_kmh,
            "reason": new_order.reason,
            "issued_by": new_order.issued_by_officer
        }
    }
    try:
        await telemetry_manager.broadcast(ws_event)
    except Exception:
        pass

    return CautionOrderResponse(
        order_id=new_order.order_id,
        order_number=new_order.order_number,
        section_id=new_order.section_id,
        section_code=section.section_code,
        section_name=section.section_name,
        start_station=section.start_station,
        end_station=section.end_station,
        km_from=new_order.km_from,
        km_to=new_order.km_to,
        max_speed_kmh=new_order.max_speed_kmh,
        reason=new_order.reason,
        issued_by_officer=new_order.issued_by_officer,
        status=new_order.status,
        issued_at=new_order.issued_at,
        revoked_at=new_order.revoked_at
    )


@router.get("/active", response_model=List[CautionOrderResponse])
def get_active_caution_orders(db: Session = Depends(get_db)):
    """
    Fetch all active Caution Orders joined with section details.
    """
    seed_default_sections_if_empty(db)

    results = (
        db.query(CautionOrder, Section)
        .join(Section, CautionOrder.section_id == Section.section_id)
        .filter(CautionOrder.status == CautionOrderStatus.ACTIVE.value)
        .order_by(CautionOrder.issued_at.desc())
        .all()
    )

    response_list = []
    for order, section in results:
        response_list.append(
            CautionOrderResponse(
                order_id=order.order_id,
                order_number=order.order_number,
                section_id=order.section_id,
                section_code=section.section_code,
                section_name=section.section_name,
                start_station=section.start_station,
                end_station=section.end_station,
                km_from=order.km_from,
                km_to=order.km_to,
                max_speed_kmh=order.max_speed_kmh,
                reason=order.reason,
                issued_by_officer=order.issued_by_officer,
                status=order.status,
                issued_at=order.issued_at,
                revoked_at=order.revoked_at
            )
        )
    return response_list


@router.get("/sections")
def get_sections(db: Session = Depends(get_db)):
    """
    Fetch available railway section corridors.
    """
    seed_default_sections_if_empty(db)
    return db.query(Section).all()


@router.patch("/{order_id}/revoke", response_model=CautionOrderResponse)
async def revoke_caution_order(order_id: int, db: Session = Depends(get_db)):
    """
    Revoke an existing active speed restriction order.
    Broadcasts revocation event over WebSocket.
    """
    order = db.query(CautionOrder).filter(CautionOrder.order_id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail=f"Caution order ID {order_id} not found.")

    if order.status == CautionOrderStatus.REVOKED.value:
        raise HTTPException(status_code=400, detail="Caution order is already revoked.")

    order.status = CautionOrderStatus.REVOKED.value
    order.revoked_at = datetime.utcnow()
    db.commit()
    db.refresh(order)

    section = db.query(Section).filter(Section.section_id == order.section_id).first()
    section_code = section.section_code if section else f"SEC-{order.section_id}"

    # Broadcast revocation
    ws_event = {
        "event_type": "CAUTION_ORDER_REVOKED",
        "timestamp": order.revoked_at.isoformat(),
        "data": {
            "order_id": order.order_id,
            "order_number": order.order_number,
            "section_code": section_code,
            "status": "REVOKED"
        }
    }
    try:
        await telemetry_manager.broadcast(ws_event)
    except Exception:
        pass

    return CautionOrderResponse(
        order_id=order.order_id,
        order_number=order.order_number,
        section_id=order.section_id,
        section_code=section_code,
        section_name=section.section_name if section else "",
        start_station=section.start_station if section else "",
        end_station=section.end_station if section else "",
        km_from=order.km_from,
        km_to=order.km_to,
        max_speed_kmh=order.max_speed_kmh,
        reason=order.reason,
        issued_by_officer=order.issued_by_officer,
        status=order.status,
        issued_at=order.issued_at,
        revoked_at=order.revoked_at
    )


@router.get("/{order_id}/pdf")
def generate_caution_order_pdf(order_id: int, db: Session = Depends(get_db)):
    """
    Generates a printable Indian Railways Form T/409 PDF document with verification QR code.
    """
    order = db.query(CautionOrder).filter(CautionOrder.order_id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail=f"Caution Order #{order_id} not found")

    section = db.query(Section).filter(Section.section_id == order.section_id).first()
    section_code = section.section_code if section else "NDLS-CNB"
    section_name = section.section_name if section else "Main Corridor"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'T409Title',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            alignment=1, # Center
            textColor=colors.HexColor("#1e293b")
        )
        subtitle_style = ParagraphStyle(
            'T409Subtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            alignment=1,
            textColor=colors.HexColor("#0284c7")
        )
        normal_style = ParagraphStyle(
            'T409Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        bold_label = ParagraphStyle(
            'T409Bold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#0f172a")
        )

        elements = []

        # Header
        elements.append(Paragraph("INDIAN RAILWAYS / Operating Department", subtitle_style))
        elements.append(Paragraph("FORM T/409 - CAUTION ORDER (TSR)", title_style))
        elements.append(Paragraph("Sanctioned Temporary Speed Restriction & Loco Dispatch Notice", subtitle_style))
        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=15))

        # Order metadata grid
        data_meta = [
            [Paragraph("Caution Order No:", bold_label), Paragraph(order.order_number, normal_style),
             Paragraph("Date & Time:", bold_label), Paragraph(order.issued_at.strftime("%Y-%m-%d %H:%M UTC"), normal_style)],
            [Paragraph("Section Corridor:", bold_label), Paragraph(f"{section_code} ({section_name})", normal_style),
             Paragraph("Status:", bold_label), Paragraph(order.status, bold_label)],
            [Paragraph("Issued By Officer:", bold_label), Paragraph(order.issued_by_officer, normal_style),
             Paragraph("Authority Standard:", bold_label), Paragraph("G&SR Para 4.09 / Form T/409", normal_style)]
        ]

        t_meta = Table(data_meta, colWidths=[120, 160, 120, 140])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_meta)
        elements.append(Spacer(1, 15))

        # Speed Restriction Box
        elements.append(Paragraph("RESTRICTION MANDATE FOR LOCO PILOT & GUARD", bold_label))
        elements.append(Spacer(1, 6))

        speed_box_data = [
            [
                Paragraph("<b>KM POST BOUNDS</b>", bold_label),
                Paragraph("<b>RESTRICTED SPEED</b>", bold_label),
                Paragraph("<b>REASON / CAUSE OF CAUTION</b>", bold_label)
            ],
            [
                Paragraph(f"<b>KM {order.km_from} to KM {order.km_to}</b>", normal_style),
                Paragraph(f"<font size=14 color='#dc2626'><b>{order.max_speed_kmh} KM/H</b></font>", normal_style),
                Paragraph(order.reason, normal_style)
            ]
        ]
        t_speed = Table(speed_box_data, colWidths=[160, 140, 240])
        t_speed.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#fff1f1")),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor("#ef4444")),
            ('INNERGRID', (0, 0), (-1, -1), 1, colors.HexColor("#fca5a5")),
            ('ALIGN', (1, 1), (1, 1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_speed)
        elements.append(Spacer(1, 20))

        # Instructions text
        elements.append(Paragraph("<b>LOCO PILOT ACKNOWLEDGEMENT & DISPATCH INSTRUCTIONS:</b>", bold_label))
        instructions = (
            "1. The Loco Pilot and Guard of all trains passing through this section must observe the speed restriction specified above.<br/>"
            "2. Whistle continuously on approach to the caution indicator and engineering boards.<br/>"
            "3. Resume normal section speed only after the rear-most vehicle has completely passed the Termination Board (T/P or T/G)."
        )
        elements.append(Paragraph(instructions, normal_style))
        elements.append(Spacer(1, 25))

        # Signatures & QR block
        sig_data = [
            [
                Paragraph("<b>Digital Verification QR</b><br/>[ TEJAS-T409-VALIDATED ]", normal_style),
                Paragraph("<b>________________________</b><br/>Station Master / Controller Sign", normal_style),
                Paragraph("<b>________________________</b><br/>Loco Pilot Signature & Copy Recd", normal_style)
            ]
        ]
        t_sig = Table(sig_data, colWidths=[180, 180, 180])
        t_sig.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ]))
        elements.append(t_sig)

        doc.build(elements)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=Form_T409_{order.order_number}.pdf"}
        )

    except ImportError:
        # Fallback if ReportLab not installed: text response
        content = f"INDIAN RAILWAYS FORM T/409 - CAUTION ORDER #{order.order_number}\n"
        content += f"Section: {section_code} | KM {order.km_from} - {order.km_to}\n"
        content += f"Speed Limit: {order.max_speed_kmh} KM/H\nReason: {order.reason}\nIssued By: {order.issued_by_officer}"
        return Response(content=content, media_type="text/plain")
