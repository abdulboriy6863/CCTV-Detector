from datetime import datetime
from sqlalchemy import (
    Column, BigInteger, Integer, String, Float, DateTime,
    Enum, Text, Index, Boolean,
)
from app.core.database import Base


class CCTVSnapshot(Base):
    """
    tb_cctv_snapshots — Vehicle detection records.
    Keeps ALL original columns from CCTV-Screenshot + new EV columns.
    """
    __tablename__ = "tb_cctv_snapshots"

    # ===== ORIGINAL COLUMNS (from CCTV-Screenshot) =====
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True, index=True)
    cs_id = Column(String(50), nullable=False, index=True, comment="Charge Station ID")
    cp_id = Column(String(50), nullable=False, index=True, comment="Charge Point ID")
    connector_id = Column(Integer, nullable=True, default=1, comment="Connector ID")
    session_id = Column(String(100), nullable=True, index=True, comment="Charging Session ID")
    plate_number = Column(String(30), nullable=True, index=True, comment="Detected License Plate")
    event_type = Column(
        Enum("START", "END", "MANUAL", "MOTION", name="event_type_enum"),
        nullable=False, default="MANUAL",
        comment="Trigger Event Type"
    )
    image_path = Column(String(255), nullable=False, comment="Relative path to stored image")
    ai_confidence = Column(Float, nullable=True, comment="AI LPR confidence (0.0-1.0)")
    status = Column(String(20), default="SUCCESS", comment="Processing status")
    notes = Column(Text, nullable=True, comment="Additional notes")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # ===== NEW EV DETECTION COLUMNS =====
    vehicle_type = Column(
        Enum("EV", "REGULAR", "COMMERCIAL", "UNKNOWN", name="vehicle_type_enum"),
        default="UNKNOWN", nullable=False,
        comment="Vehicle type: EV (electric), REGULAR, COMMERCIAL"
    )
    is_ev = Column(Boolean, default=False, nullable=False, comment="Quick EV flag")
    plate_color = Column(String(20), nullable=True, comment="Plate background color (blue/white/yellow)")
    alert_sent = Column(Boolean, default=False, nullable=False, comment="Alert sent for non-EV")
    alert_type = Column(String(50), nullable=True, comment="Alert type (NON_EV_WARNING etc.)")
    detection_source = Column(
        Enum("MANUAL_UPLOAD", "CCTV_AUTO", "API_TRIGGER", name="detection_source_enum"),
        default="MANUAL_UPLOAD", nullable=False,
        comment="How was this detected"
    )
    raw_ocr_text = Column(String(100), nullable=True, comment="Raw OCR output text")
    plate_region_image = Column(String(255), nullable=True, comment="Path to cropped plate image")

    __table_args__ = (
        Index("idx_cs_cp_created", "cs_id", "cp_id", "created_at"),
        Index("idx_plate_created", "plate_number", "created_at"),
        Index("idx_vehicle_type", "vehicle_type"),
        Index("idx_is_ev", "is_ev"),
        {"extend_existing": True}
    )

    def __repr__(self):
        return f"<CCTVSnapshot id={self.id} plate={self.plate_number} type={self.vehicle_type}>"


class CCTVCamera(Base):
    """
    tb_cctv_cameras — Camera connection details.
    EXACT same structure as CCTV-Screenshot (no changes needed).
    """
    __tablename__ = "tb_cctv_cameras"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    cs_id = Column(String(50), nullable=False, index=True, comment="Charge Station ID")
    cp_id = Column(String(50), nullable=True, index=True, comment="Charge Point ID")
    camera_name = Column(String(100), nullable=True, comment="Camera name")
    camera_type = Column(
        Enum("RTSP", "HTTP_SNAPSHOT", "ONVIF", name="camera_type_enum"),
        default="RTSP", nullable=False
    )
    stream_url = Column(String(500), nullable=False, comment="RTSP or HTTP Snapshot URL")
    ip_address = Column(String(50), nullable=True)
    port = Column(Integer, nullable=True, default=554)
    username = Column(String(100), nullable=True)
    password = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_camera_cs_cp", "cs_id", "cp_id"),
        {"extend_existing": True}
    )

    def __repr__(self):
        return f"<CCTVCamera id={self.id} cs_id={self.cs_id} type={self.camera_type}>"
