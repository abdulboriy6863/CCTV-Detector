from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class EventTypeEnum(str, Enum):
    START = "START"
    END = "END"
    MANUAL = "MANUAL"
    MOTION = "MOTION"


class VehicleTypeEnum(str, Enum):
    EV = "EV"
    REGULAR = "REGULAR"
    COMMERCIAL = "COMMERCIAL"
    UNKNOWN = "UNKNOWN"


class DetectionSourceEnum(str, Enum):
    MANUAL_UPLOAD = "MANUAL_UPLOAD"
    CCTV_AUTO = "CCTV_AUTO"
    API_TRIGGER = "API_TRIGGER"


class CameraTypeEnum(str, Enum):
    RTSP = "RTSP"
    HTTP_SNAPSHOT = "HTTP_SNAPSHOT"
    ONVIF = "ONVIF"


# ===== Detection Schemas =====

class DetectionResult(BaseModel):
    """Result returned from the detection pipeline."""
    success: bool
    plate_number: Optional[str] = None
    vehicle_type: VehicleTypeEnum = VehicleTypeEnum.UNKNOWN
    is_ev: bool = False
    plate_color: Optional[str] = None
    confidence: float = 0.0
    raw_ocr_text: Optional[str] = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    plate_region_base64: Optional[str] = None  # For API response
    plate_crop_bytes: Optional[bytes] = None


# ===== Snapshot Schemas =====

class SnapshotResponse(BaseModel):
    id: int
    cs_id: str
    cp_id: str
    connector_id: Optional[int] = 1
    session_id: Optional[str] = None
    plate_number: Optional[str] = None
    event_type: str
    image_path: str
    image_url: Optional[str] = None
    download_url: Optional[str] = None
    ai_confidence: Optional[float] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    vehicle_type: Optional[str] = "UNKNOWN"
    is_ev: bool = False
    plate_color: Optional[str] = None
    alert_sent: bool = False
    alert_type: Optional[str] = None
    detection_source: Optional[str] = "MANUAL_UPLOAD"
    raw_ocr_text: Optional[str] = None
    plate_region_image: Optional[str] = None
    camera_name: Optional[str] = None
    battery_soc: Optional[int] = None
    charge_power_kw: Optional[float] = None
    violation_type: Optional[str] = None
    violation_label_kr: Optional[str] = None
    violation_label_uz: Optional[str] = None
    action_required_kr: Optional[str] = None
    action_required_uz: Optional[str] = None
    action_required: Optional[str] = None
    stay_duration_seconds: Optional[int] = None
    stay_duration_formatted: Optional[str] = None
    is_ongoing: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SnapshotFilterParams(BaseModel):
    plate_number: Optional[str] = None
    cs_id: Optional[str] = None
    cp_id: Optional[str] = None
    vehicle_type: Optional[VehicleTypeEnum] = None
    is_ev: Optional[bool] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


# ===== Camera Schemas =====

class CameraBase(BaseModel):
    cs_id: str
    cp_id: Optional[str] = None
    camera_name: Optional[str] = None
    camera_type: CameraTypeEnum = CameraTypeEnum.RTSP
    stream_url: Optional[str] = None
    ip_address: Optional[str] = None
    port: Optional[int] = 554
    username: Optional[str] = None
    password: Optional[str] = None
    brand: Optional[str] = None
    is_active: bool = True


class CameraCreate(CameraBase):
    force_save: Optional[bool] = False


class CameraUpdate(BaseModel):
    camera_name: Optional[str] = None
    camera_type: Optional[CameraTypeEnum] = None
    stream_url: Optional[str] = None
    ip_address: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    brand: Optional[str] = None
    is_active: Optional[bool] = None
    force_save: Optional[bool] = False


class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CameraProbeRequest(BaseModel):
    ip_address: Optional[str] = None
    port: Optional[int] = None
    camera_type: Optional[CameraTypeEnum] = None
    stream_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    brand: Optional[str] = None


class CameraProbeResponse(BaseModel):
    success: bool
    message: str
    camera_type: Optional[str] = None
    working_stream_url: Optional[str] = None
    image_base64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


# ===== Stats & Realtime Slot Status Schemas =====

class SlotStatusResponse(BaseModel):
    """Real-time occupancy and stream status of a CCTV-monitored parking slot."""
    camera_id: int
    camera_name: str
    cs_id: str
    cp_id: str
    camera_type: str = "RTSP"
    is_active: bool = True
    is_online: bool = True
    stream_url: Optional[str] = None
    live_stream_url: Optional[str] = None
    is_occupied: bool = False
    current_plate: Optional[str] = None
    vehicle_type: Optional[str] = None
    is_ev: bool = False
    plate_color: Optional[str] = None
    session_id: Optional[str] = None
    entry_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    duration_seconds: int = 0
    duration_formatted: Optional[str] = None
    last_image_url: Optional[str] = None
    last_error: Optional[str] = None
    connector_status: Optional[str] = "AVAILABLE"
    connector_status_kr: Optional[str] = "사용 가능"
    connector_status_uz: Optional[str] = "Mavjud (Bo'sh)"
    is_charging: bool = False
    battery_soc: Optional[int] = None
    charge_power_kw: float = 0.0
    charged_energy_kwh: float = 0.0
    charging_duration_seconds: int = 0
    charging_duration_formatted: Optional[str] = "0분"
    overstay_seconds: int = 0
    overstay_formatted: Optional[str] = "0분"
    violation_type: str = "NONE"
    violation_label_kr: str = "정상"
    violation_label_uz: str = "Normal"
    violation_level: str = "info"
    action_required_kr: Optional[str] = "—"
    action_required_uz: Optional[str] = "—"


class DashboardStats(BaseModel):
    total_detections: int = 0
    ev_count: int = 0
    regular_count: int = 0
    today_detections: int = 0
    today_ev: int = 0
    today_regular: int = 0
    today_alerts: int = 0
    active_cameras: int = 0

