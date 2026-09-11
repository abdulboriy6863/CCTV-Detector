"""Detection API — upload image for plate recognition & EV classification."""
import base64
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional

logger = logging.getLogger(__name__)

from app.core.database import get_db
from app.models.snapshot import CCTVSnapshot
from app.schemas.snapshot import SnapshotResponse, EventTypeEnum
from app.services.detector.pipeline import detection_pipeline
from app.services.storage_service import storage_service

router = APIRouter()


@router.post("/upload", summary="Upload image for EV detection & plate recognition")
async def detect_upload(
    file: UploadFile = File(...),
    cs_id: str = Form("CS_DEFAULT"),
    cp_id: str = Form("CP01"),
    db: Session = Depends(get_db)
):
    """
    Upload a vehicle image to:
    1. Detect license plate (YOLOv8)
    2. Read plate text (PaddleOCR)
    3. Classify EV or Regular (HSV color)
    4. Save to database
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Run detection pipeline
    result = await detection_pipeline.detect(image_bytes)

    if not result.success or not result.plate_number:
        # Still return detection info even if no plate found
        return {
            "success": False,
            "message": result.error_message or "No license plate detected",
            "processing_time_ms": round(result.processing_time_ms, 1),
            "data": None
        }

    # Save full image
    relative_path = storage_service.generate_relative_path(
        cs_id=cs_id, cp_id=cp_id, plate_number=result.plate_number
    )
    await storage_service.save_image(image_bytes, relative_path)

    # Save plate crop image
    plate_region_path = None
    if result.plate_crop_bytes:
        plate_region_path = relative_path.replace(".jpg", "_plate.jpg")
        await storage_service.save_image(result.plate_crop_bytes, plate_region_path)

    # Determine alert for non-EV
    is_non_ev = not result.is_ev and result.vehicle_type != "UNKNOWN"

    # Save to DB with graceful fallback if DB is unreachable
    snapshot_id = None
    try:
        snapshot = CCTVSnapshot(
            cs_id=cs_id,
            cp_id=cp_id,
            connector_id=1,
            session_id=f"UPLOAD_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            plate_number=result.plate_number,
            event_type=EventTypeEnum.MANUAL.value,
            image_path=relative_path,
            ai_confidence=result.confidence,
            status="SUCCESS",
            notes=f"Manual upload detection. Time: {result.processing_time_ms:.1f}ms",
            vehicle_type=result.vehicle_type,
            is_ev=result.is_ev,
            plate_color=result.plate_color,
            alert_sent=is_non_ev,
            alert_type="NON_EV_WARNING" if is_non_ev else None,
            detection_source="MANUAL_UPLOAD",
            raw_ocr_text=result.raw_ocr_text,
            plate_region_image=plate_region_path,
            created_at=datetime.utcnow()
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        snapshot_id = snapshot.id
    except Exception as e:
        logger.warning(f"Database persistence skipped (DB connection issue): {e}")
        try:
            db.rollback()
        except Exception:
            pass

    # Build response
    response_data = {
        "id": snapshot_id or 1,
        "plate_number": result.plate_number,
        "vehicle_type": result.vehicle_type,
        "is_ev": result.is_ev,
        "plate_color": result.plate_color,
        "confidence": round(result.confidence, 3),
        "raw_ocr_text": result.raw_ocr_text,
        "processing_time_ms": round(result.processing_time_ms, 1),
        "alert": "NON_EV_WARNING" if is_non_ev else None,
        "image_url": f"/api/v1/vehicles/{snapshot_id}/image" if snapshot_id else None,
    }

    # Include plate region base64 for immediate UI display
    if result.plate_crop_bytes:
        response_data["plate_region_base64"] = base64.b64encode(result.plate_crop_bytes).decode("utf-8")

    return {
        "success": True,
        "message": f"{'⚡ EV' if result.is_ev else '🚗 일반'} 차량 감지됨: {result.plate_number}",
        "processing_time_ms": round(result.processing_time_ms, 1),
        "data": response_data
    }
