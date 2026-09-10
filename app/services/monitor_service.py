"""Background CCTV monitor — periodic plate detection."""
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.schemas.snapshot import CameraTypeEnum, EventTypeEnum
from app.services.camera_service import camera_service
from app.services.detector.pipeline import detection_pipeline
from app.services.storage_service import storage_service

logger = logging.getLogger("cctv_monitor")


class AutoMonitorService:
    """
    Periodic CCTV Monitor:
    - Captures frames from registered cameras
    - Runs detection pipeline (plate detection + EV classification)
    - Saves NEW vehicles to DB
    - Sends alerts for non-EV vehicles
    - Deduplicates same car within 30 minutes
    """

    def __init__(self, interval_seconds: int = 15):
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.last_seen_state: Dict[str, Dict] = {}

    async def start(self):
        if self.is_running:
            return
        if not settings.MONITOR_ENABLED:
            logger.info("Auto monitor is DISABLED in settings.")
            return
        self.is_running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Auto CCTV Monitor started (interval={self.interval_seconds}s)")

    async def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Auto CCTV Monitor stopped.")

    async def _monitor_loop(self):
        while self.is_running:
            try:
                await self.check_all_cameras()
            except Exception as e:
                logger.error(f"Monitor cycle error: {e}")
            await asyncio.sleep(self.interval_seconds)

    async def check_all_cameras(self):
        db: Session = SessionLocal()
        try:
            cameras = db.query(CCTVCamera).filter(CCTVCamera.is_active.is_(True)).all()
            for cam in cameras:
                await self._inspect_camera(cam, db)
        finally:
            db.close()

    async def _inspect_camera(self, camera: CCTVCamera, db: Session):
        camera_key = f"{camera.cs_id}_{camera.cp_id or 'CP01'}"

        # 1. Capture frame
        capture_result = await camera_service.capture_snapshot(
            camera_type=CameraTypeEnum(camera.camera_type),
            stream_url=camera.stream_url,
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "CP01",
            username=camera.username,
            password=camera.password,
            ip_address=camera.ip_address,
            port=camera.port
        )

        if not capture_result.success or not capture_result.image_bytes:
            return

        # 2. Run detection pipeline
        det_result = await detection_pipeline.detect(capture_result.image_bytes)

        if not det_result.success or not det_result.plate_number:
            if camera_key in self.last_seen_state:
                del self.last_seen_state[camera_key]
            return

        detected_plate = det_result.plate_number.upper()

        # 3. Deduplication
        last_state = self.last_seen_state.get(camera_key)
        if last_state:
            if (last_state.get("last_plate") == detected_plate and
                    (datetime.utcnow() - last_state.get("last_saved_at", datetime.min)) < timedelta(minutes=30)):
                return

        # 4. Save image
        relative_path = storage_service.generate_relative_path(
            cs_id=camera.cs_id, cp_id=camera.cp_id or "CP01",
            plate_number=detected_plate
        )
        await storage_service.save_image(capture_result.image_bytes, relative_path)

        # Save plate crop if available
        plate_region_path = None
        if det_result.plate_crop_bytes:
            plate_region_path = relative_path.replace(".jpg", "_plate.jpg")
            await storage_service.save_image(det_result.plate_crop_bytes, plate_region_path)

        # 5. Determine alert
        is_non_ev = not det_result.is_ev and det_result.vehicle_type != "UNKNOWN"

        # 6. Save to DB
        snapshot = CCTVSnapshot(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "CP01",
            connector_id=1,
            session_id=f"AUTO_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            plate_number=detected_plate,
            event_type=EventTypeEnum.MOTION.value,
            image_path=relative_path,
            ai_confidence=det_result.confidence,
            status="SUCCESS",
            notes=f"Auto detected via {capture_result.protocol}",
            vehicle_type=det_result.vehicle_type,
            is_ev=det_result.is_ev,
            plate_color=det_result.plate_color,
            alert_sent=is_non_ev,
            alert_type="NON_EV_WARNING" if is_non_ev else None,
            detection_source="CCTV_AUTO",
            raw_ocr_text=det_result.raw_ocr_text,
            plate_region_image=plate_region_path,
            created_at=datetime.utcnow()
        )
        db.add(snapshot)
        db.commit()

        self.last_seen_state[camera_key] = {
            "last_plate": detected_plate,
            "last_saved_at": datetime.utcnow()
        }

        ev_emoji = "⚡" if det_result.is_ev else "🚨"
        logger.info(f"{ev_emoji} [{camera_key}] {det_result.vehicle_type}: {detected_plate}")


auto_monitor_service = AutoMonitorService(interval_seconds=settings.MONITOR_INTERVAL_SECONDS)
