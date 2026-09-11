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
    - Manages Parking Sessions:
        * On Arrival: Records START event with unique session_id
        * While Parked: Tracks active state without creating duplicates
        * On Departure (after 3 missed cycles): Records END event with duration
    - Sends alerts for non-EV vehicles
    """

    def __init__(self, interval_seconds: int = 15, exit_threshold_cycles: int = 3):
        self.interval_seconds = interval_seconds
        self.exit_threshold_cycles = exit_threshold_cycles
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
            if not cameras:
                return
            # Run all active camera inspections concurrently
            tasks = [self._inspect_camera(cam, db) for cam in cameras]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            db.close()

    async def _close_session(self, camera: CCTVCamera, camera_key: str, state: Dict, db: Session):
        """Record END event when a vehicle departs."""
        now = datetime.utcnow()
        entry_at = state.get("entry_at", now)
        last_seen_at = state.get("last_seen_at", now)
        duration_seconds = max(0, int((last_seen_at - entry_at).total_seconds()))
        duration_minutes = max(1, round(duration_seconds / 60))

        plate = state.get("plate", "UNKNOWN")
        session_id = state.get("session_id")

        exit_snapshot = CCTVSnapshot(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "CP01",
            connector_id=1,
            session_id=session_id,
            plate_number=plate,
            event_type=EventTypeEnum.END.value,
            image_path=state.get("image_path", ""),
            ai_confidence=state.get("confidence", 0.0),
            status="SUCCESS",
            notes=f"Chiqish: Jami {duration_minutes} daqiqa to'xtab turdi",
            vehicle_type=state.get("vehicle_type", "UNKNOWN"),
            is_ev=state.get("is_ev", False),
            plate_color=state.get("plate_color"),
            alert_sent=False,
            alert_type=None,
            detection_source="CCTV_AUTO",
            raw_ocr_text=state.get("raw_ocr_text"),
            plate_region_image=state.get("plate_region_path"),
            created_at=now
        )
        db.add(exit_snapshot)
        db.commit()

        logger.info(f"🚗💨 [{camera_key}] Chiqib ketdi: {plate} (To'xtab turish: {duration_minutes} daq)")

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
            # Network or camera error: do not increment missed cycles
            return

        # 2. Run detection pipeline
        det_result = await detection_pipeline.detect(capture_result.image_bytes)

        # CASE A: No plate recognized in this frame
        if not det_result.success or not det_result.plate_number:
            if camera_key in self.last_seen_state:
                state = self.last_seen_state[camera_key]
                state["missed_cycles"] = state.get("missed_cycles", 0) + 1
                if state["missed_cycles"] >= self.exit_threshold_cycles:
                    # Vehicle has departed
                    await self._close_session(camera, camera_key, state, db)
                    del self.last_seen_state[camera_key]
            return

        detected_plate = det_result.plate_number.strip().upper()
        now = datetime.utcnow()
        last_state = self.last_seen_state.get(camera_key)

        # CASE B: Same vehicle is still present
        if last_state and last_state.get("plate") == detected_plate:
            last_state["last_seen_at"] = now
            last_state["missed_cycles"] = 0
            return

        # CASE C: A different vehicle arrived while previous was tracked
        if last_state and last_state.get("plate") != detected_plate:
            # Close previous vehicle's session
            await self._close_session(camera, camera_key, last_state, db)
            del self.last_seen_state[camera_key]

        # CASE D: New Vehicle Arrival (START Event)
        session_id = f"PARK_{camera.cs_id}_{camera.cp_id or 'CP01'}_{now.strftime('%Y%m%d%H%M%S')}"

        # Save snapshot image
        relative_path = storage_service.generate_relative_path(
            cs_id=camera.cs_id, cp_id=camera.cp_id or "CP01",
            plate_number=detected_plate
        )
        await storage_service.save_image(capture_result.image_bytes, relative_path)

        # Save plate crop if available
        plate_region_path = None
        plate_crop_bytes = getattr(det_result, 'plate_crop_bytes', None)
        if plate_crop_bytes:
            plate_region_path = relative_path.replace(".jpg", "_plate.jpg")
            await storage_service.save_image(plate_crop_bytes, plate_region_path)

        is_non_ev = not det_result.is_ev and det_result.vehicle_type != "UNKNOWN"

        entry_snapshot = CCTVSnapshot(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "CP01",
            connector_id=1,
            session_id=session_id,
            plate_number=detected_plate,
            event_type=EventTypeEnum.START.value,
            image_path=relative_path,
            ai_confidence=det_result.confidence,
            status="SUCCESS",
            notes="Kirish: To'xtab turish boshlandi",
            vehicle_type=det_result.vehicle_type,
            is_ev=det_result.is_ev,
            plate_color=det_result.plate_color,
            alert_sent=is_non_ev,
            alert_type="NON_EV_WARNING" if is_non_ev else None,
            detection_source="CCTV_AUTO",
            raw_ocr_text=det_result.raw_ocr_text,
            plate_region_image=plate_region_path,
            created_at=now
        )
        db.add(entry_snapshot)
        db.commit()

        self.last_seen_state[camera_key] = {
            "plate": detected_plate,
            "session_id": session_id,
            "entry_at": now,
            "last_seen_at": now,
            "missed_cycles": 0,
            "vehicle_type": det_result.vehicle_type,
            "is_ev": det_result.is_ev,
            "plate_color": det_result.plate_color,
            "confidence": det_result.confidence,
            "image_path": relative_path,
            "plate_region_path": plate_region_path,
            "raw_ocr_text": det_result.raw_ocr_text,
        }

        ev_emoji = "⚡" if det_result.is_ev else "🚨"
        logger.info(f"{ev_emoji} [{camera_key}] Kirish (START): {det_result.vehicle_type} - {detected_plate} (Session: {session_id})")


auto_monitor_service = AutoMonitorService(
    interval_seconds=settings.MONITOR_INTERVAL_SECONDS,
    exit_threshold_cycles=3
)
