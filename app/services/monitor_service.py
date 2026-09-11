"""Background CCTV monitor — periodic plate detection."""
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, not_, select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.schemas.snapshot import CameraTypeEnum, EventTypeEnum
from app.services.camera_service import camera_service
from app.services.detector.pipeline import detection_pipeline
from app.services.storage_service import storage_service

logger = logging.getLogger("cctv_monitor")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_same_plate(plate1: Optional[str], plate2: Optional[str]) -> bool:
    """
    Compare two license plate strings with tolerance for OCR jitter / minor typos.
    - Exact match after cleaning whitespace and dashes
    - Levenshtein distance <= 1 for plates with length >= 5
    """
    if not plate1 or not plate2:
        return False
    p1 = "".join(plate1.split()).upper().replace("-", "")
    p2 = "".join(plate2.split()).upper().replace("-", "")
    if p1 == p2:
        return True
    if len(p1) >= 5 and len(p2) >= 5 and abs(len(p1) - len(p2)) <= 1:
        if _levenshtein_distance(p1, p2) <= 1:
            return True
    return False


def format_duration(duration_seconds: int) -> str:
    """Format duration in seconds into human-readable Uzbek string."""
    if duration_seconds < 60:
        return f"{max(1, duration_seconds)} soniya"
    mins = duration_seconds // 60
    secs = duration_seconds % 60
    if duration_seconds < 3600:
        if secs > 0:
            return f"{mins} daqiqa {secs} soniya"
        return f"{mins} daqiqa"
    hours = duration_seconds // 3600
    rem_mins = (duration_seconds % 3600) // 60
    if rem_mins > 0:
        return f"{hours} soat {rem_mins} daqiqa"
    return f"{hours} soat"


class AutoMonitorService:
    """
    Periodic CCTV Monitor with robust session management:
    - Captures frames from registered cameras
    - Runs detection pipeline (plate detection + EV classification)
    - Manages Parking Sessions:
        * On startup: Restores unclosed sessions from DB
        * On Arrival: Records START event (idempotent — checks DB first)
        * While Parked: Tracks active state with fuzzy OCR matching
        * On Plate Change: Requires transition debounce (2 cycles) to confirm
        * On Departure: Records END event with precise duration
    - Uses per-camera asyncio locks to prevent race conditions
    """

    def __init__(
        self,
        interval_seconds: int = 15,
        exit_threshold_cycles: int = 3,
        transition_threshold_cycles: int = 2
    ):
        self.interval_seconds = interval_seconds
        self.exit_threshold_cycles = exit_threshold_cycles
        self.transition_threshold_cycles = transition_threshold_cycles
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.last_seen_state: Dict[str, Dict] = {}
        self._camera_locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, camera_key: str) -> asyncio.Lock:
        """Get or create an asyncio lock for a specific camera."""
        if camera_key not in self._camera_locks:
            self._camera_locks[camera_key] = asyncio.Lock()
        return self._camera_locks[camera_key]

    def _restore_sessions_from_db(self):
        """
        On startup, load unclosed parking sessions from DB into last_seen_state.
        An unclosed session = a START record with no matching END for the same session_id.
        This prevents duplicate START events after server restart.
        """
        db: Session = SessionLocal()
        try:
            # Find all START records that have no corresponding END
            ended_session_ids = (
                select(CCTVSnapshot.session_id)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.END.value)
                .filter(CCTVSnapshot.session_id.isnot(None))
            )

            unclosed_starts = (
                db.query(CCTVSnapshot)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.START.value)
                .filter(CCTVSnapshot.detection_source == "CCTV_AUTO")
                .filter(CCTVSnapshot.session_id.isnot(None))
                .filter(~CCTVSnapshot.session_id.in_(ended_session_ids))
                .order_by(desc(CCTVSnapshot.created_at))
                .all()
            )

            restored_count = 0
            seen_camera_keys = set()
            for snap in unclosed_starts:
                camera_key = f"{snap.cs_id}_{snap.cp_id or 'CP01'}"
                # Only restore the most recent unclosed session per camera
                if camera_key in seen_camera_keys:
                    continue
                seen_camera_keys.add(camera_key)

                self.last_seen_state[camera_key] = {
                    "plate": snap.plate_number,
                    "session_id": snap.session_id,
                    "entry_at": snap.created_at,
                    "last_seen_at": snap.created_at,
                    "missed_cycles": 0,
                    "pending_new_plate": None,
                    "pending_cycles": 0,
                    "vehicle_type": snap.vehicle_type,
                    "is_ev": snap.is_ev,
                    "plate_color": snap.plate_color,
                    "confidence": snap.ai_confidence or 0.0,
                    "image_path": snap.image_path or "",
                    "plate_region_path": snap.plate_region_image,
                    "raw_ocr_text": snap.raw_ocr_text,
                }
                restored_count += 1
                logger.info(
                    f"🔄 Sessiya tiklandi: [{camera_key}] {snap.plate_number} "
                    f"(Session: {snap.session_id}, Kirish: {snap.created_at})"
                )

            if restored_count:
                logger.info(f"🔄 Jami {restored_count} ta yopilmagan sessiya DB dan tiklandi.")
            else:
                logger.info("✅ Yopilmagan sessiya yo'q — toza holatda boshlanmoqda.")
        except Exception as e:
            logger.error(f"Sessiyalarni tiklashda xatolik: {e}")
        finally:
            db.close()

    def _find_open_session_in_db(self, cs_id: str, cp_id: str, db: Session) -> Optional[CCTVSnapshot]:
        """
        Check DB for an open session (START without matching END) for this camera.
        Returns the open START snapshot or None.
        """
        try:
            ended_session_ids = (
                select(CCTVSnapshot.session_id)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.END.value)
                .filter(CCTVSnapshot.session_id.isnot(None))
            )

            open_start = (
                db.query(CCTVSnapshot)
                .filter(CCTVSnapshot.cs_id == cs_id)
                .filter(CCTVSnapshot.cp_id == cp_id)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.START.value)
                .filter(CCTVSnapshot.detection_source == "CCTV_AUTO")
                .filter(CCTVSnapshot.session_id.isnot(None))
                .filter(~CCTVSnapshot.session_id.in_(ended_session_ids))
                .order_by(desc(CCTVSnapshot.created_at))
                .first()
            )
            return open_start
        except Exception as e:
            logger.error(f"DB open session check error: {e}")
            return None

    async def start(self):
        if self.is_running:
            return
        if not settings.MONITOR_ENABLED:
            logger.info("Auto monitor is DISABLED in settings.")
            return

        # Restore unclosed sessions from DB before starting
        self._restore_sessions_from_db()

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

            # Deduplicate cameras by (cs_id, cp_id) to prevent double-inspection
            seen_keys = set()
            unique_cameras = []
            for cam in cameras:
                key = f"{cam.cs_id}_{cam.cp_id or 'CP01'}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_cameras.append(cam)

            tasks = [self._inspect_camera_safe(cam, db) for cam in unique_cameras]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            db.close()

    async def _inspect_camera_safe(self, camera: CCTVCamera, db: Session):
        """Wrapper that acquires per-camera lock before inspection."""
        camera_key = f"{camera.cs_id}_{camera.cp_id or 'CP01'}"
        lock = self._get_lock(camera_key)
        async with lock:
            await self._inspect_camera(camera, camera_key, db)

    async def _close_session(self, camera: CCTVCamera, camera_key: str, state: Dict, db: Session):
        """Record END event when a vehicle departs."""
        now = datetime.utcnow()
        entry_at = state.get("entry_at", now)
        last_seen_at = state.get("last_seen_at", now)
        duration_seconds = max(0, int((last_seen_at - entry_at).total_seconds()))
        duration_text = format_duration(duration_seconds)

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
            notes=f"Chiqish: Jami {duration_text} to'xtab turdi",
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

        logger.info(f"🚗💨 [{camera_key}] Chiqib ketdi: {plate} (To'xtab turish: {duration_text})")

    async def _inspect_camera(self, camera: CCTVCamera, camera_key: str, db: Session):
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
                state["pending_new_plate"] = None
                state["pending_cycles"] = 0
                if state["missed_cycles"] >= self.exit_threshold_cycles:
                    # Vehicle has departed
                    await self._close_session(camera, camera_key, state, db)
                    del self.last_seen_state[camera_key]
            return

        detected_plate = det_result.plate_number.strip().upper()
        now = datetime.utcnow()
        last_state = self.last_seen_state.get(camera_key)

        # CASE B: Same vehicle is still present (with fuzzy OCR match support)
        if last_state and is_same_plate(last_state.get("plate"), detected_plate):
            last_state["last_seen_at"] = now
            last_state["missed_cycles"] = 0
            last_state["pending_new_plate"] = None
            last_state["pending_cycles"] = 0
            # If newer frame has higher confidence, update plate display string
            if det_result.confidence > last_state.get("confidence", 0.0):
                last_state["plate"] = detected_plate
                last_state["confidence"] = det_result.confidence
            return

        # CASE C: A different vehicle is detected while previous was active
        if last_state and not is_same_plate(last_state.get("plate"), detected_plate):
            # Check if this new plate is already pending confirmation
            pending_plate = last_state.get("pending_new_plate")
            if pending_plate and is_same_plate(pending_plate, detected_plate):
                last_state["pending_cycles"] = last_state.get("pending_cycles", 0) + 1
                if last_state["pending_cycles"] >= self.transition_threshold_cycles:
                    # Confirmed: Old vehicle departed and new vehicle took the slot
                    await self._close_session(camera, camera_key, last_state, db)
                    del self.last_seen_state[camera_key]
                else:
                    # Waiting for confirmation on next cycle
                    return
            else:
                # First time seeing this new plate candidate
                last_state["pending_new_plate"] = detected_plate
                last_state["pending_cycles"] = 1
                return

        # CASE D: New Vehicle Arrival (START Event)
        # Idempotent check: Is there already an open session for this camera in DB?
        existing_open = self._find_open_session_in_db(
            camera.cs_id, camera.cp_id or "CP01", db
        )
        if existing_open and is_same_plate(existing_open.plate_number, detected_plate):
            # Already have an open session for this plate — restore to memory, don't create duplicate
            logger.info(
                f"🔄 [{camera_key}] Ochiq sessiya DB da mavjud: {existing_open.plate_number} "
                f"(Session: {existing_open.session_id}) — dublikat START yaratilmadi"
            )
            self.last_seen_state[camera_key] = {
                "plate": existing_open.plate_number,
                "session_id": existing_open.session_id,
                "entry_at": existing_open.created_at,
                "last_seen_at": now,
                "missed_cycles": 0,
                "pending_new_plate": None,
                "pending_cycles": 0,
                "vehicle_type": existing_open.vehicle_type or det_result.vehicle_type,
                "is_ev": existing_open.is_ev,
                "plate_color": existing_open.plate_color or det_result.plate_color,
                "confidence": max(existing_open.ai_confidence or 0.0, det_result.confidence),
                "image_path": existing_open.image_path or "",
                "plate_region_path": existing_open.plate_region_image,
                "raw_ocr_text": existing_open.raw_ocr_text or det_result.raw_ocr_text,
            }
            return

        # If there's an open session for a DIFFERENT plate, close it first
        if existing_open and not is_same_plate(existing_open.plate_number, detected_plate):
            logger.info(
                f"🔄 [{camera_key}] DB dagi ochiq sessiya ({existing_open.plate_number}) "
                f"yangi mashina ({detected_plate}) uchun yopilmoqda."
            )
            close_state = {
                "plate": existing_open.plate_number,
                "session_id": existing_open.session_id,
                "entry_at": existing_open.created_at,
                "last_seen_at": now,
                "confidence": existing_open.ai_confidence or 0.0,
                "image_path": existing_open.image_path or "",
                "vehicle_type": existing_open.vehicle_type or "UNKNOWN",
                "is_ev": existing_open.is_ev or False,
                "plate_color": existing_open.plate_color,
                "raw_ocr_text": existing_open.raw_ocr_text,
                "plate_region_path": existing_open.plate_region_image,
            }
            await self._close_session(camera, camera_key, close_state, db)

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
            "pending_new_plate": None,
            "pending_cycles": 0,
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
    exit_threshold_cycles=3,
    transition_threshold_cycles=2
)
