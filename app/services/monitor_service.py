"""Background CCTV monitor — periodic plate detection and parking session management."""
import asyncio
import logging
import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.config import settings, get_kst_now
from app.core.database import SessionLocal
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.schemas.snapshot import CameraTypeEnum, EventTypeEnum
from app.services.camera_service import camera_service
from app.services.detector.pipeline import detection_pipeline
from app.services.detector.plate_validator import KoreanPlateValidator
from app.services.storage_service import storage_service
from app.services.charger_service import charger_service, format_duration_kr

logger = logging.getLogger("cctv_monitor")


def is_same_plate(plate1: Optional[str], plate2: Optional[str]) -> bool:
    """
    Compare two license plate strings with tolerance for OCR jitter, minor typos,
    and vertical charging cable occlusion.
    """
    if not plate1 or not plate2:
        return False
    return KoreanPlateValidator.is_cable_occlusion_match(plate1, plate2)



def format_duration(duration_seconds: int) -> str:
    """Format duration in seconds into standard Korean string."""
    return format_duration_kr(duration_seconds)


class AutoMonitorService:
    """
    Periodic CCTV Monitor with deterministic, lock-protected Parking Sessions:
    - Captures frames from registered cameras
    - Runs detection pipeline (plate detection + EV classification)
    - Manages Parking Sessions:
        * On Startup: Restores unclosed sessions from DB into active_sessions
        * Scenario 1: No plate recognized -> After exit_threshold_cycles (3), logs ONE END event
        * Scenario 2: Same plate recognized -> Updates last_seen_at (zero DB writes, no duplicates)
        * Scenario 3: Different plate recognized -> Debounced transition (2 cycles), closes old (ONE END), opens new (ONE START)
        * Scenario 4: Slot was empty -> Logs ONE START event
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
        self.active_sessions: Dict[str, Dict] = {}
        self.camera_health: Dict[str, Dict] = {}
        self._camera_locks: Dict[str, asyncio.Lock] = {}

    def get_all_slot_statuses(self, db: Session) -> list:
        """
        Aggregate live parking slot states across all cameras.
        Combines DB camera records with active memory sessions, health states, and CSMS charging status.
        """
        cameras = db.query(CCTVCamera).order_by(CCTVCamera.id.asc()).all()
        statuses = []
        now = get_kst_now()

        for cam in cameras:
            camera_key = f"cam_{cam.id}"
            session_info = self.active_sessions.get(camera_key)
            health_info = self.camera_health.get(camera_key, {
                "is_online": cam.is_active,
                "last_error": None
            })

            is_occupied = session_info is not None
            current_plate = session_info.get("plate") if is_occupied else None
            vehicle_type = session_info.get("vehicle_type") if is_occupied else None
            is_ev = session_info.get("is_ev", False) if is_occupied else False
            plate_color = session_info.get("plate_color") if is_occupied else None
            session_id = session_info.get("session_id") if is_occupied else None
            entry_at = session_info.get("entry_at") if is_occupied else None
            last_seen_at = session_info.get("last_seen_at") if is_occupied else None

            duration_seconds = 0
            duration_formatted = None
            if is_occupied and entry_at:
                duration_seconds = max(0, int((now - entry_at).total_seconds()))
                duration_formatted = format_duration_kr(duration_seconds)

            live_stream_url = f"/api/v1/cameras/{cam.id}/live" if cam.is_active else None

            # Fetch live charger status from CSMS
            charger_info = charger_service.get_charger_realtime_status(
                cs_id=cam.cs_id,
                cp_id=cam.cp_id,
                db=db,
                session_info=session_info
            )

            statuses.append({
                "camera_id": cam.id,
                "camera_name": cam.camera_name or f"{cam.cs_id} - {cam.cp_id or 'BNS00000'}",
                "cs_id": cam.cs_id,
                "cp_id": cam.cp_id or "BNS00000",
                "camera_type": cam.camera_type,
                "is_active": cam.is_active,
                "is_online": health_info.get("is_online", cam.is_active),
                "stream_url": cam.stream_url,
                "live_stream_url": live_stream_url,
                "is_occupied": is_occupied,
                "current_plate": current_plate,
                "vehicle_type": vehicle_type,
                "is_ev": is_ev,
                "plate_color": plate_color,
                "session_id": session_id,
                "entry_at": entry_at,
                "last_seen_at": last_seen_at,
                "duration_seconds": duration_seconds,
                "duration_formatted": duration_formatted,
                "last_image_url": session_info.get("image_path") if is_occupied else None,
                "last_error": health_info.get("last_error"),
                # Live CSMS Charger & Violation info
                "connector_status": charger_info.get("connector_status", "AVAILABLE"),
                "connector_status_kr": charger_info.get("connector_status_kr", "사용 가능"),
                "connector_status_uz": charger_info.get("connector_status_uz", "Mavjud (Bo'sh)"),
                "is_charging": charger_info.get("is_charging", False),
                "battery_soc": charger_info.get("battery_soc"),
                "charge_power_kw": charger_info.get("charge_power_kw", 0.0),
                "charged_energy_kwh": charger_info.get("charged_energy_kwh", 0.0),
                "charging_duration_seconds": charger_info.get("charging_duration_seconds", 0),
                "charging_duration_formatted": charger_info.get("charging_duration_formatted", "0분"),
                "overstay_seconds": charger_info.get("overstay_seconds", 0),
                "overstay_formatted": charger_info.get("overstay_formatted", "0분"),
                "violation_type": charger_info.get("violation_type", "NONE"),
                "violation_label_kr": charger_info.get("violation_label_kr", "정상"),
                "violation_label_uz": charger_info.get("violation_label_uz", "Normal"),
                "violation_level": charger_info.get("violation_level", "info"),
                "action_required_kr": charger_info.get("action_required_kr", "—"),
                "action_required_uz": charger_info.get("action_required_uz", "—"),
            })

        return statuses

    @property
    def last_seen_state(self) -> Dict[str, Dict]:
        """Backward compatibility alias for tests and external inspection."""
        return self.active_sessions

    @last_seen_state.setter
    def last_seen_state(self, value: Dict[str, Dict]):
        self.active_sessions = value

    def _get_lock(self, camera_key: str) -> asyncio.Lock:
        """Get or create an asyncio lock for a specific camera."""
        if camera_key not in self._camera_locks:
            self._camera_locks[camera_key] = asyncio.Lock()
        return self._camera_locks[camera_key]

    def _restore_sessions_from_db(self):
        """
        On startup, load unclosed parking sessions from DB into active_sessions.
        An unclosed session = a START record that has no END record with the same session_id.
        """
        db: Session = SessionLocal()
        try:
            # 1. Collect all session_ids that already have an END event
            ended_rows = (
                db.query(CCTVSnapshot.session_id)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.END.value)
                .filter(CCTVSnapshot.session_id.isnot(None))
                .all()
            )
            ended_session_ids = {r[0] for r in ended_rows if r[0]}

            # 2. Get all registered cameras
            cameras = db.query(CCTVCamera).all()
            cam_by_id = {c.id: c for c in cameras}

            # 3. Get all START events
            start_snapshots = (
                db.query(CCTVSnapshot)
                .filter(CCTVSnapshot.event_type == EventTypeEnum.START.value)
                .filter(CCTVSnapshot.detection_source == "CCTV_AUTO")
                .filter(CCTVSnapshot.session_id.isnot(None))
                .order_by(desc(CCTVSnapshot.created_at), desc(CCTVSnapshot.id))
                .all()
            )

            restored_count = 0
            seen_camera_keys = set()
            for snap in start_snapshots:
                # Match camera_key: check if session_id has _CAM{id}_
                matched_cam_id = None
                if snap.session_id and "_CAM" in snap.session_id:
                    try:
                        part = snap.session_id.split("_CAM")[1]
                        matched_cam_id = int(part.split("_")[0])
                    except Exception:
                        matched_cam_id = None

                if matched_cam_id and matched_cam_id in cam_by_id:
                    camera_key = f"cam_{matched_cam_id}"
                else:
                    # Match by station and charger
                    matching_cams = [c for c in cameras if c.cs_id == snap.cs_id and c.cp_id == snap.cp_id]
                    if matching_cams:
                        camera_key = f"cam_{matching_cams[0].id}"
                    else:
                        continue

                if camera_key in seen_camera_keys:
                    continue
                seen_camera_keys.add(camera_key)

                # Only restore if this session was never closed
                if snap.session_id not in ended_session_ids:
                    self.active_sessions[camera_key] = {
                        "session_id": snap.session_id,
                        "plate": snap.plate_number,
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
                logger.info(f"🔄 Jami {restored_count} ta faol sessiya DB dan tiklandi.")
            else:
                logger.info("✅ Ochiq sessiya yo'q — monitoring toza holatda boshlanmoqda.")
        except Exception as e:
            logger.error(f"Sessiyalarni tiklashda xatolik: {e}")
        finally:
            db.close()

    async def start(self):
        if self.is_running:
            return
        if not settings.MONITOR_ENABLED:
            logger.info("Auto monitor is DISABLED in settings.")
            return

        # Restore unclosed sessions from DB before starting loop
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
            cameras = db.query(CCTVCamera).filter(CCTVCamera.is_active.is_(True)).order_by(CCTVCamera.id.asc()).all()
            if not cameras:
                return

            for cam in cameras:
                try:
                    await self._inspect_camera_safe(cam, db)
                except Exception as cam_err:
                    logger.error(f"Error inspecting camera {cam.id} ({cam.cs_id}/{cam.cp_id}): {cam_err}")
                await asyncio.sleep(0.5)

            import gc
            gc.collect()
        finally:
            db.close()

    async def _inspect_camera_safe(self, camera: CCTVCamera, db: Session):
        """Wrapper that acquires per-camera lock before inspection."""
        camera_key = f"cam_{camera.id}"
        lock = self._get_lock(camera_key)
        async with lock:
            await self._inspect_camera(camera, camera_key, db)

    async def _record_start_event(
        self,
        camera: CCTVCamera,
        camera_key: str,
        detected_plate: str,
        det_result,
        image_bytes: bytes,
        now: datetime,
        db: Session
    ):
        """Record ONE START event in DB and update active_sessions state."""
        session_id = f"PARK_{camera.cs_id}_{camera.cp_id or 'CP01'}_CAM{camera.id}_{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        relative_path = storage_service.generate_relative_path(
            cs_id=camera.cs_id, cp_id=camera.cp_id or "BNS00000",
            plate_number=detected_plate
        )
        await storage_service.save_image(image_bytes, relative_path)

        plate_region_path = None
        plate_crop_bytes = getattr(det_result, 'plate_crop_bytes', None)
        if plate_crop_bytes:
            plate_region_path = relative_path.replace(".jpg", "_plate.jpg")
            await storage_service.save_image(plate_crop_bytes, plate_region_path)

        is_non_ev = not det_result.is_ev and det_result.vehicle_type != "UNKNOWN"

        entry_snapshot = CCTVSnapshot(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "BNS00000",
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

        self.active_sessions[camera_key] = {
            "session_id": session_id,
            "plate": detected_plate,
            "anchor_plate": detected_plate,
            "anchor_confidence": det_result.confidence,
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

    async def _close_session(
        self,
        camera: CCTVCamera,
        camera_key: str,
        state: Dict,
        db: Session,
        departure_time: Optional[datetime] = None
    ):
        """Record ONE END event when a vehicle departs."""
        if charger_service.is_physically_plugged(camera.cs_id, camera.cp_id, db):
            logger.info(f"🔌 [{camera_key}] Cannot close session for {state.get('plate')}: Charger is still plugged in.")
            return False

        now = get_kst_now()
        entry_at = state.get("entry_at", now)


        if departure_time:
            duration_seconds = max(1, int((departure_time - entry_at).total_seconds()))
        else:
            last_seen_at = state.get("last_seen_at", now)
            if last_seen_at > entry_at:
                duration_seconds = int((last_seen_at - entry_at).total_seconds())
            else:
                # If only recognized in initial frame before departing:
                est_leave = now - timedelta(seconds=self.exit_threshold_cycles * self.interval_seconds)
                duration_seconds = max(self.interval_seconds, int((est_leave - entry_at).total_seconds()))
                if duration_seconds < 1:
                    duration_seconds = max(1, int((now - entry_at).total_seconds()))

        duration_text = format_duration(duration_seconds)

        plate = state.get("plate", "UNKNOWN")
        session_id = state.get("session_id")

        exit_snapshot = CCTVSnapshot(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id or "BNS00000",
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
            cp_id=camera.cp_id or "BNS00000",
            username=camera.username,
            password=camera.password,
            ip_address=camera.ip_address,
            port=camera.port
        )

        current_health = self.camera_health.get(camera_key, {"is_online": True, "failed_count": 0})
        if not capture_result.success or not capture_result.image_bytes:
            failed_count = current_health.get("failed_count", 0) + 1
            # Mark offline only after 3 consecutive failures (approx 45s of total failure)
            is_offline = failed_count >= 3
            self.camera_health[camera_key] = {
                "is_online": not is_offline,
                "failed_count": failed_count,
                "last_error": capture_result.error_message or "Frame capture failed",
                "last_checked_at": get_kst_now()
            }
            return

        # Record online state and reset failed count
        self.camera_health[camera_key] = {
            "is_online": True,
            "failed_count": 0,
            "last_error": None,
            "last_checked_at": get_kst_now()
        }

        # 2. Run detection pipeline
        det_result = await detection_pipeline.detect(capture_result.image_bytes)

        now = get_kst_now()
        active = self.active_sessions.get(camera_key)

        # 3. Check hardware ground truth from CSMS Charger status
        is_plugged = charger_service.is_physically_plugged(
            cs_id=camera.cs_id,
            cp_id=camera.cp_id,
            db=db
        )

        # ----------------------------------------------------
        # SCENARIO 1: No plate recognized in this frame (or confidence too low)
        # ----------------------------------------------------
        if not det_result.success or not det_result.plate_number or det_result.confidence < 0.45:
            if active:
                vehicle_is_visible = getattr(det_result, 'vehicle_present', False)
                # If charger is actively plugged or YOLO sees a car in the slot, KEEP SESSION ALIVE!
                if is_plugged or vehicle_is_visible:
                    active["last_seen_at"] = now
                    active["missed_cycles"] = 0
                    active["pending_new_plate"] = None
                    active["pending_cycles"] = 0
                    if vehicle_is_visible and not is_plugged:
                        logger.debug(f"🚗 [{camera_key}] Plate not recognized, but vehicle bbox is still visible. Keeping session alive for {active.get('plate')}.")
                    return

                active["missed_cycles"] = active.get("missed_cycles", 0) + 1
                active["pending_new_plate"] = None
                active["pending_cycles"] = 0
                if active["missed_cycles"] >= self.exit_threshold_cycles:
                    logger.info(f"🚪 [{camera_key}] Slot confirmed empty (no vehicle & unplugged for {self.exit_threshold_cycles} cycles). Closing session for {active.get('plate')}.")
                    # Vehicle has officially departed (unplugged + no car in frame) -> Close session
                    closed = await self._close_session(camera, camera_key, active, db)
                    if closed is not False:
                        del self.active_sessions[camera_key]
            return


        detected_plate = det_result.plate_number.strip().upper()

        # ----------------------------------------------------
        # SCENARIO 2: Slot is currently occupied
        # ----------------------------------------------------
        if active:
            anchor_or_curr = active.get("anchor_plate") or active.get("plate")
            # Subcase 2A: Same vehicle or cable-occluded variant of anchor plate
            if is_same_plate(anchor_or_curr, detected_plate) or is_same_plate(active.get("plate"), detected_plate):
                active["last_seen_at"] = now
                active["missed_cycles"] = 0
                active["pending_new_plate"] = None
                active["pending_cycles"] = 0
                if det_result.confidence > active.get("confidence", 0.0):
                    active["plate"] = detected_plate
                    active["confidence"] = det_result.confidence
                return

            # If charger is plugged, do not allow false transition caused by OCR noise
            if is_plugged:
                active["last_seen_at"] = now
                active["missed_cycles"] = 0
                active["pending_new_plate"] = None
                active["pending_cycles"] = 0
                logger.info(f"🔌 [{camera_key}] Charger is plugged in. Retaining anchor session {anchor_or_curr} (ignored noisy candidate {detected_plate})")
                return


            # Subcase 2B: Different plate detected -> Debounce transition
            pending_plate = active.get("pending_new_plate")
            if pending_plate and is_same_plate(pending_plate, detected_plate):
                active["pending_cycles"] = active.get("pending_cycles", 0) + 1
                if active["pending_cycles"] >= self.transition_threshold_cycles:
                    # CONFIRMED TRANSITION:
                    # 1. Close old vehicle session with departure_time = now
                    await self._close_session(camera, camera_key, active, db, departure_time=now)
                    del self.active_sessions[camera_key]

                    # 2. Open new vehicle session
                    await self._record_start_event(
                        camera, camera_key, detected_plate, det_result, capture_result.image_bytes, now, db
                    )
                    return
                else:
                    # Waiting for second confirmation cycle
                    return
            else:
                # First time seeing this different plate candidate
                active["pending_new_plate"] = detected_plate
                active["pending_cycles"] = 1
                return

        # ----------------------------------------------------
        # SCENARIO 3: Slot was EMPTY -> New vehicle arrives
        # ----------------------------------------------------
        await self._record_start_event(
            camera, camera_key, detected_plate, det_result, capture_result.image_bytes, now, db
        )


auto_monitor_service = AutoMonitorService(
    interval_seconds=settings.MONITOR_INTERVAL_SECONDS,
    exit_threshold_cycles=6,
    transition_threshold_cycles=3
)
