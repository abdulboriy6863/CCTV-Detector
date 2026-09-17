"""Unit tests for CCTV Parking Session Lifecycle."""
import os
import sys
import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import Base
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.schemas.snapshot import EventTypeEnum, DetectionResult, VehicleTypeEnum
from app.services.camera_service import CaptureResult
from app.services.monitor_service import (
    AutoMonitorService,
    is_same_plate,
    format_duration
)



class TestParkingSessionLifecycle(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        from sqlalchemy.pool import StaticPool
        # In-memory SQLite for testing with StaticPool
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        self.camera = CCTVCamera(
            id=1,
            cs_id="CS_TEST_01",
            cp_id="CP01",
            camera_name="Test Cam",
            camera_type="RTSP",
            stream_url="rtsp://fake-stream",
            is_active=True
        )
        self.db.add(self.camera)
        self.db.commit()

        self.monitor = AutoMonitorService(
            interval_seconds=15,
            exit_threshold_cycles=3,
            transition_threshold_cycles=2
        )
        self.monitor.active_sessions.clear()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_duration_formatting(self):
        self.assertEqual(format_duration(25), "25초")
        self.assertEqual(format_duration(60), "1분")
        self.assertEqual(format_duration(95), "1분 35초")
        self.assertEqual(format_duration(600), "10분")
        self.assertEqual(format_duration(3600), "1시간")
        self.assertEqual(format_duration(3660), "1시간 1분")

    def test_fuzzy_plate_matching(self):
        self.assertTrue(is_same_plate("81머2072", "81머2072"))
        self.assertTrue(is_same_plate("81머 2072", "81머2072"))
        # 1-character OCR jitter
        self.assertTrue(is_same_plate("52어0580", "52어0586"))
        self.assertTrue(is_same_plate("81머2072", "81머2078"))
        # Completely different plates
        self.assertFalse(is_same_plate("58버6091", "81머2072"))
        self.assertFalse(is_same_plate(None, "81머2072"))

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_session_lifecycle_start_stay_and_end(self, mock_capture, mock_detect, mock_save_img):
        mock_capture.return_value = CaptureResult(
            success=True,
            image_bytes=b"fake_jpeg_data",
            protocol="RTSP"
        )

        # 1. First cycle: Car "81머2072" arrives
        mock_detect.return_value = DetectionResult(
            success=True,
            plate_number="81머2072",
            vehicle_type=VehicleTypeEnum.EV,
            is_ev=True,
            confidence=0.92,
            raw_ocr_text="81머2072"
        )

        camera_key = "CS_TEST_01_CP01"
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Check DB: Exactly 1 record (START)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertEqual(records[0].plate_number, "81머2072")
        self.assertTrue(records[0].is_ev)
        self.assertTrue(records[0].session_id.startswith("PARK_CS_TEST_01_CP01_"))

        self.assertIn(camera_key, self.monitor.last_seen_state)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["plate"], "81머2072")
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 0)

        # 2. Cycles 2 & 3: Car is STILL PARKED (Same plate)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Check DB: Still only 1 record (No duplicates created!)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 0)

        # Simulate 10 minutes elapsed
        now_time = datetime.utcnow()
        self.monitor.last_seen_state[camera_key]["entry_at"] = now_time - timedelta(minutes=10)
        self.monitor.last_seen_state[camera_key]["last_seen_at"] = now_time

        # 3. Cycle 4: Frame misses plate (temporary blur)
        mock_detect.return_value = DetectionResult(success=False, plate_number=None)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 1)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 4. Cycle 5: Second missed frame
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 2)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 5. Cycle 6: Third missed frame -> Vehicle has officially EXITED!
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Check DB: Now 2 records (1 START, 1 END)
        records = self.db.query(CCTVSnapshot).order_by(CCTVSnapshot.id.asc()).all()
        self.assertEqual(len(records), 2)
        start_rec = records[0]
        end_rec = records[1]

        self.assertEqual(start_rec.event_type, EventTypeEnum.START.value)
        self.assertEqual(end_rec.event_type, EventTypeEnum.END.value)
        self.assertEqual(end_rec.session_id, start_rec.session_id)
        self.assertEqual(end_rec.plate_number, "81머2072")
        self.assertTrue("10분" in end_rec.notes)

        # Tracking state should now be cleared
        self.assertNotIn(camera_key, self.monitor.last_seen_state)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_ocr_jitter_does_not_split_session(self, mock_capture, mock_detect, mock_save_img):
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        camera_key = "CS_TEST_01_CP01"

        # 1. Car arrives: detected as 52어0580 (confidence 0.85)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="52어0580", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.85
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 2. Next cycle: OCR slightly shifts to 52어0586 (confidence 0.96)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="52어0586", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.96
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Should NOT create a second session! DB still has 1 record
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["plate"], "52어0586")

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_session_transition_debounce_when_different_car_arrives(self, mock_capture, mock_detect, mock_save_img):
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        camera_key = "CS_TEST_01_CP01"

        # 1. Car A arrives
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="58버6091", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.88
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].plate_number, "58버6091")
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)

        # 2. Car B detected for the 1st cycle (pending debounce)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.95
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # DB should STILL have only 1 record (Car A is not abruptly evicted on 1 cycle)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)

        # 3. Car B detected for the 2nd cycle (transition confirmed!)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # DB should now have 3 records: Car A START, Car A END, Car B START
        records = self.db.query(CCTVSnapshot).order_by(CCTVSnapshot.id.asc()).all()
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].plate_number, "58버6091")
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)

        self.assertEqual(records[1].plate_number, "58버6091")
        self.assertEqual(records[1].event_type, EventTypeEnum.END.value)

        self.assertEqual(records[2].plate_number, "81머2072")
        self.assertEqual(records[2].event_type, EventTypeEnum.START.value)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_network_glitch_does_not_drop_state(self, mock_capture, mock_detect, mock_save_img):
        camera_key = "CS_TEST_01_CP01"
        # 1. Car arrives
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.90
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        self.assertIn(camera_key, self.monitor.last_seen_state)

        # 2. Network error on camera snapshot
        mock_capture.return_value = CaptureResult(success=False, error_message="RTSP timeout", protocol="RTSP")
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Missed cycles should NOT increase due to camera connection glitch
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 0)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_idempotent_start_prevents_duplicate_on_restart(self, mock_capture, mock_detect, mock_save_img):
        """Simulate server restart: open session exists in DB, should not create duplicate START."""
        camera_key = f"cam_{self.camera.id}"
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")

        # Pre-create an open session in DB (simulating previous server run)
        existing_start = CCTVSnapshot(
            cs_id="CS_TEST_01",
            cp_id="CP01",
            connector_id=1,
            session_id=f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911080348",
            plate_number="81머2072",
            event_type=EventTypeEnum.START.value,
            image_path="some/path.jpg",
            ai_confidence=0.87,
            status="SUCCESS",
            notes="Kirish: To'xtab turish boshlandi",
            vehicle_type="EV",
            is_ev=True,
            detection_source="CCTV_AUTO",
            created_at=datetime.utcnow() - timedelta(minutes=5)
        )
        self.db.add(existing_start)
        self.db.commit()

        # Simulate monitor startup (calls _restore_sessions_from_db)
        with patch("app.services.monitor_service.SessionLocal", side_effect=self.SessionLocal):
            self.monitor._restore_sessions_from_db()

        # Now monitor detects the same car
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.96
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # DB should STILL have only 1 START record (no duplicate!)
        start_records = self.db.query(CCTVSnapshot).filter(
            CCTVSnapshot.event_type == EventTypeEnum.START.value
        ).all()
        self.assertEqual(len(start_records), 1)
        self.assertEqual(start_records[0].session_id, f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911080348")

        # Monitor state should be restored from DB
        self.assertIn(camera_key, self.monitor.active_sessions)
        self.assertEqual(self.monitor.active_sessions[camera_key]["plate"], "81머2072")
        self.assertEqual(
            self.monitor.active_sessions[camera_key]["session_id"],
            f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911080348"
        )

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_session_restore_on_startup(self, mock_capture, mock_detect, mock_save_img):
        """Test that _restore_sessions_from_db correctly loads unclosed sessions."""
        # Create an open session (START without END)
        open_start = CCTVSnapshot(
            cs_id="CS_TEST_01", cp_id="CP01", connector_id=1,
            session_id=f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911070000",
            plate_number="58버6091", event_type=EventTypeEnum.START.value,
            image_path="test.jpg", ai_confidence=0.88, status="SUCCESS",
            notes="Kirish", vehicle_type="REGULAR", is_ev=False,
            detection_source="CCTV_AUTO",
            created_at=datetime.utcnow() - timedelta(minutes=30)
        )
        # Create a closed session (both START and END)
        closed_start = CCTVSnapshot(
            cs_id="CS_TEST_01", cp_id="CP01", connector_id=1,
            session_id=f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911060000",
            plate_number="12가3456", event_type=EventTypeEnum.START.value,
            image_path="test2.jpg", ai_confidence=0.90, status="SUCCESS",
            notes="Kirish", vehicle_type="EV", is_ev=True,
            detection_source="CCTV_AUTO",
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
        closed_end = CCTVSnapshot(
            cs_id="CS_TEST_01", cp_id="CP01", connector_id=1,
            session_id=f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911060000",
            plate_number="12가3456", event_type=EventTypeEnum.END.value,
            image_path="test2.jpg", ai_confidence=0.90, status="SUCCESS",
            notes="Chiqish: Jami 10 daqiqa to'xtab turdi", vehicle_type="EV", is_ev=True,
            detection_source="CCTV_AUTO",
            created_at=datetime.utcnow() - timedelta(hours=1, minutes=50)
        )
        self.db.add_all([open_start, closed_start, closed_end])
        self.db.commit()

        # Create a fresh monitor and patch SessionLocal to use our test DB
        fresh_monitor = AutoMonitorService(interval_seconds=15, exit_threshold_cycles=3)
        with patch("app.services.monitor_service.SessionLocal", side_effect=self.SessionLocal):
            fresh_monitor._restore_sessions_from_db()

        # Only the OPEN session should be restored
        camera_key = f"cam_{self.camera.id}"
        self.assertIn(camera_key, fresh_monitor.active_sessions)
        self.assertEqual(fresh_monitor.active_sessions[camera_key]["plate"], "58버6091")
        self.assertEqual(
            fresh_monitor.active_sessions[camera_key]["session_id"],
            f"PARK_CS_TEST_01_CP01_CAM{self.camera.id}_20260911070000"
        )

    def test_get_all_slot_statuses_multi_camera(self):
        """Test that get_all_slot_statuses aggregates cameras and occupancy state correctly."""
        # Add a second camera
        cam2 = CCTVCamera(
            id=2,
            cs_id="CS_TEST_01",
            cp_id="CP02",
            camera_name="Bay 2 Cam",
            camera_type="RTSP",
            stream_url="rtsp://fake-stream-2",
            is_active=True
        )
        self.db.add(cam2)
        self.db.commit()

        # Simulate Slot 1 is occupied by an EV
        self.monitor.active_sessions[f"cam_{self.camera.id}"] = {
            "session_id": "PARK_01",
            "plate": "52어0586",
            "vehicle_type": "EV",
            "is_ev": True,
            "plate_color": "blue",
            "entry_at": datetime.now() - timedelta(minutes=15),
            "last_seen_at": datetime.now()
        }

        # Query slot statuses
        statuses = self.monitor.get_all_slot_statuses(self.db)
        self.assertEqual(len(statuses), 2)

        # Verify Slot 1
        slot1 = next(s for s in statuses if s["camera_id"] == 1)
        self.assertTrue(slot1["is_occupied"])
        self.assertEqual(slot1["current_plate"], "52어0586")
        self.assertTrue(slot1["is_ev"])
        self.assertIn("분", slot1["duration_formatted"])

        # Verify Slot 2 (Empty)
        slot2 = next(s for s in statuses if s["camera_id"] == 2)
        self.assertFalse(slot2["is_occupied"])
        self.assertIsNone(slot2["current_plate"])


    @patch("app.services.monitor_service.charger_service.is_physically_plugged")
    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_cable_occluded_24hour_charging_session_stays_alive(
        self, mock_capture, mock_detect, mock_save_img, mock_plugged
    ):
        """
        Verify that a car charging for 24 hours with constant cable occlusion jitter
        (47호6633 -> 47오3633 -> 47오8633 -> 47오9633 -> missed OCR)
        stays in ONE single continuous session with zero premature START/END spam.
        """
        camera_key = f"cam_{self.camera.id}"
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")

        # 1. Car arrives and clear plate 47호6633 is detected before plugging
        mock_plugged.return_value = False
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="47호6633", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.95
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertEqual(records[0].plate_number, "47호6633")

        # 2. Driver plugs in cable -> charger becomes plugged in
        mock_plugged.return_value = True

        # 3. Simulate continuous cycles with cable occlusion variations
        jitter_plates = ["47오3633", "47오8633", "47오9633", "47호3633", "47호8633"]
        for i in range(20):
            plate_candidate = jitter_plates[i % len(jitter_plates)]
            mock_detect.return_value = DetectionResult(
                success=True, plate_number=plate_candidate, vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.88
            )
            await self.monitor._inspect_camera(self.camera, camera_key, self.db)

            # DB must STILL have only the single 1 START record!
            records = self.db.query(CCTVSnapshot).all()
            self.assertEqual(len(records), 1)

        # 4. Simulate OCR complete failure (cable blocking plate)
        for _ in range(10):
            mock_detect.return_value = DetectionResult(success=False, plate_number=None, vehicle_present=True)
            await self.monitor._inspect_camera(self.camera, camera_key, self.db)
            records = self.db.query(CCTVSnapshot).all()
            self.assertEqual(len(records), 1)

        # 5. Simulate 24 hours elapsed
        now_time = datetime.utcnow()
        self.monitor.active_sessions[camera_key]["entry_at"] = now_time - timedelta(hours=24)
        self.monitor.active_sessions[camera_key]["last_seen_at"] = now_time

        # 6. Driver unplugs cable and car departs
        mock_plugged.return_value = False
        mock_detect.return_value = DetectionResult(success=False, plate_number=None, vehicle_present=False)

        # 3 cycles of empty slot -> officially exits
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Verify DB has exactly 2 records: 1 START and 1 END with 1 day (24 hours) duration
        records = self.db.query(CCTVSnapshot).order_by(CCTVSnapshot.id.asc()).all()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertEqual(records[1].event_type, EventTypeEnum.END.value)
        self.assertEqual(records[0].session_id, records[1].session_id)
        self.assertTrue("1일" in records[1].notes or "24시간" in records[1].notes)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_start_event_persists_base64_plate_crop(self, mock_capture, mock_detect, mock_save_img):
        """Verify that START event saves the lightweight base64 plate crop to plate_region_image."""
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_save_img.return_value = "2026/09/17/test_snap.jpg"
        camera_key = "CS_TEST_01_CP01"

        b64_crop = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ_SAMPLE_PLATE_CROP"
        mock_detect.return_value = DetectionResult(
            success=True,
            plate_number="81머2072",
            vehicle_type=VehicleTypeEnum.EV,
            is_ev=True,
            confidence=0.96,
            plate_crop_base64=b64_crop
        )

        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertEqual(records[0].plate_region_image, b64_crop)
        self.assertEqual(records[0].plate_number, "81머2072")

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_end_event_zero_image_writes(self, mock_capture, mock_detect, mock_save_img):
        """Verify that vehicle END (departure) event creates zero new disk images."""
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_save_img.return_value = "2026/09/17/test_snap.jpg"
        camera_key = "CS_TEST_01_CP01"

        # 1. Car arrives (START) -> image saved
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.96
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(mock_save_img.call_count, 1)

        # 2. Reset mock save count
        mock_save_img.reset_mock()

        # 3. 3 empty cycles -> car leaves (END)
        mock_detect.return_value = DetectionResult(success=False, plate_number=None, vehicle_present=False)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Verify: Zero image writes were performed on exit!
        self.assertEqual(mock_save_img.call_count, 0)

        # Verify DB records
        records = self.db.query(CCTVSnapshot).order_by(CCTVSnapshot.id.asc()).all()
        self.assertEqual(len(records), 2)
    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_multi_cycle_session_retention_during_intermittent_frame_loss(
        self, mock_capture, mock_detect, mock_save_img
    ):
        """Verify session stays alive across 10 cycles when vehicle bbox is visible even if OCR is intermittently missed."""
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_save_img.return_value = "2026/09/17/test_snap.jpg"
        camera_key = f"cam_{self.camera.id}"

        # 1. Car arrives
        mock_detect.return_value = DetectionResult(
            success=True,
            plate_number="52어0586",
            vehicle_type=VehicleTypeEnum.EV,
            is_ev=True,
            confidence=0.92,
            vehicle_present=True
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 2. Intermittent OCR failure for 10 cycles, but YOLO vehicle bbox detected (vehicle_present=True)
        for _ in range(10):
            mock_detect.return_value = DetectionResult(
                success=False,
                plate_number=None,
                vehicle_present=True
            )
            await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # DB must still have only 1 START record (not closed, no END record)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertIn(camera_key, self.monitor.active_sessions)
        self.assertEqual(self.monitor.active_sessions[camera_key]["missed_cycles"], 0)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_weak_ocr_jitter_rejected_during_parking(
        self, mock_capture, mock_detect, mock_save_img
    ):
        """Verify that weak candidate OCR (<0.60) is ignored as noise while vehicle remains parked."""
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_save_img.return_value = "2026/09/17/test_snap.jpg"
        camera_key = f"cam_{self.camera.id}"

        # 1. Car arrives with strong anchor confidence (0.95)
        mock_detect.return_value = DetectionResult(
            success=True,
            plate_number="81머2072",
            vehicle_type=VehicleTypeEnum.EV,
            is_ev=True,
            confidence=0.95,
            vehicle_present=True
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 2. Noisy low-confidence random plate glitch (0.50 confidence)
        mock_detect.return_value = DetectionResult(
            success=True,
            plate_number="99하9999",
            vehicle_type=VehicleTypeEnum.REGULAR,
            is_ev=False,
            confidence=0.50,
            vehicle_present=True
        )
        await self.monitor._inspect_camera(self.camera, camera_key, self.db)

        # Must NOT set pending transition or disrupt anchor plate
        active = self.monitor.active_sessions[camera_key]
        self.assertEqual(active["plate"], "81머2072")
        self.assertIsNone(active.get("pending_new_plate"))
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)


if __name__ == "__main__":
    unittest.main()


