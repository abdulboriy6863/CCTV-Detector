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
    format_duration,
    _levenshtein_distance
)


class TestParkingSessionLifecycle(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # In-memory SQLite for testing
        self.engine = create_engine("sqlite:///:memory:")
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

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_duration_formatting(self):
        self.assertEqual(format_duration(25), "25 soniya")
        self.assertEqual(format_duration(60), "1 daqiqa")
        self.assertEqual(format_duration(95), "1 daqiqa 35 soniya")
        self.assertEqual(format_duration(600), "10 daqiqa")
        self.assertEqual(format_duration(3600), "1 soat")
        self.assertEqual(format_duration(3660), "1 soat 1 daqiqa")

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

        await self.monitor._inspect_camera(self.camera, self.db)

        # Check DB: Exactly 1 record (START)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)
        self.assertEqual(records[0].plate_number, "81머2072")
        self.assertTrue(records[0].is_ev)
        self.assertTrue(records[0].session_id.startswith("PARK_CS_TEST_01_CP01_"))

        camera_key = "CS_TEST_01_CP01"
        self.assertIn(camera_key, self.monitor.last_seen_state)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["plate"], "81머2072")
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 0)

        # 2. Cycles 2 & 3: Car is STILL PARKED (Same plate)
        await self.monitor._inspect_camera(self.camera, self.db)
        await self.monitor._inspect_camera(self.camera, self.db)

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
        await self.monitor._inspect_camera(self.camera, self.db)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 1)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 4. Cycle 5: Second missed frame
        await self.monitor._inspect_camera(self.camera, self.db)
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 2)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 5. Cycle 6: Third missed frame -> Vehicle has officially EXITED!
        await self.monitor._inspect_camera(self.camera, self.db)

        # Check DB: Now 2 records (1 START, 1 END)
        records = self.db.query(CCTVSnapshot).order_by(CCTVSnapshot.id.asc()).all()
        self.assertEqual(len(records), 2)
        start_rec = records[0]
        end_rec = records[1]

        self.assertEqual(start_rec.event_type, EventTypeEnum.START.value)
        self.assertEqual(end_rec.event_type, EventTypeEnum.END.value)
        self.assertEqual(end_rec.session_id, start_rec.session_id)
        self.assertEqual(end_rec.plate_number, "81머2072")
        self.assertTrue("Chiqish: Jami 10 daqiqa to'xtab turdi" in end_rec.notes)

        # Tracking state should now be cleared
        self.assertNotIn(camera_key, self.monitor.last_seen_state)

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_ocr_jitter_does_not_split_session(self, mock_capture, mock_detect, mock_save_img):
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")

        # 1. Car arrives: detected as 52어0580 (confidence 0.85)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="52어0580", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.85
        )
        await self.monitor._inspect_camera(self.camera, self.db)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)

        # 2. Next cycle: OCR slightly shifts to 52어0586 (confidence 0.96)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="52어0586", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.96
        )
        await self.monitor._inspect_camera(self.camera, self.db)

        # Should NOT create a second session! DB still has 1 record
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)
        camera_key = "CS_TEST_01_CP01"
        self.assertEqual(self.monitor.last_seen_state[camera_key]["plate"], "52어0586")

    @patch("app.services.monitor_service.storage_service.save_image", new_callable=AsyncMock)
    @patch("app.services.monitor_service.detection_pipeline.detect", new_callable=AsyncMock)
    @patch("app.services.monitor_service.camera_service.capture_snapshot", new_callable=AsyncMock)
    async def test_session_transition_debounce_when_different_car_arrives(self, mock_capture, mock_detect, mock_save_img):
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")

        # 1. Car A arrives
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="58버6091", vehicle_type=VehicleTypeEnum.REGULAR, is_ev=False, confidence=0.88
        )
        await self.monitor._inspect_camera(self.camera, self.db)

        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].plate_number, "58버6091")
        self.assertEqual(records[0].event_type, EventTypeEnum.START.value)

        # 2. Car B detected for the 1st cycle (pending debounce)
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.95
        )
        await self.monitor._inspect_camera(self.camera, self.db)

        # DB should STILL have only 1 record (Car A is not abruptly evicted on 1 cycle)
        records = self.db.query(CCTVSnapshot).all()
        self.assertEqual(len(records), 1)

        # 3. Car B detected for the 2nd cycle (transition confirmed!)
        await self.monitor._inspect_camera(self.camera, self.db)

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
        # 1. Car arrives
        mock_capture.return_value = CaptureResult(success=True, image_bytes=b"fake_jpeg", protocol="RTSP")
        mock_detect.return_value = DetectionResult(
            success=True, plate_number="81머2072", vehicle_type=VehicleTypeEnum.EV, is_ev=True, confidence=0.90
        )
        await self.monitor._inspect_camera(self.camera, self.db)

        camera_key = "CS_TEST_01_CP01"
        self.assertIn(camera_key, self.monitor.last_seen_state)

        # 2. Network error on camera snapshot
        mock_capture.return_value = CaptureResult(success=False, error_message="RTSP timeout", protocol="RTSP")
        await self.monitor._inspect_camera(self.camera, self.db)

        # Missed cycles should NOT increase due to camera connection glitch
        self.assertEqual(self.monitor.last_seen_state[camera_key]["missed_cycles"], 0)
        self.assertEqual(len(self.db.query(CCTVSnapshot).all()), 1)


if __name__ == "__main__":
    unittest.main()

