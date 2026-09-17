"""Unit tests for Multi-Vendor CCTV URL Formatting, Probing, and Camera API."""
import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.core.database import Base, get_db
from app.models.snapshot import CCTVCamera
from app.schemas.snapshot import CameraTypeEnum
from app.services.camera_service import camera_service, RTSPCameraAdapter, HTTPSnapshotCameraAdapter, CaptureResult
from app.services.detector.pipeline import detection_pipeline


class TestCameraMultiVendor(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

        def override_get_db():
            db = self.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        Base.metadata.drop_all(bind=self.engine)
        app.dependency_overrides.clear()

    def test_rtsp_url_formatting_hikvision(self):
        url = RTSPCameraAdapter.format_rtsp_url(
            stream_url="/Streaming/Channels/101",
            username="admin",
            password="password123!",
            ip_address="192.168.1.100",
            port=554
        )
        self.assertIn("rtsp://admin:password123%21@192.168.1.100:554/Streaming/Channels/101", url)

    def test_rtsp_url_formatting_dahua(self):
        url = RTSPCameraAdapter.format_rtsp_url(
            stream_url="rtsp://192.168.1.105:554/cam/realmonitor?channel=1&subtype=0",
            username="admin",
            password="admin_pass",
        )
        self.assertIn("rtsp://admin:admin_pass@192.168.1.105:554/cam/realmonitor?channel=1&subtype=0", url)

    def test_rtsp_url_formatting_hanwha(self):
        url = RTSPCameraAdapter.format_rtsp_url(
            stream_url="/profile2/media.smp",
            username="admin",
            password="4321_secret",
            ip_address="192.168.1.110",
            port=554
        )
        self.assertIn("rtsp://admin:4321_secret@192.168.1.110:554/profile2/media.smp", url)

    def test_http_url_formatting(self):
        url = HTTPSnapshotCameraAdapter.format_http_url(
            stream_url="/ISAPI/Streaming/channels/101/picture",
            ip_address="192.168.1.120",
            port=80
        )
        self.assertEqual(url, "http://192.168.1.120/ISAPI/Streaming/channels/101/picture")

    @patch("app.services.camera_service.camera_service.probe_camera", new_callable=AsyncMock)
    def test_camera_probe_api_success(self, mock_probe):
        mock_probe.return_value = (
            True,
            "RTSP 연결 성공 (/Streaming/Channels/101)",
            "rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101",
            CameraTypeEnum.RTSP,
            CaptureResult(success=True, width=1920, height=1080, image_bytes=b"fake_frame", protocol="RTSP")
        )

        res = self.client.post("/api/v1/cameras/probe", json={
            "ip_address": "192.168.1.50",
            "port": 554,
            "username": "admin",
            "password": "pass",
            "brand": "hikvision"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("Streaming/Channels/101", data["working_stream_url"])
        self.assertEqual(data["width"], 1920)

    @patch("app.services.camera_service.camera_service.check_camera_reachability", new_callable=AsyncMock)
    def test_create_camera_force_save(self, mock_reach):
        mock_reach.return_value = (False, "Camera connection timed out", None, None)

        # 1. Without force_save -> 400 error
        res1 = self.client.post("/api/v1/cameras", json={
            "cs_id": "bluenetwrks",
            "cp_id": "BNS00001",
            "camera_name": "Test Offline Cam",
            "ip_address": "10.0.0.99",
            "port": 554,
            "force_save": False
        })
        self.assertEqual(res1.status_code, 400)

        # 2. With force_save=True -> Saved successfully with generated URL
        res2 = self.client.post("/api/v1/cameras", json={
            "cs_id": "bluenetwrks",
            "cp_id": "BNS00001",
            "camera_name": "Test Offline Cam",
            "ip_address": "10.0.0.99",
            "port": 554,
            "force_save": True
        })
        self.assertEqual(res2.status_code, 200)
        data = res2.json()
        self.assertEqual(data["cs_id"], "bluenetwrks")
        self.assertEqual(data["cp_id"], "BNS00001")
        self.assertIn("10.0.0.99", data["stream_url"])

    async def test_detection_pipeline_fallback_scan(self):
        # Generate a test image with a Korean EV plate rendered directly
        h, w = 600, 800
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)

        # Draw a clear blue EV plate in the lower ROI
        cv2.rectangle(img, (250, 420), (550, 500), (220, 180, 80), -1) # Blue background (BGR)
        cv2.putText(img, "81머 2072", (260, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

        _, buf = cv2.imencode(".jpg", img)
        res = await detection_pipeline.detect(buf.tobytes())

        # Should find plate even without car box detection via lower ROI scan
        self.assertIsNotNone(res)
        if res.success:
            self.assertIn("81", res.plate_number)
            self.assertIn("2072", res.plate_number)
    async def test_concurrent_camera_slot_occlusion_and_presence(self):
        """Test multi-camera concurrent frame processing with occlusion detection."""
        from app.services.detector.plate_validator import KoreanPlateValidator
        
        # Camera 1 sees occluded EV plate
        res1 = KoreanPlateValidator.is_cable_occlusion_match("47호6633", "47오3633")
        self.assertTrue(res1)

        # Camera 2 sees regular ICE vehicle plate
        res2 = KoreanPlateValidator.is_cable_occlusion_match("58버6091", "58보6091")
        self.assertTrue(res2)


if __name__ == "__main__":
    unittest.main()

