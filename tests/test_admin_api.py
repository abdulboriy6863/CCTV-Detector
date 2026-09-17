"""
Unit tests for Admin API endpoints (/api/v1/admin/purge-data, /api/v1/admin/reset-sessions).
"""

import unittest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.snapshot import CCTVSnapshot
from app.models.camera import CCTVCamera
from app.models.alert import NonEVAlert
from app.services.monitor_service import auto_monitor_service


class TestAdminAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        cls.TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

        def override_get_db():
            db = cls.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=cls.engine)
        app.dependency_overrides.clear()

    def setUp(self):
        db = self.TestingSessionLocal()
        db.query(NonEVAlert).delete()
        db.query(CCTVSnapshot).delete()
        db.query(CCTVCamera).delete()

        # Seed sample camera and snapshot
        cam = CCTVCamera(
            id=1,
            cs_id="bluenetwrks",
            cp_id="BNS00000",
            camera_name="Test CCTV",
            camera_type="RTSP",
            stream_url="rtsp://fake/stream1",
            is_active=True
        )
        snap = CCTVSnapshot(
            cs_id="bluenetwrks",
            cp_id="BNS00000",
            plate_number="81머2072",
            event_type="START",
            is_ev=True,
            vehicle_type="EV",
            created_at=datetime.utcnow()
        )
        db.add_all([cam, snap])
        db.commit()
        db.close()

        auto_monitor_service.active_sessions["cam_1"] = {"plate": "81머2072"}

    def test_reset_active_sessions_endpoint(self):
        self.assertIn("cam_1", auto_monitor_service.active_sessions)
        res = self.client.post("/api/v1/admin/reset-sessions")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["cleared_count"], 1)
        self.assertEqual(len(auto_monitor_service.active_sessions), 0)

    def test_purge_data_endpoint(self):
        db = self.TestingSessionLocal()
        self.assertEqual(db.query(CCTVSnapshot).count(), 1)
        self.assertEqual(db.query(CCTVCamera).count(), 1)
        db.close()

        payload = {
            "purge_snapshots": True,
            "purge_alerts": True,
            "purge_cameras": True,
            "delete_images": False,
            "reset_active_sessions": True
        }
        res = self.client.post("/api/v1/admin/purge-data", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["deleted_snapshots"], 1)
        self.assertEqual(data["deleted_cameras"], 1)

        db = self.TestingSessionLocal()
        self.assertEqual(db.query(CCTVSnapshot).count(), 0)
        self.assertEqual(db.query(CCTVCamera).count(), 0)
        db.close()


if __name__ == "__main__":
    unittest.main()
