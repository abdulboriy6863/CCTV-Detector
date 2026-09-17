import unittest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.snapshot import CCTVSnapshot


class TestVehiclesAPI(unittest.TestCase):
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
        db.query(CCTVSnapshot).delete()
        
        # Add 3 snapshots with different dates
        snap1 = CCTVSnapshot(
            cs_id="bluenetwrks",
            cp_id="BNS00000",
            plate_number="52어0586",
            event_type="START",
            image_path="snapshots/test1.jpg",
            is_ev=True,
            vehicle_type="EV",
            created_at=datetime(2026, 9, 14, 10, 0, 0)
        )
        snap2 = CCTVSnapshot(
            cs_id="bluenetwrks",
            cp_id="BNS00000",
            plate_number="52어0586",
            event_type="END",
            image_path="snapshots/test2.jpg",
            is_ev=True,
            vehicle_type="EV",
            created_at=datetime(2026, 9, 14, 10, 30, 0)
        )
        snap3 = CCTVSnapshot(
            cs_id="bluenetwrks",
            cp_id="BNS99999",
            plate_number="58버6091",
            event_type="START",
            image_path="snapshots/test3.jpg",
            is_ev=False,
            vehicle_type="REGULAR",
            created_at=datetime(2026, 9, 10, 14, 0, 0)
        )
        db.add_all([snap1, snap2, snap3])
        db.commit()
        db.close()

    def test_list_vehicles_date_filter_today(self):
        # Filter for 2026-09-14
        res = self.client.get("/api/v1/vehicles?start_date=2026-09-14&end_date=2026-09-14")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["items"]), 2)

    def test_list_vehicles_date_filter_past(self):
        # Filter for 2026-09-10
        res = self.client.get("/api/v1/vehicles?start_date=2026-09-10&end_date=2026-09-10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["plate_number"], "58버6091")

    def test_export_csv_endpoint(self):
        res = self.client.get("/api/v1/vehicles/export?start_date=2026-09-10&end_date=2026-09-14")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.headers["content-type"])
        content = res.content.decode("utf-8-sig")
        self.assertIn("52어0586", content)
        self.assertIn("58버6091", content)
        self.assertIn("충전기 ID", content)
        self.assertIn("CCTV 명칭", content)
        self.assertIn("차량 번호판", content)
        self.assertNotIn("Stansiya", content)
        self.assertNotIn("Kirish", content)

    def test_vehicle_stay_duration_calculation(self):
        # Retrieve snapshots for 2026-09-14 (has paired START and END session with 30 min duration)
        res = self.client.get("/api/v1/vehicles?start_date=2026-09-14&end_date=2026-09-14")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["items"]), 2)
        # Check both START and END items
        for item in data["items"]:
            self.assertIn("stay_duration_seconds", item)
            self.assertIn("stay_duration_formatted", item)
            self.assertIn("is_ongoing", item)

    def test_ongoing_session_flag(self):
        # Filter for 2026-09-10 (snap3 is START only with no matching END)
        res = self.client.get("/api/v1/vehicles?start_date=2026-09-10&end_date=2026-09-10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        item = data["items"][0]
        self.assertTrue(item["is_ongoing"])
        self.assertGreater(item["stay_duration_seconds"], 0)
        self.assertIsNotNone(item["stay_duration_formatted"])


if __name__ == "__main__":
    unittest.main()
