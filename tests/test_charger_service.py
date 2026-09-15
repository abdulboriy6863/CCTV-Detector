"""Unit tests for ChargerService and violation logic."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import Base
from app.core.config import get_kst_now
from app.services.charger_service import charger_service, format_duration_kr


class TestChargerService(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_duration_formatting_kr(self):
        self.assertEqual(format_duration_kr(0), "0분")
        self.assertEqual(format_duration_kr(45), "45초")
        self.assertEqual(format_duration_kr(60), "1분")
        self.assertEqual(format_duration_kr(130), "2분 10초")
        self.assertEqual(format_duration_kr(3600), "1시간")
        self.assertEqual(format_duration_kr(5400), "1시간 30분")

    def test_validate_station_and_charger_empty(self):
        valid, msg = charger_service.validate_station_and_charger("", "CP01", self.db)
        self.assertFalse(valid)
        self.assertIn("필수", msg)

    def test_vacant_slot_status(self):
        status = charger_service.get_charger_realtime_status(
            cs_id="CS01",
            cp_id="CP01",
            db=self.db,
            session_info=None
        )
        self.assertEqual(status["violation_type"], "NONE")
        self.assertEqual(status["connector_status"], "AVAILABLE")
        self.assertEqual(status["connector_status_kr"], "사용 가능")

    def test_non_ev_violation(self):
        now = get_kst_now()
        # Non-EV vehicle: immediate violation even at 0 or 1 minute
        session_info = {
            "entry_at": now - timedelta(minutes=1),
            "is_ev": False,
            "plate": "12가3456"
        }
        status = charger_service.get_charger_realtime_status(
            cs_id="CS01",
            cp_id="CP01",
            db=self.db,
            session_info=session_info
        )
        self.assertEqual(status["violation_type"], "NON_EV_PARKED")
        self.assertEqual(status["violation_label_kr"], "일반차 불법 주차")
        self.assertEqual(status["violation_label_uz"], "Oddiy avtomobil (No-EV)")
        self.assertEqual(status["violation_level"], "danger")
        self.assertEqual(status["action_required_kr"], "즉시 이동 주차 필요")
        self.assertEqual(status["action_required_uz"], "Darhol joyni bo'shating")

    def test_ev_not_charging_grace_period(self):
        now = get_kst_now()
        # Parked 2 minutes ago (within 5 min grace period)
        session_info = {
            "entry_at": now - timedelta(minutes=2),
            "is_ev": True,
            "plate": "81머2072"
        }
        status = charger_service.get_charger_realtime_status(
            cs_id="CS01",
            cp_id="CP01",
            db=self.db,
            session_info=session_info
        )
        self.assertEqual(status["violation_type"], "PREPARING")
        self.assertIn("충전 대기 중", status["violation_label_kr"])
        self.assertEqual(status["violation_level"], "info")

    def test_ev_not_charging_violation_after_5_mins(self):
        now = get_kst_now()
        # Parked 6 minutes ago (exceeded 5 min grace period)
        session_info = {
            "entry_at": now - timedelta(minutes=6),
            "is_ev": True,
            "plate": "81머2072"
        }
        status = charger_service.get_charger_realtime_status(
            cs_id="CS01",
            cp_id="CP01",
            db=self.db,
            session_info=session_info
        )
        self.assertEqual(status["violation_type"], "NOT_CHARGING")
        self.assertIn("미충전 점유", status["violation_label_kr"])
        self.assertIn("Zaryadsiz turish", status["violation_label_uz"])
        self.assertEqual(status["violation_level"], "warning")
        self.assertEqual(status["action_required_kr"], "5분 내 충전 시작 또는 이동")
        self.assertEqual(status["action_required_uz"], "5 min ichida zaryadlang yoki oling")


if __name__ == "__main__":
    unittest.main()
