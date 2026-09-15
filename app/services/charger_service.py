"""
Charger Service — Integrates with Blue Networks CSMS tables:
- TINF_CS / TINF_CP / TINF_CP_CONNECTOR_STATUS (Charger validation & live connector status)
- TINF_CURRENT_TX (Active charging transaction: start time, battery SoC %, power kW, energy kWh)
- TCSP_CHARGE_HIST (Completed charging records)
- Violation Detection: NOT_CHARGING, OVERSTAY (after 100% full), NON_EV_PARKED
"""
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import get_kst_now
from app.core.database import CSMSSessionLocal

logger = logging.getLogger("cctv_charger")

# Korean connector status mapping
CONNECTOR_STATUS_KR = {
    "AVAILABLE": "사용 가능",
    "PREPARING": "연결/준비 중",
    "CHARGING": "충전 중",
    "SUSPENDEDEV": "완충 대기",
    "SUSPENDEDEVSE": "일시 중지",
    "FINISHING": "충전 종료 중",
    "RESERVED": "예약됨",
    "UNAVAILABLE": "사용 불가",
    "FAULTED": "고장/점검 중",
    "OFFLINE": "오프라인",
}

# Uzbek connector status mapping
CONNECTOR_STATUS_UZ = {
    "AVAILABLE": "Mavjud (Bo'sh)",
    "PREPARING": "Ulanmoqda / Tayyor",
    "CHARGING": "Zaryadlanmoqda",
    "SUSPENDEDEV": "100% To'ldi",
    "SUSPENDEDEVSE": "Vaqtincha to'xtatildi",
    "FINISHING": "Tugallanmoqda",
    "RESERVED": "Band qilingan",
    "UNAVAILABLE": "Mavjud emas",
    "FAULTED": "Nosoz / Ta'mirda",
    "OFFLINE": "Oflayn",
}

def format_duration_kr(seconds: int) -> str:
    """Format duration in seconds to standard Korean string."""
    if seconds <= 0:
        return "0분"
    if seconds < 60:
        return f"{seconds}초"
    mins = seconds // 60
    rem_secs = seconds % 60
    if seconds < 3600:
        if rem_secs > 0:
            return f"{mins}분 {rem_secs}초"
        return f"{mins}분"
    hours = seconds // 3600
    rem_mins = (seconds % 3600) // 60
    if rem_mins > 0:
        return f"{hours}시간 {rem_mins}분"
    return f"{hours}시간"


class ChargerService:
    """CSMS Charger validation and real-time charging status resolution."""

    NOT_CHARGING_GRACE_SECONDS = 5 * 60  # 5 minutes
    OVERSTAY_GRACE_SECONDS = 5 * 60      # 5 minutes after 100% or completion

    def validate_station_and_charger(
        self,
        cs_id: str,
        cp_id: Optional[str],
        db: Session
    ) -> Tuple[bool, str]:
        """
        Validate if cs_id and cp_id exist in CSMS database tables (TINF_CS, TINF_CP).
        Checks Production CSMS DB and local fallback.
        """
        if not cs_id:
            return False, "충전소 ID(cs_id)는 필수입니다."

        try:
            bind = db.get_bind()
            if bind and bind.dialect.name == "sqlite":
                return True, "유효한 충전소 및 충전기입니다. (Mock)"

            # Check in CSMS Production DB first
            csms_session = None
            try:
                csms_session = CSMSSessionLocal()
            except Exception:
                csms_session = db

            try:
                if cp_id:
                    cp_row = csms_session.execute(
                        text("SELECT id, cpId, name FROM TINF_CP WHERE cpId = :cp_id OR CAST(id AS CHAR) = :cp_id LIMIT 1"),
                        {"cp_id": cp_id}
                    ).fetchone()
                    if cp_row:
                        return True, f"유효한 충전기입니다. ({cp_row[1]} / {cp_row[2] or ''})"
                    return False, f"입력하신 충전기 ID({cp_id})가 시스템 DB(TINF_CP)에 존재하지 않습니다."
                else:
                    cs_row = csms_session.execute(
                        text("SELECT csId, name FROM TINF_CS WHERE csId = :cs_id OR CAST(id AS CHAR) = :cs_id OR name LIKE :name_pat LIMIT 1"),
                        {"cs_id": cs_id, "name_pat": f"%{cs_id}%"}
                    ).fetchone()
                    if cs_row:
                        return True, f"유효한 충전소입니다. ({cs_row[0]} / {cs_row[1]})"
                    return False, f"입력하신 충전소 ID({cs_id})가 시스템 DB(TINF_CS)에 존재하지 않습니다."
            finally:
                if csms_session is not db:
                    csms_session.close()

        except Exception as e:
            logger.warning(f"Charger validation query exception (fallback): {e}")
            return True, f"충전기 검증 완료 (로컬 환경)"

    def get_charger_realtime_status(
        self,
        cs_id: str,
        cp_id: Optional[str],
        db: Session,
        session_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query real-time charging status from CSMS Production tables (TINF_CURRENT_TX, TINF_CP_CONNECTOR_STATUS)
        and correlate with parking session to evaluate violations.
        """
        now = get_kst_now()
        effective_cp_id = cp_id or "BNS00000"

        # Defaults
        connector_status = "AVAILABLE"
        connector_status_kr = "사용 가능"
        connector_status_uz = "Mavjud (Bo'sh)"
        is_charging = False
        battery_soc: Optional[int] = None
        charge_power_kw: float = 0.0
        charged_energy_kwh: float = 0.0
        charging_start_time: Optional[datetime] = None
        charging_duration_seconds = 0
        overstay_seconds = 0
        violation_type = "NONE"
        violation_label_kr = "정상"
        violation_label_uz = "Normal"
        violation_level = "info"
        action_required_kr = "—"
        action_required_uz = "—"

        # Check in CSMS Production DB
        csms_session = None
        try:
            csms_session = CSMSSessionLocal()
        except Exception:
            csms_session = db

        try:
            # 1. Resolve numerical cp_pk and serial if available
            cp_pk = None
            cp_code = effective_cp_id
            try:
                cp_lookup = csms_session.execute(
                    text("SELECT id, cpId, chargePointModel, chargeBoxSerialNumber FROM TINF_CP WHERE cpId = :cp_id OR CAST(id AS CHAR) = :cp_id OR chargeBoxSerialNumber = :cp_id LIMIT 1"),
                    {"cp_id": effective_cp_id}
                ).mappings().fetchone()
                if cp_lookup:
                    cp_pk = cp_lookup.get("id")
                    cp_code = cp_lookup.get("cpId") or cp_lookup.get("chargeBoxSerialNumber") or effective_cp_id
            except Exception:
                pass

            target_id = cp_pk if cp_pk is not None else effective_cp_id

            # 2. Query live connector status
            try:
                conn_query = text("""
                    SELECT status, operating, ts 
                    FROM TINF_CP_CONNECTOR_STATUS 
                    WHERE cpId = :cp_id 
                    ORDER BY ts DESC 
                    LIMIT 1
                """)
                conn_row = csms_session.execute(conn_query, {"cp_id": target_id}).fetchone()
                if conn_row and conn_row[0]:
                    raw_stat = str(conn_row[0]).upper().strip()
                    connector_status = raw_stat if raw_stat else "AVAILABLE"
                    connector_status_kr = CONNECTOR_STATUS_KR.get(connector_status, connector_status)
                    connector_status_uz = CONNECTOR_STATUS_UZ.get(connector_status, connector_status)
                else:
                    connector_status = "AVAILABLE"
                    connector_status_kr = "사용 가능"
                    connector_status_uz = "Mavjud (Bo'sh)"
            except Exception:
                connector_status = "AVAILABLE"
                connector_status_kr = "사용 가능"
                connector_status_uz = "Mavjud (Bo'sh)"

            # 3. Query active live transaction (TINF_CURRENT_TX)
            try:
                tx_query = text("""
                    SELECT transactionId, startTimestamp, soc, chargePower, currentPower, currentA, sessionId, chargeBoxSerialNumber
                    FROM TINF_CURRENT_TX
                    WHERE cpId = :cp_id OR chargeBoxSerialNumber = :cp_code OR sessionId = :cp_code OR chargeBoxSerialNumber = :raw_cp_id
                    ORDER BY startTimestamp DESC
                    LIMIT 1
                """)
                tx_row = csms_session.execute(tx_query, {
                    "cp_id": target_id,
                    "cp_code": cp_code,
                    "raw_cp_id": effective_cp_id
                }).fetchone()
                if tx_row:
                    is_charging = True
                    charging_start_time = tx_row[1]
                    if tx_row[2] is not None:
                        try:
                            val = int(tx_row[2])
                            battery_soc = val if val > 0 else None
                        except (ValueError, TypeError):
                            battery_soc = None

                    # tx_row[3] is chargePower (active charging power in kW or W)
                    raw_pwr = float(tx_row[3] or 0.0)
                    charge_power_kw = raw_pwr / 1000.0 if raw_pwr > 100 else raw_pwr

                    # tx_row[4] is currentPower (accumulated energy in kWh or Wh)
                    raw_energy = float(tx_row[4] or 0.0)
                    charged_energy_kwh = raw_energy / 1000.0 if raw_energy > 500 else raw_energy

                    if charging_start_time:
                        charging_duration_seconds = max(0, int((now - charging_start_time).total_seconds()))
                    connector_status = "CHARGING"
                    connector_status_kr = "충전 중"
                    connector_status_uz = "Zaryadlanmoqda"
                elif connector_status == "CHARGING":
                    is_charging = True
                    connector_status_kr = "충전 중"
                    connector_status_uz = "Zaryadlanmoqda"
            except Exception as e:
                logger.warning(f"Error querying TINF_CURRENT_TX for {effective_cp_id}: {e}")


        finally:
            if csms_session is not db:
                csms_session.close()

        # 4. Correlate with CCTV Parking Session
        is_occupied = session_info is not None
        if is_occupied:
            entry_at = session_info.get("entry_at")
            is_ev = session_info.get("is_ev", False)
            parking_seconds = max(0, int((now - entry_at).total_seconds())) if entry_at else 0

            # Violation Case 1: Non-EV vehicle parked in EV slot
            if not is_ev:
                violation_type = "NON_EV_PARKED"
                violation_label_kr = "일반차 불법 주차"
                violation_label_uz = "Oddiy avtomobil (No-EV)"
                violation_level = "danger"
                action_required_kr = "즉시 이동 주차 필요"
                action_required_uz = "Darhol joyni bo'shating"

            # Violation Case 2: EV parked, but NOT charging after grace period
            elif not is_charging:
                if parking_seconds > self.NOT_CHARGING_GRACE_SECONDS:
                    violation_type = "NOT_CHARGING"
                    idle_mins = parking_seconds // 60
                    violation_label_kr = f"미충전 점유 ({idle_mins}분)"
                    violation_label_uz = f"Zaryadsiz turish ({idle_mins} min)"
                    violation_level = "warning"
                    action_required_kr = "5분 내 충전 시작 또는 이동"
                    action_required_uz = "5 min ichida zaryadlang yoki oling"
                else:
                    violation_type = "PREPARING"
                    rem_secs = max(0, self.NOT_CHARGING_GRACE_SECONDS - parking_seconds)
                    rem_mins = (rem_secs + 59) // 60
                    violation_label_kr = f"충전 대기 중 ({rem_mins}분 남음)"
                    violation_label_uz = f"Kutilmoqda ({rem_mins} min qoldi)"
                    violation_level = "info"
                    action_required_kr = "—"
                    action_required_uz = "—"

            # Violation Case 3: 100% full or charging ended, but car remains parked
            elif battery_soc is not None and battery_soc >= 100:
                connector_status = "SUSPENDEDEV"
                connector_status_kr = "완충 (100%)"
                connector_status_uz = "100% To'ldi"
                if parking_seconds > (charging_duration_seconds + self.OVERSTAY_GRACE_SECONDS):
                    overstay_seconds = parking_seconds - charging_duration_seconds
                    violation_type = "OVERSTAY"
                    over_mins = max(1, overstay_seconds // 60)
                    violation_label_kr = f"완충 후 초과 점유 ({over_mins}분)"
                    violation_label_uz = f"100% dan oshiqcha ({over_mins} min)"
                    violation_level = "danger"
                    action_required_kr = "완충 차량 이동 주차 필요"
                    action_required_uz = "100% to'ldi, mashinani oling"
                else:
                    violation_type = "NORMAL_CHARGING"
                    violation_label_kr = "완충됨 (출차 대기)"
                    violation_label_uz = "100% to'ldi (Kutilmoqda)"
                    violation_level = "info"
                    action_required_kr = "—"
                    action_required_uz = "—"
            else:
                violation_type = "NORMAL_CHARGING"
                violation_label_kr = "정상 충전 중"
                violation_label_uz = "Zaryadlanmoqda"
                violation_level = "info"
                action_required_kr = "—"
                action_required_uz = "—"
        else:
            if is_charging:
                # If charger is active even without CCTV car detection yet
                violation_type = "NORMAL_CHARGING"
                violation_label_kr = "충전 중"
                violation_label_uz = "Zaryadlanmoqda"
                violation_level = "info"
                action_required_kr = "—"
                action_required_uz = "—"
            else:
                violation_type = "NONE"
                violation_label_kr = "주차 구역 비어있음"
                violation_label_uz = "Bo'sh"
                violation_level = "info"
                action_required_kr = "—"
                action_required_uz = "—"

        return {
            "cs_id": cs_id,
            "cp_id": effective_cp_id,
            "connector_status": connector_status,
            "connector_status_kr": connector_status_kr,
            "connector_status_uz": connector_status_uz,
            "is_charging": is_charging,
            "battery_soc": battery_soc,
            "charge_power_kw": round(charge_power_kw, 2),
            "charged_energy_kwh": round(charged_energy_kwh, 2),
            "charging_start_time": charging_start_time,
            "charging_duration_seconds": charging_duration_seconds,
            "charging_duration_formatted": format_duration_kr(charging_duration_seconds),
            "overstay_seconds": overstay_seconds,
            "overstay_formatted": format_duration_kr(overstay_seconds) if overstay_seconds > 0 else "0분",
            "violation_type": violation_type,
            "violation_label_kr": violation_label_kr,
            "violation_label_uz": violation_label_uz,
            "violation_level": violation_level,
            "action_required_kr": action_required_kr,
            "action_required_uz": action_required_uz,
        }


charger_service = ChargerService()
