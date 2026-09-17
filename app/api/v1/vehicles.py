"""Vehicle log API — list, view, export, delete detected vehicles."""
import csv
import io
import re
import base64
from datetime import datetime, date
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.config import get_kst_now
from app.core.database import get_db
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.schemas.snapshot import SnapshotResponse, VehicleTypeEnum
from app.services.monitor_service import auto_monitor_service
from app.services.charger_service import charger_service, format_duration_kr
from app.services.storage_service import storage_service

router = APIRouter()


def _parse_date_bounds(start_date_str: Optional[str], end_date_str: Optional[str]):
    """Helper to parse YYYY-MM-DD date strings into start and end datetimes."""
    from_dt = None
    to_dt = None
    if start_date_str:
        try:
            clean_str = start_date_str.split("T")[0]
            from_dt = datetime.strptime(f"{clean_str} 00:00:00", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    if end_date_str:
        try:
            clean_str = end_date_str.split("T")[0]
            to_dt = datetime.strptime(f"{clean_str} 23:59:59", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return from_dt, to_dt


def _format_korean_notes(notes: Optional[str], event_type: str) -> str:
    """Format stored duration and note strings cleanly into pure Korean."""
    if not notes:
        return "출차 완료" if event_type == "END" else ("입차 기록" if event_type == "START" else "-")
    t = notes
    if "진행 중" in t:
        return t
    if "boshlandi" in t.lower() or event_type == "START":
        return "입차 (주차 시작)"
    t = re.sub(r"^Chiqish:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^Kirish:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Jami\s*", "총 ", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*kun", r"\1일 ", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*soat", r"\1시간 ", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*daqiqa", r"\1분 ", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*soniya", r"\1초", t, flags=re.IGNORECASE)
    t = re.sub(r"to['’]xtab turdi", "주차", t, flags=re.IGNORECASE)
    t = re.sub(r"To['’]xtab turish:\s*", "총 ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip() or ("출차 완료" if event_type == "END" else "입차 기록")



@router.get("", summary="List detected vehicles with filtering")
def list_vehicles(
    plate_number: Optional[str] = Query(None),
    vehicle_type: Optional[str] = Query(None),
    is_ev: Optional[bool] = Query(None),
    cs_id: Optional[str] = Query(None),
    violation_filter: Optional[str] = Query(None, description="ALL, NON_EV, NOT_CHARGING, OVERSTAY, NORMAL"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(CCTVSnapshot)

    if plate_number:
        query = query.filter(CCTVSnapshot.plate_number.ilike(f"%{plate_number}%"))
    if vehicle_type:
        query = query.filter(CCTVSnapshot.vehicle_type == vehicle_type)
    if is_ev is not None:
        query = query.filter(CCTVSnapshot.is_ev == is_ev)
    if cs_id:
        query = query.filter(CCTVSnapshot.cs_id == cs_id)

    # Violation quick filters
    if violation_filter == "NON_EV":
        query = query.filter(CCTVSnapshot.is_ev.is_(False))
    elif violation_filter == "EV":
        query = query.filter(CCTVSnapshot.is_ev.is_(True))

    # Date filtering
    dt_from, dt_to = _parse_date_bounds(start_date, end_date)
    if dt_from:
        query = query.filter(CCTVSnapshot.created_at >= dt_from)
    elif from_date:
        query = query.filter(CCTVSnapshot.created_at >= from_date)

    if dt_to:
        query = query.filter(CCTVSnapshot.created_at <= dt_to)
    elif to_date:
        query = query.filter(CCTVSnapshot.created_at <= to_date)

    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(desc(CCTVSnapshot.created_at)).offset(offset).limit(page_size).all()

    # Pre-fetch camera names map
    cameras = db.query(CCTVCamera).all()
    cam_name_map = {}
    for idx, cam in enumerate(cameras, start=1):
        cam_key = (cam.cs_id, cam.cp_id or "BNS00000")
        cam_name = cam.camera_name or f"CCTV {idx}"
        cam_name_map[cam_key] = cam_name

    # Pre-fetch session records to compute exact durations
    session_ids = [s.session_id for s in items if s.session_id]
    session_start_map = {}
    session_end_map = {}
    if session_ids:
        related_snaps = db.query(CCTVSnapshot).filter(CCTVSnapshot.session_id.in_(set(session_ids))).all()
        for rs in related_snaps:
            if rs.event_type == "START" and rs.session_id not in session_start_map:
                session_start_map[rs.session_id] = rs
            elif rs.event_type == "END" and rs.session_id not in session_end_map:
                session_end_map[rs.session_id] = rs

    now_kst = get_kst_now()
    results = []
    for snap in items:
        item = SnapshotResponse.model_validate(snap)
        item.image_url = f"/api/v1/vehicles/{snap.id}/image"
        item.download_url = f"/api/v1/vehicles/{snap.id}/download"
        
        # Camera name resolution
        cam_key = (snap.cs_id, snap.cp_id or "BNS00000")
        item.camera_name = cam_name_map.get(cam_key, f"CCTV ({snap.cp_id or '1'})")

        # Active session correlation from memory
        matched_active = None
        for cam_key_str, s_dict in auto_monitor_service.active_sessions.items():
            if snap.session_id and s_dict.get("session_id") == snap.session_id:
                matched_active = s_dict
                break
            if snap.cs_id and snap.cp_id and s_dict.get("cs_id") == snap.cs_id and s_dict.get("cp_id") == snap.cp_id:
                matched_active = s_dict
                break

        # Calculate stay duration
        if snap.session_id and snap.session_id in session_end_map:
            # Session has completed (has END event)
            end_snap = session_end_map[snap.session_id]
            start_snap = session_start_map.get(snap.session_id, snap)
            dur_sec = max(0, int((end_snap.created_at - start_snap.created_at).total_seconds()))
            item.is_ongoing = False
            item.stay_duration_seconds = dur_sec
            item.stay_duration_formatted = format_duration_kr(dur_sec)
        elif snap.event_type == "START":
            # Session is ongoing
            item.is_ongoing = True
            entry_time = matched_active.get("entry_at", snap.created_at) if matched_active else snap.created_at
            ongoing_sec = max(0, int((now_kst - entry_time).total_seconds()))
            item.stay_duration_seconds = ongoing_sec
            item.stay_duration_formatted = format_duration_kr(ongoing_sec)
            item.notes = f"입차 (주차 진행 중: {item.stay_duration_formatted})"
        elif snap.event_type == "END":
            item.is_ongoing = False
            start_snap = session_start_map.get(snap.session_id)
            if start_snap:
                dur_sec = max(0, int((snap.created_at - start_snap.created_at).total_seconds()))
            else:
                dur_sec = 0
            item.stay_duration_seconds = dur_sec
            item.stay_duration_formatted = format_duration_kr(dur_sec)
        # Fallback and normalize plate_region_image for END events from session's START event
        if not item.plate_region_image and snap.session_id and snap.session_id in session_start_map:
            item.plate_region_image = session_start_map[snap.session_id].plate_region_image
        if item.plate_region_image and not item.plate_region_image.startswith("data:image") and len(item.plate_region_image) > 60 and not ("/" in item.plate_region_image or "\\" in item.plate_region_image):
            item.plate_region_image = f"data:image/jpeg;base64,{item.plate_region_image}"

        # Determine violation & action labels
        if not snap.is_ev:
            item.violation_type = "NON_EV_PARKED"
            item.violation_label_kr = "일반차 불법 주차"
            item.violation_label_uz = "Oddiy avtomobil (No-EV)"
            item.action_required_kr = "즉시 이동 주차 필요" if snap.event_type != "END" else "출차 완료"
            item.action_required_uz = "Darhol joyni bo'shating" if snap.event_type != "END" else "Chiqib ketgan"
            item.action_required = item.action_required_kr
        else:
            if snap.event_type != "END":
                effective_session = matched_active or {
                    "entry_at": snap.created_at,
                    "is_ev": snap.is_ev,
                    "plate": snap.plate_number,
                    "session_id": snap.session_id,
                    "cs_id": snap.cs_id,
                    "cp_id": snap.cp_id
                }
                chg_status = charger_service.get_charger_realtime_status(
                    cs_id=snap.cs_id, cp_id=snap.cp_id, db=db, session_info=effective_session
                )
                item.battery_soc = chg_status.get("battery_soc")
                item.charge_power_kw = chg_status.get("charge_power_kw")
                item.violation_type = chg_status.get("violation_type", "NORMAL_CHARGING")
                item.violation_label_kr = chg_status.get("violation_label_kr", "정상")
                item.violation_label_uz = chg_status.get("violation_label_uz", "Normal")
                item.action_required_kr = chg_status.get("action_required_kr", "—")
                item.action_required_uz = chg_status.get("action_required_uz", "—")
            else:
                item.violation_type = "NORMAL_CHARGING"
                item.violation_label_kr = "정상"
                item.violation_label_uz = "Normal"
                item.action_required_kr = "출차 완료"
                item.action_required_uz = "Chiqib ketgan"
            item.action_required = item.action_required_kr

        results.append(item)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": results
    }


@router.get("/export", summary="Export vehicles log as CSV/Excel in clean Korean")
def export_vehicles_csv(
    plate_number: Optional[str] = Query(None),
    vehicle_type: Optional[str] = Query(None),
    is_ev: Optional[bool] = Query(None),
    cs_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(CCTVSnapshot)

    if plate_number:
        query = query.filter(CCTVSnapshot.plate_number.ilike(f"%{plate_number}%"))
    if vehicle_type:
        query = query.filter(CCTVSnapshot.vehicle_type == vehicle_type)
    if is_ev is not None:
        query = query.filter(CCTVSnapshot.is_ev == is_ev)
    if cs_id:
        query = query.filter(CCTVSnapshot.cs_id == cs_id)

    dt_from, dt_to = _parse_date_bounds(start_date, end_date)
    if dt_from:
        query = query.filter(CCTVSnapshot.created_at >= dt_from)
    if dt_to:
        query = query.filter(CCTVSnapshot.created_at <= dt_to)

    items = query.order_by(desc(CCTVSnapshot.created_at)).limit(5000).all()

    # Pre-fetch camera names map
    cameras = db.query(CCTVCamera).all()
    cam_name_map = {}
    for idx, cam in enumerate(cameras, start=1):
        cam_key = (cam.cs_id, cam.cp_id or "BNS00000")
        cam_name = cam.camera_name or f"CCTV {idx}"
        cam_name_map[cam_key] = cam_name

    # Build CSV with UTF-8 BOM for Excel
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)

    # Pure Korean Header row matching compact table
    writer.writerow([
        "ID",
        "감지 일시 (KST)",
        "충전기 ID (CP ID)",
        "CCTV 명칭",
        "차량 번호판",
        "구분",
        "차량 유형",
        "전기차 여부",
        "인식 정확도",
        "배터리 SoC & 충전 전력",
        "위반 상태",
        "조치 필요 사항",
        "주차 시간 및 비고"
    ])

    # Pre-fetch session records to compute exact durations for export
    session_ids = [s.session_id for s in items if s.session_id]
    session_start_map = {}
    session_end_map = {}
    if session_ids:
        related_snaps = db.query(CCTVSnapshot).filter(CCTVSnapshot.session_id.in_(set(session_ids))).all()
        for rs in related_snaps:
            if rs.event_type == "START" and rs.session_id not in session_start_map:
                session_start_map[rs.session_id] = rs
            elif rs.event_type == "END" and rs.session_id not in session_end_map:
                session_end_map[rs.session_id] = rs

    now_kst = get_kst_now()

    for item in items:
        event_label = "입차" if item.event_type == "START" else ("출차" if item.event_type == "END" else item.event_type)
        type_label = "전기차 (EV)" if item.is_ev else ("일반차" if item.vehicle_type == "REGULAR" else item.vehicle_type)
        ev_label = "전기차" if item.is_ev else "일반차"
        conf_str = f"{(item.ai_confidence * 100):.0f}%" if item.ai_confidence is not None else "-"
        time_str = item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else ""

        # Exact duration string
        if item.session_id and item.session_id in session_end_map:
            end_snap = session_end_map[item.session_id]
            start_snap = session_start_map.get(item.session_id, item)
            dur_sec = max(0, int((end_snap.created_at - start_snap.created_at).total_seconds()))
            dur_text = f"총 {format_duration_kr(dur_sec)} 주차"
        elif item.event_type == "START":
            ongoing_sec = max(0, int((now_kst - item.created_at).total_seconds()))
            dur_text = f"주차 진행 중 ({format_duration_kr(ongoing_sec)})"
        elif item.event_type == "END":
            start_snap = session_start_map.get(item.session_id)
            if start_snap:
                dur_sec = max(0, int((item.created_at - start_snap.created_at).total_seconds()))
                dur_text = f"총 {format_duration_kr(dur_sec)} 주차"
            else:
                dur_text = _format_korean_notes(item.notes, item.event_type)
        else:
            dur_text = _format_korean_notes(item.notes, item.event_type)

        cam_key = (item.cs_id, item.cp_id or "BNS00000")
        cctv_name = cam_name_map.get(cam_key, f"CCTV ({item.cp_id or '1'})")

        # Battery SoC & Charging power info
        battery_power_str = "—"
        if not item.is_ev:
            viol_label = "일반차 불법 주차"
            act_label = "즉시 이동 주차 필요" if item.event_type != "END" else "출차 완료"
        else:
            if item.event_type != "END":
                chg_status = charger_service.get_charger_realtime_status(
                    cs_id=item.cs_id, cp_id=item.cp_id, db=db, session_info={"entry_at": item.created_at, "is_ev": True}
                )
                soc = chg_status.get("battery_soc")
                pwr = chg_status.get("charge_power_kw")
                if soc is not None and soc > 0:
                    battery_power_str = f"{soc}%" + (f" ({pwr} kW)" if pwr else "")
                elif pwr and pwr > 0:
                    battery_power_str = f"{pwr} kW"
                viol_label = chg_status.get("violation_label_kr", "정상")
                act_label = chg_status.get("action_required_kr", "—")
            else:
                viol_label = "정상"
                act_label = "출차 완료"

        writer.writerow([
            item.id,
            time_str,
            item.cp_id or "BNS00000",
            cctv_name,
            item.plate_number or "미인식",
            event_label,
            type_label,
            ev_label,
            conf_str,
            battery_power_str,
            viol_label,
            act_label,
            dur_text
        ])

    csv_data = output.getvalue().encode("utf-8-sig")
    s_date = start_date or "all"
    e_date = end_date or "all"
    filename = f"cctv_vehicles_report_{s_date}_to_{e_date}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/{vehicle_id}", response_model=SnapshotResponse)
def get_vehicle(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Vehicle record not found")
    item = SnapshotResponse.model_validate(snap)
    item.image_url = f"/api/v1/vehicles/{snap.id}/image"
    return item


def _resolve_snapshot_image_bytes(snap: CCTVSnapshot, db: Session) -> Optional[bytes]:
    """Helper to resolve image bytes from disk file, base64 field, or paired START session."""
    # 1. Try disk full image file
    if snap.image_path:
        abs_path = storage_service.get_absolute_path(snap.image_path)
        if abs_path.exists():
            try:
                with open(abs_path, "rb") as f:
                    return f.read()
            except Exception:
                pass

    # 2. Check if plate_region_image contains Base64 data URL
    if snap.plate_region_image:
        pri = snap.plate_region_image.strip()
        if pri.startswith("data:image"):
            comma_idx = pri.find(",")
            if comma_idx != -1:
                try:
                    data = base64.b64decode(pri[comma_idx + 1:], validate=True)
                    if len(data) >= 20:
                        return data
                except Exception:
                    pass
        elif len(pri) > 60 and not (pri.endswith(".jpg") or pri.endswith(".png") or "/" in pri or "\\" in pri):
            # Raw base64 string
            try:
                data = base64.b64decode(pri, validate=True)
                if len(data) >= 20:
                    return data
            except Exception:
                pass
        else:
            # File path on disk
            crop_path = storage_service.get_absolute_path(pri)
            if crop_path.exists():
                try:
                    with open(crop_path, "rb") as f:
                        return f.read()
                except Exception:
                    pass

    # 3. If this is an END event, fall back to the session's START event image
    if snap.event_type == "END" and snap.session_id:
        start_snap = db.query(CCTVSnapshot).filter(
            CCTVSnapshot.session_id == snap.session_id,
            CCTVSnapshot.event_type == "START"
        ).first()
        if start_snap and start_snap.id != snap.id:
            return _resolve_snapshot_image_bytes(start_snap, db)

    return None


@router.get("/{vehicle_id}/image", summary="View vehicle image with base64 and session fallback")
def view_vehicle_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Vehicle record not found")
    
    img_bytes = _resolve_snapshot_image_bytes(snap, db)
    if img_bytes:
        return Response(content=img_bytes, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
    
    raise HTTPException(status_code=404, detail="Image file not found")


@router.get("/{vehicle_id}/plate-image", summary="View cropped plate image with base64 support")
def view_plate_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Vehicle record not found")
    
    # Check base64 in plate_region_image first
    if snap.plate_region_image:
        pri = snap.plate_region_image.strip()
        if pri.startswith("data:image"):
            comma_idx = pri.find(",")
            if comma_idx != -1:
                try:
                    img_bytes = base64.b64decode(pri[comma_idx + 1:], validate=True)
                    if len(img_bytes) >= 20:
                        return Response(content=img_bytes, media_type="image/jpeg")
                except Exception:
                    pass
        elif len(pri) > 60 and not (pri.endswith(".jpg") or pri.endswith(".png") or "/" in pri or "\\" in pri):
            try:
                img_bytes = base64.b64decode(pri, validate=True)
                if len(img_bytes) >= 20:
                    return Response(content=img_bytes, media_type="image/jpeg")
            except Exception:
                pass
        else:
            abs_path = storage_service.get_absolute_path(pri)
            if abs_path.exists():
                return FileResponse(path=str(abs_path), media_type="image/jpeg")

    # Fallback to full frame image bytes
    img_bytes = _resolve_snapshot_image_bytes(snap, db)
    if img_bytes:
        return Response(content=img_bytes, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Plate image not found")


@router.get("/{vehicle_id}/download", summary="Download vehicle image")
def download_vehicle_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Vehicle record not found")
    
    img_bytes = _resolve_snapshot_image_bytes(snap, db)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="Image file not found")
    
    clean_plate = re.sub(r'[^a-zA-Z0-9_-]', '', snap.plate_number or '') or f"vehicle_{snap.id}"
    date_str = snap.created_at.strftime("%Y%m%d_%H%M%S") if snap.created_at else "snapshot"
    filename = f"{clean_plate}_{snap.vehicle_type}_{date_str}.jpg"
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.delete("/{vehicle_id}", summary="Delete vehicle record")
def delete_vehicle(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(snap)
    db.commit()
    return {"status": "success", "message": f"Record {vehicle_id} deleted"}
