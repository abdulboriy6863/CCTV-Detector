"""Vehicle log API — list, view, export, delete detected vehicles."""
import csv
import io
from datetime import datetime, date
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.database import get_db
from app.models.snapshot import CCTVSnapshot
from app.schemas.snapshot import SnapshotResponse, VehicleTypeEnum
from app.services.storage_service import storage_service

router = APIRouter()


def _parse_date_bounds(start_date_str: Optional[str], end_date_str: Optional[str]):
    """Helper to parse YYYY-MM-DD date strings into start and end datetimes."""
    from_dt = None
    to_dt = None
    if start_date_str:
        try:
            # Handle YYYY-MM-DD or full ISO
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


@router.get("", summary="List detected vehicles with filtering")
def list_vehicles(
    plate_number: Optional[str] = Query(None),
    vehicle_type: Optional[str] = Query(None),
    is_ev: Optional[bool] = Query(None),
    cs_id: Optional[str] = Query(None),
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

    results = []
    for snap in items:
        item = SnapshotResponse.model_validate(snap)
        item.image_url = f"/api/v1/vehicles/{snap.id}/image"
        item.download_url = f"/api/v1/vehicles/{snap.id}/download"
        results.append(item)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": results
    }


@router.get("/export", summary="Export vehicles log as CSV/Excel")
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

    # Build CSV with UTF-8 BOM for Excel
    output = io.StringIO()
    # Write UTF-8 BOM
    output.write("\ufeff")
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "ID",
        "Vaqt / 시간 (KST)",
        "Stansiya / 충전소 (CS ID)",
        "Zaryadka Joyi / 베이 (CP ID)",
        "Davlat Raqami / 차량 번호판",
        "Hodisa / 이벤트",
        "Turi / 차량 유형",
        "EV Tasdiqlandi / 전기차 여부",
        "Ishonch / 정확도",
        "Izoh / 비고",
        "Aniqlash Manbai / 감지 소스"
    ])

    for item in items:
        event_label = "Kirish (입차)" if item.event_type == "START" else ("Chiqish (출차)" if item.event_type == "END" else item.event_type)
        type_label = "EV (전기차)" if item.is_ev else ("Oddiy (일반차)" if item.vehicle_type == "REGULAR" else item.vehicle_type)
        conf_str = f"{(item.ai_confidence * 100):.0f}%" if item.ai_confidence is not None else "-"
        time_str = item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else ""

        writer.writerow([
            item.id,
            time_str,
            item.cs_id or "bluenetwrks",
            item.cp_id or "BNS00000",
            item.plate_number or "Noma'lum",
            event_label,
            type_label,
            "Ha (예)" if item.is_ev else "Yo'q (아니오)",
            conf_str,
            item.notes or "",
            item.detection_source or ""
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


@router.get("/{vehicle_id}/image", summary="View vehicle image")
def view_vehicle_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Not found")
    abs_path = storage_service.get_absolute_path(snap.image_path)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path=str(abs_path), media_type="image/jpeg")


@router.get("/{vehicle_id}/plate-image", summary="View cropped plate image")
def view_plate_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap or not snap.plate_region_image:
        raise HTTPException(status_code=404, detail="Plate image not found")
    abs_path = storage_service.get_absolute_path(snap.plate_region_image)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Plate image file not found")
    return FileResponse(path=str(abs_path), media_type="image/jpeg")


@router.get("/{vehicle_id}/download", summary="Download vehicle image")
def download_vehicle_image(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Not found")
    abs_path = storage_service.get_absolute_path(snap.image_path)
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    filename = f"{snap.plate_number or 'UNKNOWN'}_{snap.vehicle_type}_{snap.created_at.strftime('%Y%m%d_%H%M%S')}.jpg"
    return FileResponse(path=str(abs_path), media_type="image/jpeg", filename=filename,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.delete("/{vehicle_id}", summary="Delete vehicle record")
def delete_vehicle(vehicle_id: int, db: Session = Depends(get_db)):
    snap = db.query(CCTVSnapshot).filter(CCTVSnapshot.id == vehicle_id).first()
    if not snap:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(snap)
    db.commit()
    return {"status": "success", "message": f"Record {vehicle_id} deleted"}
