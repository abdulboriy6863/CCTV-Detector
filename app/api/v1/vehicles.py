"""Vehicle log API — list, view, delete detected vehicles."""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.database import get_db
from app.models.snapshot import CCTVSnapshot
from app.schemas.snapshot import SnapshotResponse, VehicleTypeEnum
from app.services.storage_service import storage_service

router = APIRouter()


@router.get("", summary="List detected vehicles with filtering")
def list_vehicles(
    plate_number: Optional[str] = Query(None),
    vehicle_type: Optional[str] = Query(None),
    is_ev: Optional[bool] = Query(None),
    cs_id: Optional[str] = Query(None),
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
    if from_date:
        query = query.filter(CCTVSnapshot.created_at >= from_date)
    if to_date:
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
