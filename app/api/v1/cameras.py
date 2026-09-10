"""Camera management API."""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.snapshot import CCTVCamera
from app.schemas.snapshot import CameraCreate, CameraUpdate, CameraResponse

router = APIRouter()


@router.get("", response_model=List[CameraResponse], summary="List cameras")
def list_cameras(
    cs_id: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(CCTVCamera)
    if cs_id:
        query = query.filter(CCTVCamera.cs_id == cs_id)
    if is_active is not None:
        query = query.filter(CCTVCamera.is_active.is_(is_active))
    return query.all()


@router.post("", response_model=CameraResponse, summary="Add camera")
def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    cam = CCTVCamera(
        cs_id=camera_in.cs_id,
        cp_id=camera_in.cp_id,
        camera_name=camera_in.camera_name,
        camera_type=camera_in.camera_type.value,
        stream_url=camera_in.stream_url,
        ip_address=camera_in.ip_address,
        port=camera_in.port,
        username=camera_in.username,
        password=camera_in.password,
        is_active=camera_in.is_active,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@router.put("/{camera_id}", response_model=CameraResponse, summary="Update camera")
def update_camera(camera_id: int, camera_in: CameraUpdate, db: Session = Depends(get_db)):
    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    update_data = camera_in.model_dump(exclude_unset=True)
    if 'camera_type' in update_data and update_data['camera_type']:
        update_data['camera_type'] = update_data['camera_type'].value
    for key, value in update_data.items():
        setattr(cam, key, value)
    cam.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(cam)
    return cam


@router.delete("/{camera_id}", summary="Delete camera")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(cam)
    db.commit()
    return {"status": "success", "message": f"Camera {camera_id} deleted"}


@router.get("/{camera_id}", response_model=CameraResponse, summary="Get camera")
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.post("/{camera_id}/test", summary="Test camera live frame capture")
async def test_camera(camera_id: int, db: Session = Depends(get_db)):
    import base64
    from app.services.camera_service import camera_service
    from app.schemas.snapshot import CameraTypeEnum
    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    result = await camera_service.capture_snapshot(
        camera_type=CameraTypeEnum(cam.camera_type),
        stream_url=cam.stream_url,
        cs_id=cam.cs_id,
        cp_id=cam.cp_id or "CP01",
        username=cam.username,
        password=cam.password,
        ip_address=cam.ip_address,
        port=cam.port
    )
    if not result.success or not result.image_bytes:
        return {
            "success": False,
            "message": result.error_message or "Camera capture failed",
            "protocol": result.protocol
        }
    
    b64_img = base64.b64encode(result.image_bytes).decode("utf-8")
    return {
        "success": True,
        "message": f"Frame captured successfully ({result.width}x{result.height})",
        "protocol": result.protocol,
        "image_base64": b64_img
    }

