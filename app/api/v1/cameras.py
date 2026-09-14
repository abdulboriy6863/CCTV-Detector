"""Camera management API."""
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.snapshot import CCTVCamera
from app.schemas.snapshot import CameraCreate, CameraUpdate, CameraResponse, CameraTypeEnum
from app.services.camera_service import camera_service

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
async def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    stream_url = camera_in.stream_url
    if camera_in.ip_address:
        if not stream_url or camera_in.ip_address not in stream_url:
            if camera_in.camera_type == CameraTypeEnum.RTSP:
                port_val = camera_in.port or 554
                stream_url = f"rtsp://{camera_in.ip_address}:{port_val}/stream1"
            else:
                port_val = camera_in.port or 80
                port_str = f":{port_val}" if port_val != 80 else ""
                stream_url = f"http://{camera_in.ip_address}{port_str}/api/snapshot"

    final_stream_url = stream_url or f"rtsp://{camera_in.ip_address or '127.0.0.1'}:554/stream1"

    # Validate reachability of the camera before saving to DB
    is_reachable, reachability_msg = await camera_service.check_camera_reachability(
        camera_type=camera_in.camera_type,
        stream_url=final_stream_url,
        username=camera_in.username,
        password=camera_in.password,
        ip_address=camera_in.ip_address,
        port=camera_in.port or (554 if camera_in.camera_type == CameraTypeEnum.RTSP else 80),
        timeout_seconds=3.0
    )
    if not is_reachable:
        raise HTTPException(
            status_code=400,
            detail=f"Kameraga ulanib bo'lmadi: {reachability_msg}"
        )

    cam = CCTVCamera(
        cs_id=camera_in.cs_id,
        cp_id=camera_in.cp_id,
        camera_name=camera_in.camera_name,
        camera_type=camera_in.camera_type.value,
        stream_url=final_stream_url,
        ip_address=camera_in.ip_address,
        port=camera_in.port or (554 if camera_in.camera_type.value == "RTSP" else 80),
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
async def update_camera(camera_id: int, camera_in: CameraUpdate, db: Session = Depends(get_db)):
    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = camera_in.model_dump(exclude_unset=True)

    # If stream_url or ip_address changed, validate reachability
    test_type = CameraTypeEnum(update_data.get('camera_type', cam.camera_type))
    test_ip = update_data.get('ip_address', cam.ip_address)
    test_port = update_data.get('port', cam.port)
    test_url = update_data.get('stream_url', cam.stream_url)
    test_user = update_data.get('username', cam.username)
    test_pass = update_data.get('password', cam.password)

    if 'ip_address' in update_data or 'stream_url' in update_data:
        is_reachable, reachability_msg = await camera_service.check_camera_reachability(
            camera_type=test_type,
            stream_url=test_url,
            username=test_user,
            password=test_pass,
            ip_address=test_ip,
            port=test_port,
            timeout_seconds=3.0
        )
        if not is_reachable:
            raise HTTPException(
                status_code=400,
                detail=f"Kameraga ulanib bo'lmadi: {reachability_msg}"
            )

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


@router.get("/{camera_id}/stream", summary="Live continuous MJPEG video stream")
async def stream_camera(camera_id: int, db: Session = Depends(get_db)):
    """
    Streams continuous live video (MJPEG) from CCTV RTSP/HTTP camera.
    Compatible with standard <img> tags in any browser.
    """
    from fastapi.responses import StreamingResponse
    from app.services.camera_service import camera_service
    from app.schemas.snapshot import CameraTypeEnum

    cam = db.query(CCTVCamera).filter(CCTVCamera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    stream_generator = camera_service.get_live_stream(
        camera_type=CameraTypeEnum(cam.camera_type),
        stream_url=cam.stream_url,
        username=cam.username,
        password=cam.password,
        ip_address=cam.ip_address,
        port=cam.port
    )
    return StreamingResponse(
        stream_generator,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


