"""Dashboard statistics API."""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import get_kst_now
from app.core.database import get_db
from app.models.snapshot import CCTVSnapshot, CCTVCamera

router = APIRouter()


@router.get("/summary", summary="Dashboard statistics summary")
def get_stats(db: Session = Depends(get_db)):
    today_start = get_kst_now().replace(hour=0, minute=0, second=0, microsecond=0)

    total = db.query(func.count(CCTVSnapshot.id)).scalar() or 0
    ev_total = db.query(func.count(CCTVSnapshot.id)).filter(CCTVSnapshot.is_ev == True).scalar() or 0
    regular_total = db.query(func.count(CCTVSnapshot.id)).filter(CCTVSnapshot.vehicle_type == "REGULAR").scalar() or 0

    today_total = db.query(func.count(CCTVSnapshot.id)).filter(CCTVSnapshot.created_at >= today_start).scalar() or 0
    today_ev = db.query(func.count(CCTVSnapshot.id)).filter(
        CCTVSnapshot.created_at >= today_start, CCTVSnapshot.is_ev == True
    ).scalar() or 0
    today_regular = db.query(func.count(CCTVSnapshot.id)).filter(
        CCTVSnapshot.created_at >= today_start, CCTVSnapshot.vehicle_type == "REGULAR"
    ).scalar() or 0
    today_alerts = db.query(func.count(CCTVSnapshot.id)).filter(
        CCTVSnapshot.created_at >= today_start, CCTVSnapshot.alert_sent == True
    ).scalar() or 0

    active_cameras = db.query(func.count(CCTVCamera.id)).filter(CCTVCamera.is_active == True).scalar() or 0

    return {
        "total_detections": total,
        "ev_count": ev_total,
        "regular_count": regular_total,
        "today_detections": today_total,
        "today_ev": today_ev,
        "today_regular": today_regular,
        "today_alerts": today_alerts,
        "active_cameras": active_cameras
    }
