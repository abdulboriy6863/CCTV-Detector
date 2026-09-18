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

    from sqlalchemy import case, and_
    row = db.query(
        func.count(CCTVSnapshot.id).label("total"),
        func.sum(case((CCTVSnapshot.is_ev == True, 1), else_=0)).label("ev_total"),
        func.sum(case((CCTVSnapshot.vehicle_type == "REGULAR", 1), else_=0)).label("regular_total"),
        func.sum(case((CCTVSnapshot.created_at >= today_start, 1), else_=0)).label("today_total"),
        func.sum(case((and_(CCTVSnapshot.created_at >= today_start, CCTVSnapshot.is_ev == True), 1), else_=0)).label("today_ev"),
        func.sum(case((and_(CCTVSnapshot.created_at >= today_start, CCTVSnapshot.vehicle_type == "REGULAR"), 1), else_=0)).label("today_regular"),
        func.sum(case((and_(CCTVSnapshot.created_at >= today_start, CCTVSnapshot.alert_sent == True), 1), else_=0)).label("today_alerts")
    ).first()

    total = row.total or 0
    ev_total = row.ev_total or 0
    regular_total = row.regular_total or 0
    today_total = row.today_total or 0
    today_ev = row.today_ev or 0
    today_regular = row.today_regular or 0
    today_alerts = row.today_alerts or 0

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
