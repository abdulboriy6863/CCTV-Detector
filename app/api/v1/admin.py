"""
Admin Maintenance and Data Purge Endpoints.
Provides administrative actions for resetting in-memory session states,
purging test/historical vehicle snapshot data, and resetting camera configurations.
"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.models.snapshot import CCTVSnapshot
from app.models.camera import CCTVCamera
from app.models.alert import NonEVAlert
from app.services.monitor_service import auto_monitor_service
from app.services.storage_service import storage_service

logger = logging.getLogger("admin_api")

router = APIRouter()


class PurgeRequest(BaseModel):
    purge_snapshots: bool = Field(default=True, description="Purge vehicle snapshots and parking sessions")
    purge_alerts: bool = Field(default=True, description="Purge non-EV warning alerts")
    purge_cameras: bool = Field(default=False, description="Purge all registered CCTV cameras")
    delete_images: bool = Field(default=True, description="Delete physical JPEG snapshot files on disk")
    reset_active_sessions: bool = Field(default=True, description="Clear in-memory monitor sessions")


class PurgeResponse(BaseModel):
    success: bool
    message: str
    deleted_snapshots: int = 0
    deleted_alerts: int = 0
    deleted_cameras: int = 0
    deleted_images: int = 0
    sessions_cleared: int = 0


@router.post("/purge-data", response_model=PurgeResponse, summary="Purge test records and reset system")
async def purge_system_data(req: PurgeRequest, db: Session = Depends(get_db)):
    """
    Purges test data and historical snapshots to initialize a clean production environment.
    """
    deleted_snaps = 0
    deleted_alerts = 0
    deleted_cams = 0
    deleted_imgs = 0
    sessions_cleared = 0

    try:
        # 1. Reset in-memory tracking sessions
        if req.reset_active_sessions:
            sessions_cleared = len(auto_monitor_service.active_sessions)
            auto_monitor_service.active_sessions.clear()
            auto_monitor_service.camera_health.clear()
            logger.info(f"Admin reset: Cleared {sessions_cleared} active monitor sessions.")

        # 2. Purge Alerts
        if req.purge_alerts:
            deleted_alerts = db.query(NonEVAlert).delete()

        # 3. Purge Snapshots
        if req.purge_snapshots:
            deleted_snaps = db.query(CCTVSnapshot).delete()

        # 4. Purge Cameras if requested
        if req.purge_cameras:
            deleted_cams = db.query(CCTVCamera).delete()

        db.commit()

        # 5. Delete images from disk
        if req.delete_images:
            storage_dir = Path(settings.STORAGE_PATH)
            if storage_dir.exists():
                for f in storage_dir.rglob("*.jpg"):
                    try:
                        f.unlink()
                        deleted_imgs += 1
                    except Exception as e:
                        logger.warning(f"Error removing file {f}: {e}")

        msg = (
            f"Successfully purged {deleted_snaps} snapshots, {deleted_alerts} alerts, "
            f"{deleted_cams} cameras, {deleted_imgs} image files, and reset {sessions_cleared} active sessions."
        )
        logger.info(f"System purge completed: {msg}")
        return PurgeResponse(
            success=True,
            message=msg,
            deleted_snapshots=deleted_snaps,
            deleted_alerts=deleted_alerts,
            deleted_cameras=deleted_cams,
            deleted_images=deleted_imgs,
            sessions_cleared=sessions_cleared,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error during system purge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to purge data: {str(e)}"
        )


@router.post("/reset-sessions", summary="Reset in-memory active parking sessions")
async def reset_active_sessions():
    """
    Clears in-memory active parking sessions without deleting historical database records.
    """
    cleared_count = len(auto_monitor_service.active_sessions)
    auto_monitor_service.active_sessions.clear()
    auto_monitor_service.camera_health.clear()
    return {
        "success": True,
        "message": f"Successfully reset {cleared_count} active sessions.",
        "cleared_count": cleared_count
    }
