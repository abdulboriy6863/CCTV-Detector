from fastapi import APIRouter
from app.api.v1.detect import router as detect_router
from app.api.v1.vehicles import router as vehicles_router
from app.api.v1.cameras import router as cameras_router
from app.api.v1.stats import router as stats_router

api_v1_router = APIRouter()
api_v1_router.include_router(detect_router, prefix="/detect", tags=["Detection"])
api_v1_router.include_router(vehicles_router, prefix="/vehicles", tags=["Vehicles"])
api_v1_router.include_router(cameras_router, prefix="/cameras", tags=["Cameras"])
api_v1_router.include_router(stats_router, prefix="/stats", tags=["Statistics"])

__all__ = ["api_v1_router"]
