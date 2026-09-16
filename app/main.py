import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings, BASE_DIR
from app.core.database import engine, Base, get_db
from app.api.v1 import api_v1_router
from app.services.monitor_service import auto_monitor_service

logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cctv_app")

# Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME}...")

    # Ensure directories exist
    Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)

    # Create/update DB tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified.")
    except Exception as e:
        logger.warning(f"Database connection issue: {e}")

    # Start background monitor
    await auto_monitor_service.start()

    yield

    await auto_monitor_service.stop()
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    description="CCTV EV Detector — Korean EV License Plate Recognition & Classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# API routes
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


# Add Cache-Control Middleware to prevent browser caching stale dashboard and API data
@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "monitor": "running" if auto_monitor_service.is_running else "stopped",
    }


@app.get("/health/db", tags=["System"])
def db_health_check(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).scalar()
        return {"status": "connected", "database": settings.DB_NAME}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def serve_dashboard(request: Request):
    """Single-page admin dashboard."""
    dashboard_path = BASE_DIR / "app" / "templates" / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
            )
    return HTMLResponse(content="<h1>CCTV EV Detector Running. <a href='/docs'>API Docs</a></h1>")
