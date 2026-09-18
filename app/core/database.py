import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

logger = logging.getLogger(__name__)

connect_args = {}
if "mysql" in settings.database_url:
    connect_args = {"connect_timeout": 3, "read_timeout": 10, "write_timeout": 10}

engine = create_engine(
    settings.database_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=15,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args=connect_args,
    echo=settings.DB_ECHO,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# CSMS Production Cloud Engine (Read-Only)
try:
    csms_connect_args = {}
    if "mysql" in settings.csms_database_url:
        csms_connect_args = {"connect_timeout": 3, "read_timeout": 10, "write_timeout": 10}
    csms_engine = create_engine(
        settings.csms_database_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=15,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=csms_connect_args,
        echo=False,
    )
    CSMSSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=csms_engine)
except Exception as e:
    logger.warning(f"CSMS Cloud DB engine init error: {e}")
    CSMSSessionLocal = SessionLocal


def get_db() -> Generator:
    """FastAPI dependency — CCTV local database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_csms_db() -> Generator:
    """FastAPI dependency — CSMS Cloud read-only database session."""
    db = CSMSSessionLocal()
    try:
        yield db
    finally:
        db.close()

