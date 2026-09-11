import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "CCTV-EV-Detector"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # MySQL / MariaDB Database Settings
    DB_HOST: str = "211.253.38.156"
    DB_PORT: int = 3306
    DB_USER: str = "blue_networks"
    DB_PASSWORD: str = "blue_networks"
    DB_NAME: str = "blue_networks"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    # Storage Settings
    STORAGE_DIR: str = str(BASE_DIR / "storage" / "snapshots")
    MAX_FILE_SIZE_MB: int = 10

    # Security
    API_SECRET_KEY: Optional[str] = "cpos_cctv_secret_key_2026"

    # AI Model Settings
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolov8n.pt")
    YOLO_CONFIDENCE: float = 0.25
    OCR_LANGUAGE: str = "korean"
    EV_BLUE_THRESHOLD: float = 0.3

    # Monitor Settings
    MONITOR_INTERVAL_SECONDS: int = 10
    MONITOR_ENABLED: bool = True

    @property
    def database_url(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
