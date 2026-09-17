"""File storage service for CCTV snapshots."""
import os
import uuid
import zipfile
import aiofiles
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from app.core.config import settings, get_kst_now

logger = logging.getLogger(__name__)


class StorageService:
    """Local file storage for CCTV snapshots — organized by date."""

    def __init__(self, base_storage_dir: Optional[str] = None):
        self.base_dir = Path(base_storage_dir or settings.STORAGE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_relative_path(
        self, cs_id: str, cp_id: str,
        plate_number: Optional[str] = None, extension: str = "jpg"
    ) -> str:
        now = get_kst_now()
        date_folder = now.strftime("%Y/%m/%d")
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        clean_plate = (plate_number or "UNKNOWN").replace(" ", "_").upper()
        clean_cs = cs_id.replace("/", "_").replace("\\", "_")
        clean_cp = cp_id.replace("/", "_").replace("\\", "_")
        short_id = uuid.uuid4().hex[:6]
        filename = f"{clean_cs}_{clean_cp}_{clean_plate}_{timestamp_str}_{short_id}.{extension}"
        return f"{date_folder}/{filename}"

    def generate_thumbnail_path(
        self, cs_id: str, cp_id: str,
        plate_number: Optional[str] = None, extension: str = "jpg"
    ) -> str:
        base_rel = self.generate_relative_path(cs_id, cp_id, plate_number, extension)
        return f"thumbs/{base_rel}"

    def get_absolute_path(self, relative_path: str) -> Path:
        clean_rel = os.path.normpath(str(relative_path).lstrip("/"))
        return (self.base_dir / clean_rel).resolve()

    async def save_image(self, image_bytes: bytes, relative_path: str) -> str:
        abs_path = self.get_absolute_path(relative_path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(abs_path, "wb") as f:
            await f.write(image_bytes)
        logger.info(f"Saved: {abs_path} ({len(image_bytes)} bytes)")
        return relative_path

    async def read_image(self, relative_path: str) -> Optional[bytes]:
        abs_path = self.get_absolute_path(relative_path)
        if not abs_path.exists():
            return None
        async with aiofiles.open(abs_path, "rb") as f:
            return await f.read()

    def create_zip_archive(self, file_records: List[Tuple[str, str]]) -> BytesIO:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel_path, zip_name in file_records:
                abs_path = self.get_absolute_path(rel_path)
                if abs_path.exists():
                    zf.write(abs_path, arcname=zip_name)
        zip_buffer.seek(0)
        return zip_buffer

    def cleanup_temp_frames(self, temp_dir: Optional[str] = None, max_age_hours: int = 24) -> int:
        """Removes temporary camera debug frames older than max_age_hours."""
        target_dir = Path(temp_dir) if temp_dir else (self.base_dir / "temp")
        if not target_dir.exists():
            return 0
        cleaned_count = 0
        now_ts = datetime.utcnow().timestamp()
        max_age_sec = max_age_hours * 3600
        try:
            for f in target_dir.glob("*.*"):
                if f.is_file() and (now_ts - f.stat().st_mtime) > max_age_sec:
                    f.unlink()
                    cleaned_count += 1
        except Exception as e:
            logger.warning(f"Error cleaning temp frames in {target_dir}: {e}")
        return cleaned_count

    def purge_all_snapshots(self) -> int:
        """Deletes all stored JPEG/PNG snapshot files from the base storage directory."""
        if not self.base_dir.exists():
            return 0
        deleted_count = 0
        for pattern in ["*.jpg", "*.jpeg", "*.png"]:
            for f in self.base_dir.rglob(pattern):
                try:
                    if f.is_file():
                        f.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")
        logger.info(f"Purged {deleted_count} snapshot images from storage ({self.base_dir})")
        return deleted_count


storage_service = StorageService()
