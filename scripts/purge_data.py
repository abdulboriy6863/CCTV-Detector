#!/usr/bin/env python3
"""
Production Data Purge Utility for CCTV-Detector.

Usage:
  python scripts/purge_data.py --all           # Purges snapshots, alerts, and cameras
  python scripts/purge_data.py --snapshots     # Purges only snapshots and alert history
  python scripts/purge_data.py --cameras       # Purges only camera configurations
  python scripts/purge_data.py --clean-images  # Deletes stored snapshot image files from disk
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal, engine
from app.models.snapshot import CCTVSnapshot, CCTVCamera
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PurgeUtility")


def purge_snapshots(db):
    """Deletes all vehicle snapshot records and alerts from DB."""
    count_snaps = db.query(CCTVSnapshot).delete()
    db.commit()
    logger.info(f"✅ O'chirildi: {count_snaps} ta CCTVSnapshot yozuvlari.")


def purge_cameras(db):
    """Deletes all registered CCTV cameras from DB."""
    count_cams = db.query(CCTVCamera).delete()
    db.commit()
    logger.info(f"✅ O'chirildi: {count_cams} ta CCTVCamera yozuvlari.")


def clean_disk_images():
    """Removes stored vehicle snapshot JPEG files from storage directory."""
    storage_dir = Path(settings.STORAGE_PATH)
    if not storage_dir.exists():
        logger.info(f"📁 Disk papkasi topilmadi: {storage_dir}")
        return

    deleted_count = 0
    for file in storage_dir.rglob("*.jpg"):
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Faylni o'chirishda xatolik {file}: {e}")

    logger.info(f"🗑️ Diskdan {deleted_count} ta rasm fayllari o'chirildi ({storage_dir}).")


def main():
    parser = argparse.ArgumentParser(description="Purge test data and prepare clean production environment.")
    parser.add_argument("--all", action="store_true", help="Purge all snapshots, alerts, cameras, and images")
    parser.add_argument("--snapshots", action="store_true", help="Purge only vehicle snapshot history and alerts")
    parser.add_argument("--cameras", action="store_true", help="Purge only camera configurations")
    parser.add_argument("--clean-images", action="store_true", help="Delete saved image files on disk")
    parser.add_argument("--yes", "-y", action="store_true", help="Confirm execution without interactive prompt")

    args = parser.parse_args()

    if not any([args.all, args.snapshots, args.cameras, args.clean_images]):
        parser.print_help()
        sys.exit(1)

    if not args.yes:
        confirm = input("⚠️ DIQQAT! Ushbu amal bazadagi ma'lumotlarni o'chiradi. Davom ettirasizmi? (y/N): ")
        if confirm.strip().lower() not in ["y", "yes"]:
            print("❌ Bekor qilindi.")
            sys.exit(0)

    db = SessionLocal()
    try:
        if args.all or args.snapshots:
            purge_snapshots(db)
        if args.all or args.cameras:
            purge_cameras(db)
        if args.all or args.clean_images:
            clean_disk_images()

        logger.info("🎉 Tozalash jarayoni muvaffaqiyatli yakunlandi!")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Xatolik yuz berdi: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
