"""EasyOCR / PaddleOCR based Korean license plate text reader."""
import logging
import time
import cv2
import numpy as np
from typing import Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


class PlateReader:
    """
    Koreyscha davlat raqamlari matnini aniq o'qish (OCR Engine).
    Reads text and returns both text, confidence, and exact bounding box coordinates.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._reader = None
        self._reader_type = None

    def _init_reader(self):
        """Lazy loader for OCR engine — prefers EasyOCR / PaddleOCR."""
        if self._reader is not None:
            return

        # EasyOCR is highly accurate for Korean license plate numbers
        try:
            import torch
            torch.set_num_threads(1)
            import easyocr
            from app.core.config import settings, BASE_DIR
            local_model_dir = str(BASE_DIR / "models" / "easyocr")
            self._reader = easyocr.Reader(
                ['ko', 'en'],
                gpu=self.use_gpu,
                model_storage_directory=local_model_dir,
                user_network_directory=local_model_dir,
                download_enabled=False
            )
            self._reader_type = 'easyocr'
            logger.info("EasyOCR initialized successfully (Korean + English)")
            return
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

        # Fallback to PaddleOCR if available
        try:
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang='korean',
                use_gpu=self.use_gpu,
                show_log=False
            )
            self._reader_type = 'paddle'
            logger.info("PaddleOCR initialized (Korean mode)")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")

        logger.error("No OCR engine available!")

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocesses image for optimal OCR reading (contrast enhancement).
        """
        h, w = plate_img.shape[:2]
        if h < 50:
            scale = 100 / h
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def read_with_boxes(self, img: np.ndarray) -> List[Tuple[str, float, Any]]:
        """
        Reads text and returns (text, confidence, bbox).
        bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] in img coordinates.
        """
        self._init_reader()
        if self._reader is None:
            return []

        start = time.time()
        results = []

        try:
            import torch
            with torch.inference_mode():
                if self._reader_type == 'easyocr':
                    # Direct read on original color image to preserve coordinates
                    ocr_result = self._reader.readtext(img)
                    for bbox, text, conf in ocr_result:
                        results.append((text, float(conf), bbox))

                elif self._reader_type == 'paddle':
                    ocr_result = self._reader.ocr(img, cls=True)
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            bbox = line[0]
                            text = line[1][0]
                            conf = float(line[1][1])
                            results.append((text, conf, bbox))

        except Exception as e:
            logger.error(f"OCR read error: {e}")

        elapsed = (time.time() - start) * 1000
        logger.info(f"OCR detected {len(results)} text segments in {elapsed:.1f}ms")
        return results


# Singleton
plate_reader = PlateReader()
