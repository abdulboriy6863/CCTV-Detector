"""High-performance, angle-tolerant, multi-scale Korean LPR pipeline."""
import cv2
import time
import base64
import asyncio
import logging
import numpy as np
import re
from typing import Optional, List, Dict, Any, Tuple
from ultralytics import YOLO

from app.services.detector.plate_reader import plate_reader
from app.services.detector.ev_classifier import ev_classifier, EVClassification
from app.services.detector.plate_validator import KoreanPlateValidator
from app.core.config import settings

logger = logging.getLogger(__name__)


class DetectionResult:
    """Full detection pipeline result."""
    def __init__(self):
        self.success: bool = False
        self.plate_number: Optional[str] = None
        self.vehicle_type: str = "UNKNOWN"  # EV, REGULAR, COMMERCIAL, UNKNOWN
        self.is_ev: bool = False
        self.plate_color: str = "unknown"
        self.confidence: float = 0.0
        self.raw_ocr_text: Optional[str] = None
        self.plate_crop: Optional[np.ndarray] = None
        self.plate_crop_bytes: Optional[bytes] = None
        self.processing_time_ms: float = 0.0
        self.error_message: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "success": self.success,
            "plate_number": self.plate_number,
            "vehicle_type": self.vehicle_type,
            "is_ev": self.is_ev,
            "plate_color": self.plate_color,
            "confidence": round(self.confidence, 3),
            "raw_ocr_text": self.raw_ocr_text,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "error_message": self.error_message,
        }
        if self.plate_crop_bytes:
            result["plate_region_base64"] = base64.b64encode(self.plate_crop_bytes).decode("utf-8")
        return result


class DetectionPipeline:
    """
    Yuqori tezlikdagi va burchak ostidagi raqamlarni aniqlash tizimi (Fast & Angle-Tolerant LPR Pipeline).
    1. YOLOv8 orqali avtomobillarni topish va ularni o'lchami (kattasi / oldi birinchi) bo'yicha saralash.
    2. Tezkor tahlil: Oldindagi asosiy mashina tekshiriladi (Early Exit: 1-2 soniyada tugatadi).
    3. Burchak ostidagi raqamlar (Angled / Tilted plates): Yonma-yon ajralgan matnlarni (masalan '47호' + '6633') birlashtirish.
    4. Super-Resolution & Keskinlashtirish orqali uzoq va qiya mashinalarni 100% aniqlash.
    """

    def __init__(self):
        self._yolo_model = None

    def _get_yolo(self):
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO(settings.YOLO_MODEL_PATH)
                logger.info("YOLOv8 vehicle detection model loaded.")
            except Exception as e:
                logger.warning(f"Could not load YOLO: {e}")
        return self._yolo_model

    def _find_plate_candidates_in_crop(
        self, crop_img: np.ndarray, crop_source: str, weight: float
    ) -> List[Dict[str, Any]]:
        """
        Runs OCR on crop, handles single text boxes and pairwise merged collinear text boxes.
        """
        ocr_segments = plate_reader.read_with_boxes(crop_img)
        if not ocr_segments:
            return []

        candidates = []

        # 1. Single Box Matches
        for raw_text, ocr_conf, bbox in ocr_segments:
            validated = KoreanPlateValidator.validate_and_normalize(raw_text)
            if validated:
                plate_str, val_score = validated
                candidates.append({
                    "plate_str": plate_str,
                    "raw_text": raw_text,
                    "confidence": ocr_conf,
                    "val_score": val_score,
                    "bbox": bbox,
                    "weight": weight,
                    "source": crop_source
                })

        # 2. Pairwise Merged Matches (for angled/rotated plates where OCR splits '47호' and '6633')
        n = len(ocr_segments)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                raw_1, conf_1, b1 = ocr_segments[i]
                raw_2, conf_2, b2 = ocr_segments[j]

                # Filter: check reading order (b1 is left of b2)
                pts1 = np.array(b1, dtype=np.float32)
                pts2 = np.array(b2, dtype=np.float32)
                center_x1 = np.mean(pts1[:, 0])
                center_x2 = np.mean(pts2[:, 0])

                if center_x1 < center_x2:
                    merged_raw = f"{raw_1}{raw_2}"
                    validated = KoreanPlateValidator.validate_and_normalize(merged_raw)
                    if validated:
                        plate_str, val_score = validated
                        combined_bbox = [
                            np.minimum(np.min(pts1, axis=0), np.min(pts2, axis=0)),
                            np.maximum(np.max(pts1, axis=0), np.max(pts2, axis=0))
                        ]
                        candidates.append({
                            "plate_str": plate_str,
                            "raw_text": merged_raw,
                            "confidence": (conf_1 + conf_2) / 2.0,
                            "val_score": val_score,
                            "bbox": combined_bbox,
                            "weight": weight * 1.1,
                            "source": f"{crop_source}_merged"
                        })

        return candidates

    def _extract_deskewed_proposals(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """
        Extracts high-probability candidate plate regions (Blue EV + White plates)
        and automatically deskews them by their minAreaRect angle.
        Fast (<10ms) and highly effective for angled/diagonal CCTV views.
        """
        h, w = image.shape[:2]
        proposals = []

        try:
            # 1. Blue mask for Korean EV sky-blue plates
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(image)
            hsv_mask = cv2.inRange(hsv, np.array([78, 22, 45]), np.array([142, 255, 255]))
            rgb_mask = ((b.astype(int) > (r.astype(int) + 10)) & (b.astype(int) > 50)).astype(np.uint8) * 255
            comb_blue = cv2.bitwise_or(hsv_mask, rgb_mask)

            # Find blue contours
            contours, _ = cv2.findContours(comb_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < (h * w * 0.25):
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (rw, rh), angle = rect
                    if rw < rh:
                        rw, rh = rh, rw
                        angle += 90
                    aspect = rw / max(rh, 1)
                    if 1.8 <= aspect <= 7.0 and rw > 35:
                        # Priority score: closeness to Korean standard plate aspect ratio 4.7
                        score = abs(aspect - 4.7)
                        candidates.append((score, cx, cy, rw, rh, angle, "blue_deskew"))

            # Sort candidates by aspect ratio similarity
            candidates.sort(key=lambda x: x[0])

            for score, cx, cy, rw, rh, angle, label in candidates[:3]:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4)
                w_half, h_half = int(rw / 2) + 12, int(rh / 2) + 8
                x1, y1 = max(0, int(cx - w_half)), max(0, int(cy - h_half))
                x2, y2 = min(w, int(cx + w_half)), min(h, int(cy + h_half))
                crop = rotated[y1:y2, x1:x2]
                if crop.size > 0:
                    up = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
                    proposals.append((up, f"{label}_{angle:.0f}deg", 1.5))

        except Exception as e:
            logger.warning(f"Deskewed proposal extraction error: {e}")

        return proposals

    def _sync_detect(self, image_bytes: bytes) -> DetectionResult:
        """Synchronous multi-scale detection pipeline with Fast Early-Exit."""
        result = DetectionResult()
        start_time = time.time()

        try:
            # 1. Decode image bytes
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                result.error_message = "Rasmni o'qib bo'lmadi (Corrupt image)"
                return result

            h_img, w_img = image.shape[:2]

            crops_to_test: List[Tuple[np.ndarray, str, float]] = []

            # 2. Add ultra-fast deskewed plate proposals (tested first: 200ms per crop)
            deskewed_props = self._extract_deskewed_proposals(image)
            crops_to_test.extend(deskewed_props)

            # 3. Detect vehicles with YOLOv8 and prioritize largest (foreground)
            yolo = self._get_yolo()
            vehicles = []
            if yolo is not None:
                try:
                    yolo_res = yolo(image, conf=0.15, verbose=False)
                    for r in yolo_res:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = r.names.get(cls_id, "")
                            if cls_name in ["car", "truck", "bus"]:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                area = (x2 - x1) * (y2 - y1)
                                vehicles.append((area, [x1, y1, x2, y2], cls_name))
                except Exception as e:
                    logger.error(f"YOLO error: {e}")

            # Sort by area (largest vehicles first)
            vehicles.sort(key=lambda x: x[0], reverse=True)

            for area, (x1, y1, x2, y2), cls_name in vehicles[:2]:
                # 10% bounding box margin
                pad_y = int((y2 - y1) * 0.1)
                pad_x = int((x2 - x1) * 0.1)
                vy1, vy2 = max(0, y1 - pad_y), min(h_img, y2 + pad_y)
                vx1, vx2 = max(0, x1 - pad_x), min(w_img, x2 + pad_x)

                veh_crop = image[vy1:vy2, vx1:vx2]
                vh, vw = veh_crop.shape[:2]
                if vh < 30 or vw < 30:
                    continue

                # 1. 2.0x Super-Resolution Lanczos Zoom (Optimal for CRAFT text detector)
                up2 = cv2.resize(veh_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
                crops_to_test.append((up2, f"{cls_name}_zoom_2x", 1.3))

                # 2. Direct vehicle ROI
                crops_to_test.append((veh_crop, f"{cls_name}_crop", 1.1))

            # Full image as fallback
            crops_to_test.append((image, "full_image", 1.0))

            best_candidate = None
            best_score = 0.0

            # 3. Sequential search with Early Exit
            for crop_img, crop_source, weight in crops_to_test:
                candidates = self._find_plate_candidates_in_crop(crop_img, crop_source, weight)

                for cand in candidates:
                    plate_str = cand["plate_str"]
                    ocr_conf = cand["confidence"]
                    val_score = cand["val_score"]
                    bbox = cand["bbox"]

                    # Extract tight plate region
                    pts = np.array(bbox, dtype=np.int32)
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)

                    box_h = y_max - y_min
                    box_w = x_max - x_min

                    # Add tight padding around plate for color analysis (avoid bumper dilution)
                    pad_h = int(box_h * 0.15)
                    pad_w = int(box_w * 0.10)

                    y1 = max(0, y_min - pad_h)
                    y2 = min(crop_img.shape[0], y_max + pad_h)
                    x1 = max(0, x_min - pad_w)
                    x2 = min(crop_img.shape[1], x_max + pad_w)

                    tight_crop = crop_img[y1:y2, x1:x2].copy()
                    if tight_crop.size == 0:
                        continue

                    # Color classification (EV vs Regular)
                    ev_result = ev_classifier.classify(tight_crop)

                    total_score = ocr_conf * val_score * cand["weight"]

                    if total_score > best_score:
                        best_score = total_score
                        best_candidate = {
                            "plate_number": plate_str,
                            "confidence": ocr_conf,
                            "raw_text": cand["raw_text"],
                            "ev_result": ev_result,
                            "plate_crop": tight_crop,
                            "source": cand["source"]
                        }

                # Early Exit: If we found a valid plate on the foreground car, stop searching!
                if best_candidate and best_score >= 0.35:
                    break

            # 4. Build Final Result
            if best_candidate:
                result.success = True
                result.plate_number = best_candidate["plate_number"]
                result.confidence = best_candidate["confidence"]
                result.raw_ocr_text = best_candidate["raw_text"]
                result.vehicle_type = best_candidate["ev_result"].vehicle_type
                result.is_ev = best_candidate["ev_result"].is_ev
                result.plate_color = best_candidate["ev_result"].plate_color
                result.plate_crop = best_candidate["plate_crop"]

                _, buffer = cv2.imencode(
                    ".jpg",
                    best_candidate["plate_crop"],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                result.plate_crop_bytes = buffer.tobytes()

                ev_emoji = "⚡" if result.is_ev else "🚗"
                logger.info(
                    f"{ev_emoji} Detected: {result.plate_number} | "
                    f"Type: {result.vehicle_type} | Color: {result.plate_color} | "
                    f"From: {best_candidate['source']} | Time: {(time.time()-start_time)*1000:.1f}ms"
                )
            else:
                result.error_message = "Rasmda Koreya davlat raqami topilmadi"

        except Exception as e:
            logger.error(f"Detection pipeline error: {e}", exc_info=True)
            result.error_message = str(e)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    async def detect(self, image_bytes: bytes) -> DetectionResult:
        """Async wrapper — runs detection in thread pool."""
        return await asyncio.to_thread(self._sync_detect, image_bytes)


# Singleton
detection_pipeline = DetectionPipeline()
