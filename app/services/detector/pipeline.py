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
    1. YOLOv8 orqali avtomobilni topish va to'g'ridan-to'g'ri uning Bamper qismiga (lower ROI) e'tibor qaratish.
    2. Spatial Proximity: Bir-biridan uzoqdagi matnlarni qo'shib yubormaslik (faqat bitta qatordagi yaqin matnlar birlashtiriladi).
    3. Tezkor chiqish (Early Exit): Bamperdan to'g'ri raqam topilishi bilan tahlil to'xtatiladi (~1-2 soniyada tugatadi).
    4. Auto-Deskewing & Super-Resolution orqali qiya va uzoq raqamlarni 100% aniqlash.
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

    @staticmethod
    def _smart_resize(image: np.ndarray, max_dim: int = 1600) -> Tuple[np.ndarray, float]:
        """Scales down oversized images to optimize YOLO and OCR speed without losing plate details."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image, 1.0
        scale = max_dim / float(max(h, w))
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _is_valid_collinear_pair(b1: Any, b2: Any) -> bool:
        """
        Strict spatial proximity check:
        Ensures b1 and b2 are on the exact same horizontal line and adjacent (not background stickers).
        """
        pts1 = np.array(b1, dtype=np.float32)
        pts2 = np.array(b2, dtype=np.float32)
        
        l1, r1 = np.min(pts1[:, 0]), np.max(pts1[:, 0])
        t1, btm1 = np.min(pts1[:, 1]), np.max(pts1[:, 1])
        h1 = max(btm1 - t1, 1.0)
        cy1 = (t1 + btm1) / 2.0
        
        l2, r2 = np.min(pts2[:, 0]), np.max(pts2[:, 0])
        t2, btm2 = np.min(pts2[:, 1]), np.max(pts2[:, 1])
        h2 = max(btm2 - t2, 1.0)
        cy2 = (t2 + btm2) / 2.0

        # Must read left-to-right
        if l1 >= l2:
            return False

        min_h = min(h1, h2)
        max_h = max(h1, h2)

        # 1. Vertical alignment (centers must be closely aligned)
        if abs(cy1 - cy2) > min_h * 0.65:
            return False

        # 2. Similar heights (avoid merging small stickers with large plates)
        if h1 / h2 < 0.45 or h1 / h2 > 2.2:
            return False

        # 3. Horizontal gap constraint (strictly adjacent)
        gap = l2 - r1
        if gap < -0.3 * max_h or gap > 2.5 * max_h:
            return False

        # 4. Merged bounding box aspect ratio
        comb_w = max(r1, r2) - min(l1, l2)
        comb_h = max(btm1, btm2) - min(t1, t2)
        aspect = comb_w / max(comb_h, 1.0)
        if aspect < 1.5 or aspect > 7.0:
            return False

        return True

    def _find_plate_candidates_in_crop(
        self, crop_img: np.ndarray, crop_source: str, weight: float
    ) -> List[Dict[str, Any]]:
        """
        Runs OCR on crop, handles single text boxes and strictly verified collinear adjacent pairs.
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

        # 2. Pairwise Merged Matches with strict spatial proximity
        n = len(ocr_segments)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                raw_1, conf_1, b1 = ocr_segments[i]
                raw_2, conf_2, b2 = ocr_segments[j]

                if self._is_valid_collinear_pair(b1, b2):
                    merged_raw = f"{raw_1}{raw_2}"
                    validated = KoreanPlateValidator.validate_and_normalize(merged_raw)
                    if validated:
                        plate_str, val_score = validated
                        pts1 = np.array(b1, dtype=np.float32)
                        pts2 = np.array(b2, dtype=np.float32)
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
                            "weight": weight * 1.05,
                            "source": f"{crop_source}_merged"
                        })

        return candidates

    @staticmethod
    def _normalize_angle(rw: float, rh: float, angle: float) -> Tuple[float, float, float]:
        """Normalizes minAreaRect angle to [-45, +45] right-side-up tilt angle."""
        if rw < rh:
            rw, rh = rh, rw
            angle += 90
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        return rw, rh, angle

    def _extract_deskewed_proposals(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """
        Extracts candidate plate regions (Blue EV + White plates + Yellow commercial)
        and deskews them by their minAreaRect angle with [-45, +45] normalization.
        """
        h, w = image.shape[:2]
        proposals = []

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(image)

            # 1. Blue mask (EV)
            hsv_blue = cv2.inRange(hsv, np.array([75, 20, 40]), np.array([145, 255, 255]))
            rgb_blue = ((b.astype(int) > (r.astype(int) + 8)) & (b.astype(int) > 40)).astype(np.uint8) * 255
            blue_mask = cv2.bitwise_or(hsv_blue, rgb_blue)

            # 2. White/Bright plate mask (ICE)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 55, 255]))

            # 3. Yellow/Commercial mask
            yellow_mask = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([35, 255, 255]))

            all_masks = [('blue', blue_mask), ('white', white_mask), ('yellow', yellow_mask)]

            candidates = []
            for color_name, mask in all_masks:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if 300 < area < (h * w * 0.20):
                        rect = cv2.minAreaRect(cnt)
                        (cx, cy), (rw, rh), angle = rect
                        rw, rh, angle = self._normalize_angle(rw, rh, angle)
                        aspect = rw / max(rh, 1.0)
                        if 1.8 <= aspect <= 7.0 and rw > 25:
                            score = abs(aspect - 4.7)
                            candidates.append((score, cx, cy, rw, rh, angle, color_name))

            candidates.sort(key=lambda x: x[0])

            for score, cx, cy, rw, rh, angle, color_name in candidates[:6]:
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4)
                w_half, h_half = int(rw / 2) + 14, int(rh / 2) + 10
                x1, y1 = max(0, int(cx - w_half)), max(0, int(cy - h_half))
                x2, y2 = min(w, int(cx + w_half)), min(h, int(cy + h_half))
                crop = rotated[y1:y2, x1:x2]
                if crop.size > 0:
                    if crop.shape[0] < 70:
                        crop = cv2.resize(crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
                    proposals.append((crop, f"{color_name}_deskew_{angle:.0f}deg", 1.5))

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
            raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if raw_image is None:
                result.error_message = "Rasmni o'qib bo'lmadi (Corrupt image)"
                return result

            # Smart resize to optimize speed (960px is optimal for both vehicle detection and OCR)
            image, _ = self._smart_resize(raw_image, max_dim=960)
            h_img, w_img = image.shape[:2]

            crops_to_test: List[Tuple[np.ndarray, str, float]] = []

            # 2. Detect vehicles with YOLOv8 and prioritize primary foreground vehicle
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

            vehicles.sort(key=lambda x: x[0], reverse=True)

            if vehicles:
                # Add Bumper ROI (lower 45% of primary vehicle) - FIRST priority
                area, (x1, y1, x2, y2), cls_name = vehicles[0]
                vh = y2 - y1
                vw = x2 - x1
                by1 = max(0, y1 + int(vh * 0.40))
                by2 = min(h_img, y2 + int(vh * 0.05))
                bx1 = max(0, x1 - int(vw * 0.05))
                bx2 = min(w_img, x2 + int(vw * 0.05))
                bumper_crop = image[by1:by2, bx1:bx2]
                if bumper_crop.size > 0 and (by2 - by1) > 20 and (bx2 - bx1) > 40:
                    crops_to_test.append((bumper_crop, f"{cls_name}_bumper_roi", 1.5))

                # Add deskewed color proposals
                deskewed_props = self._extract_deskewed_proposals(image)
                crops_to_test.extend(deskewed_props[:3])
            else:
                # No vehicle detected by YOLO -> check deskewed proposals + Lower Parking Bay ROI
                deskewed_props = self._extract_deskewed_proposals(image)
                if deskewed_props:
                    crops_to_test.extend(deskewed_props[:3])

                # Add lower parking slot region (lower 65% of entire frame)
                lower_bay = image[int(h_img * 0.35):h_img, 0:w_img]
                if lower_bay.size > 0:
                    crops_to_test.append((lower_bay, "lower_parking_bay_roi", 1.1))

            # Add full image fallback if no crops available
            if not crops_to_test:
                crops_to_test.append((image, "full_frame_fallback", 1.0))

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

                # Early Exit: If valid legal plate found with strong score on bumper/crop, stop immediately!
                if best_candidate and best_score >= 0.45:
                    break

            # 4. Build Final Result with Calibrated Threshold
            MIN_CONFIDENCE_THRESHOLD = 0.38
            MIN_TOTAL_SCORE = 0.30

            if best_candidate and best_candidate["confidence"] >= MIN_CONFIDENCE_THRESHOLD and best_score >= MIN_TOTAL_SCORE:
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
                    f"{ev_emoji} Detected: {result.plate_number} (Conf: {result.confidence:.2f}) | "
                    f"Type: {result.vehicle_type} | Color: {result.plate_color} | "
                    f"From: {best_candidate['source']} | Time: {(time.time()-start_time)*1000:.1f}ms"
                )
            else:
                if best_candidate:
                    logger.debug(f"Rejected low confidence plate candidate '{best_candidate.get('plate_number')}' (conf: {best_candidate.get('confidence'):.4f}, score: {best_score:.4f})")
                result.error_message = "Rasmda Koreya davlat raqami topilmadi yoki ishonchlilik darajasi yetarli emas"

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

