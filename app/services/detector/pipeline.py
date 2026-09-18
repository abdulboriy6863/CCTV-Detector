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
        self.plate_crop_base64: Optional[str] = None
        self.processing_time_ms: float = 0.0
        self.error_message: Optional[str] = None
        self.vehicle_present: bool = False
        self.vehicle_box: Optional[List[int]] = None
        self.detected_vehicle_type: Optional[str] = None

    def get_raw_base64(self) -> Optional[str]:
        """Returns clean base64 string without data scheme header."""
        if self.plate_crop_base64:
            return self.plate_crop_base64
        if self.plate_crop_bytes:
            self.plate_crop_base64 = base64.b64encode(self.plate_crop_bytes).decode("utf-8")
            return self.plate_crop_base64
        return None

    def get_data_url(self) -> Optional[str]:
        """Returns standard data URL scheme for direct HTML/API rendering."""
        b64 = self.get_raw_base64()
        return f"data:image/jpeg;base64,{b64}" if b64 else None

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
            "vehicle_present": self.vehicle_present,
            "vehicle_box": self.vehicle_box,
            "detected_vehicle_type": self.detected_vehicle_type,
            "plate_region_base64": self.get_data_url(),
        }
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

        if n >= 3:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        boxes_sorted = sorted([(i, ocr_segments[i]), (j, ocr_segments[j]), (k, ocr_segments[k])], 
                                               key=lambda x: min(p[0] for p in x[1][2]))
                        
                        y_centers = [sum(p[1] for p in b[1][2]) / len(b[1][2]) for _, b in boxes_sorted]
                        if max(y_centers) - min(y_centers) > max(max(p[1] for p in b[1][2]) - min(p[1] for p in b[1][2]) for _, b in boxes_sorted) * 1.5:
                            continue  # Not collinear
                        
                        merged_raw = f"{boxes_sorted[0][1][0]}{boxes_sorted[1][1][0]}{boxes_sorted[2][1][0]}"
                        avg_conf = (boxes_sorted[0][1][1] + boxes_sorted[1][1][1] + boxes_sorted[2][1][1]) / 3.0
                        validated = KoreanPlateValidator.validate_and_normalize(merged_raw)
                        if validated:
                            plate_str, val_score = validated
                            pts_all = np.vstack((boxes_sorted[0][1][2], boxes_sorted[1][1][2], boxes_sorted[2][1][2]))
                            combined_bbox = [np.min(pts_all, axis=0).tolist(), np.max(pts_all, axis=0).tolist()]
                            candidates.append({
                                "plate_str": plate_str,
                                "raw_text": merged_raw,
                                "confidence": avg_conf,
                                "val_score": val_score,
                                "bbox": combined_bbox,
                                "weight": weight * 1.05,
                                "source": f"{crop_source}_merged_3box"
                            })

        # 3. Fallback: If no candidate found in raw crop, run on CLAHE contrast-enhanced crop
        if not candidates and crop_img.shape[0] >= 20 and crop_img.shape[1] >= 40:
            enhanced_img = plate_reader._preprocess_plate(crop_img)
            enh_segments = plate_reader.read_with_boxes(enhanced_img)
            
            for raw_text, ocr_conf, bbox in enh_segments:
                validated = KoreanPlateValidator.validate_and_normalize(raw_text)
                if validated:
                    plate_str, val_score = validated
                    candidates.append({
                        "plate_str": plate_str,
                        "raw_text": raw_text,
                        "confidence": ocr_conf,
                        "val_score": val_score,
                        "bbox": bbox,
                        "weight": weight * 1.05,
                        "source": f"{crop_source}_enhanced"
                    })

            n_enh = len(enh_segments)
            for i in range(n_enh):
                for j in range(n_enh):
                    if i == j:
                        continue
                    raw_1, conf_1, b1 = enh_segments[i]
                    raw_2, conf_2, b2 = enh_segments[j]

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
                                "weight": weight * 1.10,
                                "source": f"{crop_source}_enh_merged"
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

    def _extract_deskewed_proposals(
        self, image: np.ndarray, roi: Optional[Dict[str, float]] = None
    ) -> List[Tuple[np.ndarray, str, float]]:
        """
        Extracts candidate plate regions (Blue EV + White plates + Yellow commercial)
        inside Target Parking Bay and deskews them by their minAreaRect angle with [-45, +45] normalization.
        """
        h, w = image.shape[:2]
        proposals = []

        roi_cfg = roi if (isinstance(roi, dict) and roi) else {"x_min": 0.52, "y_min": 0.10, "x_max": 0.85, "y_max": 0.98}
        roi_xmin = float(roi_cfg.get("x_min", 0.52))
        roi_ymin = float(roi_cfg.get("y_min", 0.10))
        roi_xmax = float(roi_cfg.get("x_max", 0.85))
        roi_ymax = float(roi_cfg.get("y_max", 0.98))

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

                        # Spatial filter: Must lie within Target Bay ROI (exclude neighboring parking bays)
                        cx_norm = cx / float(w)
                        cy_norm = cy / float(h)
                        if cx_norm < roi_xmin or cx_norm > roi_xmax or cy_norm < roi_ymin or cy_norm > roi_ymax:
                            continue

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

    def _sync_detect(self, image_bytes: bytes, roi: Optional[Dict[str, float]] = None) -> DetectionResult:
        """
        Synchronous multi-scale detection pipeline with Target Parking Bay ROI filtering and Fast Early-Exit.
        
        Args:
            image_bytes: Raw captured camera image bytes
            roi: Optional dict specifying normalized Target Bay bounding box:
                 {"x_min": 0.20, "y_min": 0.10, "x_max": 0.80, "y_max": 0.98}
        """
        result = DetectionResult()
        start_time = time.time()

        roi_cfg = roi if (isinstance(roi, dict) and roi) else {"x_min": 0.52, "y_min": 0.10, "x_max": 0.85, "y_max": 0.98}
        roi_xmin = float(roi_cfg.get("x_min", 0.52))
        roi_ymin = float(roi_cfg.get("y_min", 0.10))
        roi_xmax = float(roi_cfg.get("x_max", 0.85))
        roi_ymax = float(roi_cfg.get("y_max", 0.98))
        bay_center_x = (roi_xmin + roi_xmax) / 2.0

        try:
            # 1. Decode image bytes
            np_arr = np.frombuffer(image_bytes, np.uint8)
            raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if raw_image is None:
                result.error_message = "Rasmni o'qib bo'lmadi (Corrupt image)"
                return result

            # Smart resize to optimize speed (960px is optimal for both vehicle detection and OCR)
            image, img_scale = self._smart_resize(raw_image, max_dim=960)
            h_img, w_img = image.shape[:2]

            crops_to_test: List[Tuple[np.ndarray, str, float]] = []

            # 2. Detect vehicles with YOLOv8 and prioritize primary vehicle in Target Bay ROI
            yolo = self._get_yolo()
            vehicles = []
            if yolo is not None:
                try:
                    yolo_res = yolo(image, conf=0.20, verbose=False)
                    for r in yolo_res:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = r.names.get(cls_id, "")
                            box_conf = float(box.conf[0])
                            if cls_name in ["car", "truck", "bus"]:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                area = (x2 - x1) * (y2 - y1)
                                cx_norm = ((x1 + x2) / 2.0) / float(w_img)
                                cy_bottom_norm = float(y2) / float(h_img)
                                
                                # Target Parking Bay ROI filtering:
                                # Vehicle must be horizontally centered in the target charging slot
                                if roi_xmin <= cx_norm <= roi_xmax and cy_bottom_norm >= (roi_ymin * 0.5):
                                    dist_to_center = abs(cx_norm - bay_center_x)
                                    prox_weight = max(0.2, 1.0 - 1.8 * dist_to_center)
                                    score = area * prox_weight * (box_conf + 0.5)
                                    vehicles.append((score, [x1, y1, x2, y2], cls_name, area, box_conf))
                                else:
                                    logger.debug(
                                        f"Bypassing neighboring vehicle at cx={cx_norm:.2f} (Target Bay: [{roi_xmin:.2f}, {roi_xmax:.2f}])"
                                    )
                except Exception as e:
                    logger.error(f"YOLO error: {e}")

            vehicles.sort(key=lambda x: x[0], reverse=True)

            if vehicles:
                result.vehicle_present = True
                result.vehicle_box = vehicles[0][1]
                result.detected_vehicle_type = vehicles[0][2]
                
                # Test bumper of the top target bay vehicles (primary + potential backup in slot)
                for v_score, (x1, y1, x2, y2), cls_name, _, _ in vehicles[:2]:
                    vh = y2 - y1
                    vw = x2 - x1
                    by1 = max(0, y1 + int(vh * 0.35))
                    by2 = min(h_img, y2 + int(vh * 0.10))
                    bx1 = max(0, x1 - int(vw * 0.05))
                    bx2 = min(w_img, x2 + int(vw * 0.05))
                    
                    inv_scale = 1.0 / img_scale
                    r_by1, r_by2 = int(by1 * inv_scale), int(by2 * inv_scale)
                    r_bx1, r_bx2 = int(bx1 * inv_scale), int(bx2 * inv_scale)
                    
                    bumper_crop = raw_image[r_by1:r_by2, r_bx1:r_bx2]
                    if bumper_crop.size > 0 and (r_by2 - r_by1) > 20 and (r_bx2 - r_bx1) > 40:
                        crops_to_test.append((bumper_crop, f"{cls_name}_bumper_roi", 1.5))

                # Add deskewed color proposals inside Target Bay
                deskewed_props = self._extract_deskewed_proposals(image, roi=roi_cfg)
                crops_to_test.extend(deskewed_props[:3])
            else:
                # No vehicle detected in target bay -> check deskewed proposals + Target Bay crop
                deskewed_props = self._extract_deskewed_proposals(image, roi=roi_cfg)
                if deskewed_props:
                    crops_to_test.extend(deskewed_props[:3])

                # Add Target Parking Slot region (strictly within horizontal and vertical bay bounds)
                by1_crop = max(0, int(h_img * roi_ymin))
                by2_crop = min(h_img, int(h_img * roi_ymax))
                bx1_crop = max(0, int(w_img * roi_xmin))
                bx2_crop = min(w_img, int(w_img * roi_xmax))
                target_bay_crop = image[by1_crop:by2_crop, bx1_crop:bx2_crop]
                if target_bay_crop.size > 0:
                    crops_to_test.append((target_bay_crop, "target_parking_bay_roi", 1.1))

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

                    # Color classification (EV vs Regular) with plate memory support
                    ev_result = ev_classifier.classify(tight_crop, plate_number=plate_str)

                    total_score = ocr_conf * val_score * cand["weight"]

                    if total_score > best_score:
                        best_score = total_score
                        best_candidate = {
                            "plate_number": plate_str,
                            "confidence": ocr_conf,
                            "val_score": val_score,
                            "raw_text": cand["raw_text"],
                            "ev_result": ev_result,
                            "plate_crop": tight_crop,
                            "source": cand["source"]
                        }

                # Early Exit: If valid legal plate found with strong score on bumper/crop, stop immediately!
                if best_candidate and (
                    (best_candidate["confidence"] >= 0.35 and best_score >= 0.40) or
                    (best_candidate.get("val_score", 0) >= 1.0 and best_candidate["confidence"] >= 0.16 and best_score >= 0.20)
                ):
                    break

            # 4. Build Final Result with Calibrated Threshold
            # Perfect standard plate regex format (val_score == 1.0) accepts calibrated low-res OCR confidences
            is_perfect_plate = best_candidate and best_candidate.get("val_score", 0) >= 1.0
            MIN_CONFIDENCE_THRESHOLD = 0.16 if is_perfect_plate else 0.38
            MIN_TOTAL_SCORE = 0.20 if is_perfect_plate else 0.30

            if best_candidate and best_candidate["confidence"] >= MIN_CONFIDENCE_THRESHOLD and best_score >= MIN_TOTAL_SCORE:
                result.success = True
                result.plate_number = best_candidate["plate_number"]
                result.confidence = best_candidate["confidence"]
                result.raw_ocr_text = best_candidate["raw_text"]
                result.vehicle_type = best_candidate["ev_result"].vehicle_type
                result.is_ev = best_candidate["ev_result"].is_ev
                result.plate_color = best_candidate["ev_result"].plate_color
                # Encode plate crop with optimal JPEG quality (85) and base64 representation
                crop_to_encode = best_candidate["plate_crop"]
                # Resize if excessively large to keep base64 payload under 25KB
                c_h, c_w = crop_to_encode.shape[:2]
                if c_w > 480:
                    scale = 480.0 / c_w
                    crop_to_encode = cv2.resize(crop_to_encode, (480, int(c_h * scale)), interpolation=cv2.INTER_AREA)

                jpeg_qual = getattr(settings, 'PLATE_JPEG_QUALITY', 85)
                _, buffer = cv2.imencode(
                    ".jpg",
                    crop_to_encode,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual]
                )
                result.plate_crop_bytes = buffer.tobytes()
                result.plate_crop_base64 = base64.b64encode(result.plate_crop_bytes).decode("utf-8")

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

    @staticmethod
    def crop_plate_base64(image: np.ndarray, bbox: List[int], quality: Optional[int] = None) -> Optional[str]:
        """Crops a bounding box with padding and encodes to base64 JPEG data URL."""
        if image is None or len(bbox) < 4:
            return None
        quality = quality or getattr(settings, 'PLATE_JPEG_QUALITY', 85)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        # Apply slight margin
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.15)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)
        
        crop = image[crop_y1:crop_y2, crop_x1:crop_y2]
        if crop.size == 0:
            return None
            
        c_h, c_w = crop.shape[:2]
        if c_w > 480:
            scale = 480.0 / c_w
            crop = cv2.resize(crop, (480, int(c_h * scale)), interpolation=cv2.INTER_AREA)

        ret, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        raw_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{raw_b64}"

    async def detect(self, image_bytes: bytes, roi: Optional[Dict[str, float]] = None) -> DetectionResult:
        """Async wrapper — runs detection in thread pool with Target Bay ROI support."""
        return await asyncio.to_thread(self._sync_detect, image_bytes, roi)


# Class Aliases & Singleton
PlateDetectionPipeline = DetectionPipeline
detection_pipeline = DetectionPipeline()

