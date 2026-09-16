"""EV vehicle classifier based on license plate color analysis and historical vehicle registry."""
import cv2
import logging
import numpy as np
from typing import Tuple, Optional, Set
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class EVClassification:
    """EV classification result."""
    is_ev: bool
    vehicle_type: str  # EV, REGULAR, COMMERCIAL, UNKNOWN
    plate_color: str   # blue, white, yellow, green, unknown
    blue_ratio: float  # Ratio of blue pixels (0.0 ~ 1.0)
    confidence: float  # Classification confidence


class EVClassifier:
    """
    Koreyscha davlat raqamlari foni rangini tahlil qilish (EV / Oddiy / Tijorat).
    1. Multi-Space Perceptual Color Analysis (CIE-LAB b* + Adaptive HSV + Normalized RGB)
    2. Text / Foreground Exclusion (Otsu binarization - faqat sof plita fonini tahlil qiladi)
    3. Historical EV Registry (Bir marta EV deb tasdiqlangan raqam doimiy EV sifatida saqlanadi)
    """

    def __init__(self):
        # Adaptive HSV ranges
        self.BLUE_HSV_LOWER = np.array([80, 20, 40])
        self.BLUE_HSV_UPPER = np.array([140, 255, 255])

        # Yellow (Commercial Taxi/Bus)
        self.YELLOW_LOWER = np.array([15, 60, 80])
        self.YELLOW_UPPER = np.array([35, 255, 255])

        # Green (Old legacy or corporate plates)
        self.GREEN_LOWER = np.array([36, 50, 70])
        self.GREEN_UPPER = np.array([80, 255, 255])

        # Global In-Memory EV Registry for confirmed EV vehicles
        self._confirmed_ev_plates: Set[str] = set()
        self._lock = threading.Lock()

    def register_ev_plate(self, plate_number: Optional[str]):
        """Registers a verified EV plate into memory to prevent glare-induced misclassifications."""
        if not plate_number:
            return
        clean = "".join(plate_number.split()).upper().replace("-", "")
        with self._lock:
            self._confirmed_ev_plates.add(clean)

    def is_known_ev(self, plate_number: Optional[str]) -> bool:
        """Checks if a plate has already been confirmed as an EV."""
        if not plate_number:
            return False
        clean = "".join(plate_number.split()).upper().replace("-", "")
        with self._lock:
            return clean in self._confirmed_ev_plates

    def classify(self, plate_crop: np.ndarray, plate_number: Optional[str] = None) -> EVClassification:
        """
        Classifies vehicle type based on cropped plate image and historical vehicle registry.
        """
        try:
            # 1. Check known historical EV registry first
            if plate_number and self.is_known_ev(plate_number):
                logger.info(f"⚡ [EV Confirmed via Historical Registry] {plate_number}")
                return EVClassification(
                    is_ev=True,
                    vehicle_type="EV",
                    plate_color="blue",
                    blue_ratio=0.85,
                    confidence=0.98
                )

            if plate_crop is None or plate_crop.size == 0:
                return EVClassification(
                    is_ev=False, vehicle_type="UNKNOWN",
                    plate_color="unknown", blue_ratio=0.0, confidence=0.0
                )

            h, w = plate_crop.shape[:2]
            total_pixels = h * w
            if total_pixels == 0:
                return EVClassification(
                    is_ev=False, vehicle_type="UNKNOWN",
                    plate_color="unknown", blue_ratio=0.0, confidence=0.0
                )

            # 2. Text Masking (Otsu threshold to exclude dark text pixels)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bg_pixels = max(1, cv2.countNonZero(text_mask))

            # 3. CIE-LAB Color Space Analysis (Perceptual Blue Chrominance b* channel)
            lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_lab = cv2.split(lab)
            # In OpenCV LAB, b channel represents Yellow-Blue (values < 124 represent negative b*, i.e. blue)
            lab_blue_mask = (b_lab < 122).astype(np.uint8) * 255
            lab_blue_bg = cv2.bitwise_and(lab_blue_mask, text_mask)
            lab_blue_ratio = cv2.countNonZero(lab_blue_bg) / float(bg_pixels)

            # 4. Adaptive HSV Mask
            hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
            blue_mask_hsv = cv2.inRange(hsv, self.BLUE_HSV_LOWER, self.BLUE_HSV_UPPER)
            hsv_blue_bg = cv2.bitwise_and(blue_mask_hsv, text_mask)
            hsv_blue_ratio = cv2.countNonZero(hsv_blue_bg) / float(bg_pixels)

            # 5. Normalized RGB Chromaticity (Blue Dominance)
            b, g, r = cv2.split(plate_crop)
            b_f = b.astype(np.float32)
            g_f = g.astype(np.float32)
            r_f = r.astype(np.float32)
            rgb_sum = b_f + g_f + r_f + 1e-5
            b_norm = b_f / rgb_sum
            br_diff = b_f - r_f

            rgb_blue_mask = ((b_norm > 0.355) & (br_diff > 8.0) & (b_f > 40)).astype(np.uint8) * 255
            rgb_blue_bg = cv2.bitwise_and(rgb_blue_mask, text_mask)
            rgb_blue_ratio = cv2.countNonZero(rgb_blue_bg) / float(bg_pixels)

            # Combined Effective Blue Score across representations
            effective_blue = max(lab_blue_ratio, hsv_blue_ratio, rgb_blue_ratio)
            mean_b_norm = float(np.mean(b_norm))
            mean_br_diff = float(np.mean(br_diff))
            mean_b_lab = float(np.mean(b_lab))

            # Yellow / Commercial Mask
            yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(total_pixels)

            # Green Mask
            green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
            green_ratio = cv2.countNonZero(green_mask) / float(total_pixels)

            logger.info(
                f"Color Analysis: Effective Blue={effective_blue:.2%} (LAB={lab_blue_ratio:.2%}, "
                f"HSV={hsv_blue_ratio:.2%}, RGB={rgb_blue_ratio:.2%}, mean_b*={mean_b_lab:.1f}, "
                f"B-R={mean_br_diff:.1f}), Yellow={yellow_ratio:.2%}"
            )

            # 6. Robust Decision Logic
            # Korean EV plate criteria:
            # - Effective Blue >= 15% OR
            # - LAB blue ratio >= 12% & B-R diff >= 8.0 OR
            # - Mean normalized blue >= 0.360 & B-R diff >= 10.0
            is_ev = (
                (effective_blue >= 0.15) or
                (lab_blue_ratio >= 0.12 and mean_br_diff >= 8.0) or
                (mean_b_norm >= 0.360 and mean_br_diff >= 10.0 and mean_b_lab < 125.0)
            )

            if is_ev:
                logger.info(f"⚡ [EV Confirmed] Blue plate ratio: {effective_blue:.2%}")
                if plate_number:
                    self.register_ev_plate(plate_number)
                return EVClassification(
                    is_ev=True,
                    vehicle_type="EV",
                    plate_color="blue",
                    blue_ratio=round(effective_blue, 3),
                    confidence=min(0.65 + effective_blue * 0.8, 0.99)
                )

            elif yellow_ratio > 0.20:
                logger.info(f"🚕 Commercial Vehicle detected: Yellow={yellow_ratio:.2%}")
                return EVClassification(
                    is_ev=False,
                    vehicle_type="COMMERCIAL",
                    plate_color="yellow",
                    blue_ratio=round(effective_blue, 3),
                    confidence=min(yellow_ratio * 2.0, 0.95)
                )

            elif green_ratio > 0.25:
                return EVClassification(
                    is_ev=False,
                    vehicle_type="REGULAR",
                    plate_color="green",
                    blue_ratio=round(effective_blue, 3),
                    confidence=0.85
                )

            else:
                logger.info(f"🚗 Regular ICE Vehicle: White plate (Blue={effective_blue:.2%})")
                return EVClassification(
                    is_ev=False,
                    vehicle_type="REGULAR",
                    plate_color="white",
                    blue_ratio=round(effective_blue, 3),
                    confidence=0.95
                )

        except Exception as e:
            logger.error(f"EV classification error: {e}")
            return EVClassification(
                is_ev=False,
                vehicle_type="UNKNOWN",
                plate_color="unknown",
                blue_ratio=0.0,
                confidence=0.0
            )


# Singleton
ev_classifier = EVClassifier()

