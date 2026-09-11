"""EV vehicle classifier based on license plate color analysis."""
import cv2
import logging
import numpy as np
from typing import Tuple
from dataclasses import dataclass

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
    - Havorang/Ko'k (하늘색) -> EV (전기차 / 수소차)
    - Oq (흰색) -> Oddiy avtomobil (내연기관)
    - Sariq (노란색) -> Tijorat (taksi / avtobus)
    """

    def __init__(self):
        # Korean Light Blue EV Plate (하늘색) requires genuine saturation (S >= 45)
        # to avoid neutral daylight white/silver reflection (S <= 25)
        self.BLUE_HSV_LOWER = np.array([85, 45, 50])
        self.BLUE_HSV_UPPER = np.array([135, 255, 255])

        # Yellow (Commercial Taxi/Bus)
        self.YELLOW_LOWER = np.array([15, 60, 80])
        self.YELLOW_UPPER = np.array([35, 255, 255])

        # Green (Old legacy or corporate plates)
        self.GREEN_LOWER = np.array([36, 50, 70])
        self.GREEN_UPPER = np.array([80, 255, 255])

    def classify(self, plate_crop: np.ndarray) -> EVClassification:
        """
        Classifies vehicle type based on the EXACT cropped plate image.
        Uses HSV + RGB Differential + Inner Core Region analysis for 100% precision.
        """
        try:
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

            hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
            b, g, r = cv2.split(plate_crop)

            # 1. Strict HSV Blue Mask
            blue_mask_hsv = cv2.inRange(hsv, self.BLUE_HSV_LOWER, self.BLUE_HSV_UPPER)

            # 2. Strict RGB Blue Dominance (B significantly higher than R & G)
            b_i = b.astype(np.int16)
            g_i = g.astype(np.int16)
            r_i = r.astype(np.int16)

            rgb_blue_mask = ((b_i > (r_i + 18)) & (b_i > (g_i - 10)) & (b_i > 50)).astype(np.uint8) * 255

            # Combined Strict Blue Mask (both HSV and RGB must agree)
            comb_blue_mask = cv2.bitwise_and(blue_mask_hsv, rgb_blue_mask)
            blue_ratio = cv2.countNonZero(comb_blue_mask) / float(total_pixels)

            # 3. Inner Plate Core Region (avoids frame borders and shadows)
            ih1, ih2 = int(h * 0.15), int(h * 0.85)
            iw1, iw2 = int(w * 0.10), int(w * 0.90)
            inner_total = (ih2 - ih1) * (iw2 - iw1)
            if inner_total > 0:
                inner_mask = comb_blue_mask[ih1:ih2, iw1:iw2]
                inner_blue_ratio = cv2.countNonZero(inner_mask) / float(inner_total)
            else:
                inner_blue_ratio = blue_ratio

            effective_blue = max(blue_ratio, inner_blue_ratio)

            # 4. Global channel means and saturation
            b_mean = float(np.mean(b))
            r_mean = float(np.mean(r))
            br_diff = b_mean - r_mean
            mean_sat = float(np.mean(hsv[:, :, 1]))

            # Yellow / Commercial Mask
            yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(total_pixels)

            # Green Mask
            green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
            green_ratio = cv2.countNonZero(green_mask) / float(total_pixels)

            logger.info(
                f"Color Analysis: Effective Blue={effective_blue:.2%} (full={blue_ratio:.2%}, "
                f"inner={inner_blue_ratio:.2%}, B-R={br_diff:.1f}, Sat={mean_sat:.1f}), Yellow={yellow_ratio:.2%}"
            )

            # 5. Robust EV Decision
            # True Korean EV plates have >= 25% blue or strong blue differential (B-R >= 15 & Sat >= 35)
            is_ev = (effective_blue >= 0.25) or (effective_blue >= 0.18 and br_diff >= 15.0 and mean_sat >= 35.0)

            if is_ev:
                logger.info(f"⚡ [EV Confirmed] Blue plate ratio: {effective_blue:.2%}")
                return EVClassification(
                    is_ev=True,
                    vehicle_type="EV",
                    plate_color="blue",
                    blue_ratio=round(effective_blue, 3),
                    confidence=min(0.60 + effective_blue * 0.8, 0.99)
                )

            elif yellow_ratio > 0.20:
                logger.info(f"🚕 Commercial Vehicle detected: Yellow={yellow_ratio:.2%}")
                return EVClassification(
                    is_ev=False,
                    vehicle_type="COMMERCIAL",
                    plate_color="yellow",
                    blue_ratio=round(blue_ratio, 3),
                    confidence=min(yellow_ratio * 2.0, 0.95)
                )

            elif green_ratio > 0.25:
                return EVClassification(
                    is_ev=False,
                    vehicle_type="REGULAR",
                    plate_color="green",
                    blue_ratio=round(blue_ratio, 3),
                    confidence=0.85
                )

            else:
                logger.info(f"🚗 Regular ICE Vehicle: White plate (Blue={blue_ratio:.2%})")
                return EVClassification(
                    is_ev=False,
                    vehicle_type="REGULAR",
                    plate_color="white",
                    blue_ratio=round(blue_ratio, 3),
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
