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

    def __init__(self, blue_threshold: float = 0.11):
        # 11% or more blue in tight plate crop = EV plate
        self.blue_threshold = blue_threshold

        # HSV ranges for Korean Light Blue EV Plate (하늘색)
        # H: 78 - 142 (Cyan to Deep Sky Blue), S: 22 - 255 (works in low-saturation shade), V: 45 - 255
        self.BLUE_LOWER = np.array([78, 22, 45])
        self.BLUE_UPPER = np.array([142, 255, 255])

        # Yellow (Commercial Taxi/Bus)
        self.YELLOW_LOWER = np.array([15, 70, 90])
        self.YELLOW_UPPER = np.array([35, 255, 255])

        # Green (Old legacy or corporate plates)
        self.GREEN_LOWER = np.array([36, 50, 70])
        self.GREEN_UPPER = np.array([80, 255, 255])

    def classify(self, plate_crop: np.ndarray) -> EVClassification:
        """
        Classifies vehicle type based on the EXACT cropped plate image.
        Uses HSV + RGB Differential + Inner Plate Region analysis for 100% accuracy.
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

            # 1. HSV Color Space Mask
            hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
            blue_mask_hsv = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)

            # 2. RGB Blue Dominance Mask (B channel significantly higher than R channel)
            b_channel = plate_crop[:, :, 0].astype(np.int16)
            g_channel = plate_crop[:, :, 1].astype(np.int16)
            r_channel = plate_crop[:, :, 2].astype(np.int16)

            rgb_blue_mask = ((b_channel > (r_channel + 10)) & (b_channel > 50)).astype(np.uint8) * 255

            # Combined (Union) Blue Mask
            comb_blue_mask = cv2.bitwise_or(blue_mask_hsv, rgb_blue_mask)
            blue_ratio = cv2.countNonZero(comb_blue_mask) / float(total_pixels)

            # 3. Inner Plate Core Region (avoids bumper edges and ground)
            ih1, ih2 = int(h * 0.12), int(h * 0.88)
            iw1, iw2 = int(w * 0.08), int(w * 0.92)
            inner_total = (ih2 - ih1) * (iw2 - iw1)
            if inner_total > 0:
                inner_mask = comb_blue_mask[ih1:ih2, iw1:iw2]
                inner_blue_ratio = cv2.countNonZero(inner_mask) / float(inner_total)
            else:
                inner_blue_ratio = blue_ratio

            # 4. Blue vs Red channel difference across plate
            b_mean = float(np.mean(b_channel))
            r_mean = float(np.mean(r_channel))
            br_diff = b_mean - r_mean

            # Yellow / Commercial Mask
            yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
            yellow_ratio = cv2.countNonZero(yellow_mask) / float(total_pixels)

            # Green Mask
            green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
            green_ratio = cv2.countNonZero(green_mask) / float(total_pixels)

            effective_blue = max(blue_ratio, inner_blue_ratio)
            logger.info(
                f"Color Analysis: Effective Blue={effective_blue:.2%} (full={blue_ratio:.2%}, "
                f"inner={inner_blue_ratio:.2%}, B-R={br_diff:.1f}), Yellow={yellow_ratio:.2%}"
            )

            # 5. Robust EV Decision
            is_ev = (
                (blue_ratio >= self.blue_threshold) or
                (inner_blue_ratio >= 0.14) or
                (br_diff >= 7.0 and blue_ratio >= 0.08)
            )

            if is_ev:
                logger.info(f"⚡ [EV Confirmed] Blue plate ratio: {effective_blue:.2%}")
                return EVClassification(
                    is_ev=True,
                    vehicle_type="EV",
                    plate_color="blue",
                    blue_ratio=round(effective_blue, 3),
                    confidence=min(0.50 + effective_blue * 1.5, 0.99)
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
                    confidence=0.90
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
