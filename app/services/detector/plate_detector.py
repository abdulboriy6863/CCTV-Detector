"""YOLOv8-based license plate detector."""
import logging
import time
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlateDetection:
    """A single detected license plate region."""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)


class PlateDetector:
    """
    YOLOv8 기반 번호판 검출기.
    Uses YOLOv8 to detect license plate regions in images.
    Falls back to full-image processing if no YOLO model is available.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.25):
        self.model_path = model_path
        self.confidence = confidence
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy-load YOLOv8 model."""
        if self._model_loaded:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._model_loaded = True
            logger.info(f"YOLOv8 model loaded from: {self.model_path}")
        except Exception as e:
            logger.warning(f"YOLOv8 model could not be loaded: {e}. Using fallback mode.")
            self._model = None
            self._model_loaded = True

    def detect(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Detect license plates in the image.
        Returns list of PlateDetection with cropped plate regions.
        """
        self._load_model()
        start = time.time()
        detections = []

        if self._model is not None:
            try:
                results = self._model(image, conf=self.confidence, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        # Crop the plate region
                        plate_crop = image[y1:y2, x1:x2].copy()
                        detections.append(PlateDetection(
                            bbox=[x1, y1, x2, y2],
                            confidence=conf,
                            plate_crop=plate_crop
                        ))
            except Exception as e:
                logger.error(f"YOLO detection error: {e}")

        # Fallback: if no YOLO model or no detections, use heuristic crops
        if not detections:
            detections = self._fallback_detect(image)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Plate detection: {len(detections)} plates found in {elapsed:.1f}ms")
        return detections

    def _fallback_detect(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Fallback detection using image region heuristics.
        Crops likely plate regions (center-bottom of image where plates typically are).
        """
        import cv2
        h, w = image.shape[:2]
        crops = []

        # Region 1: Center-bottom (most common plate location)
        y1, y2 = int(h * 0.55), int(h * 0.90)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        crop1 = image[y1:y2, x1:x2].copy()
        if crop1.shape[0] > 20 and crop1.shape[1] > 40:
            crops.append(PlateDetection(
                bbox=[x1, y1, x2, y2],
                confidence=0.5,
                plate_crop=crop1
            ))

        # Region 2: Full center area
        y1, y2 = int(h * 0.25), int(h * 0.80)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        crop2 = image[y1:y2, x1:x2].copy()
        if crop2.shape[0] > 20 and crop2.shape[1] > 40:
            crops.append(PlateDetection(
                bbox=[x1, y1, x2, y2],
                confidence=0.4,
                plate_crop=crop2
            ))

        # Region 3: Full image (last resort)
        crops.append(PlateDetection(
            bbox=[0, 0, w, h],
            confidence=0.3,
            plate_crop=image.copy()
        ))

        return crops


# Singleton
plate_detector = PlateDetector()
