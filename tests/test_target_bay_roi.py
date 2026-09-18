"""Tests for Target Parking Bay ROI filtering and multi-vehicle isolation."""
import unittest
import numpy as np
import cv2
import os

from app.services.detector.pipeline import DetectionPipeline, DetectionResult
from app.services.detector.plate_validator import KoreanPlateValidator


class TestTargetBayROI(unittest.TestCase):
    """Test suite for Target Parking Bay ROI filtering."""

    def setUp(self):
        self.pipeline = DetectionPipeline()
        self.sample_img_path = "/Users/abdulboriy/.gemini/antigravity/brain/3a590350-65ba-4e8f-808c-765a5528ff41/.user_uploaded/media_1789624122218.png"

    def test_multi_car_primary_bay_detection(self):
        """Verify that center EV is detected and adjacent cars are ignored in 3-car image."""
        if not os.path.exists(self.sample_img_path):
            self.skipTest("Sample multi-car image not found")

        with open(self.sample_img_path, "rb") as f:
            img_bytes = f.read()

        # 1. Detect with Center Target Bay ROI [0.25, 0.75]
        result = self.pipeline._sync_detect(
            img_bytes,
            roi={"x_min": 0.25, "y_min": 0.15, "x_max": 0.75, "y_max": 0.95}
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.plate_number)
        # Check that it detected the center car (6633)
        self.assertTrue(result.plate_number.endswith("6633"))
        self.assertTrue(result.is_ev)
        self.assertEqual(result.vehicle_type, "EV")
        self.assertEqual(result.plate_color, "blue")

    def test_adjacent_bay_isolation(self):
        """Verify that non-target bay ROI avoids selecting the center vehicle."""
        if not os.path.exists(self.sample_img_path):
            self.skipTest("Sample multi-car image not found")

        with open(self.sample_img_path, "rb") as f:
            img_bytes = f.read()

        # Target RIGHT bay only [0.65, 0.98]
        result = self.pipeline._sync_detect(
            img_bytes,
            roi={"x_min": 0.65, "y_min": 0.15, "x_max": 0.98, "y_max": 0.95}
        )

        # Center car 6633 should NOT be detected as the target plate
        if result.success and result.plate_number:
            self.assertFalse(result.plate_number.endswith("6633"))

    def test_default_roi_fallback(self):
        """Verify that passing None for ROI safely defaults to standard bay bounds."""
        if not os.path.exists(self.sample_img_path):
            self.skipTest("Sample multi-car image not found")

        with open(self.sample_img_path, "rb") as f:
            img_bytes = f.read()

        result = self.pipeline._sync_detect(img_bytes, roi=None)
        self.assertTrue(result.success)
        self.assertTrue(result.plate_number.endswith("6633"))


if __name__ == "__main__":
    unittest.main()
