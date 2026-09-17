"""Unit tests for Base64 plate crop encoding and size threshold verification."""
import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.snapshot import DetectionResult, VehicleTypeEnum
from app.services.detector.pipeline import PlateDetectionPipeline, DetectionPipeline


class TestPlateCropBase64(unittest.TestCase):

    def setUp(self):
        self.pipeline = PlateDetectionPipeline()
        # Create synthetic test image (640x480)
        self.test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simulated blue license plate
        cv2.rectangle(self.test_img, (200, 300), (440, 380), (255, 120, 50), -1)
        cv2.putText(self.test_img, "81머2072", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def test_crop_plate_base64_generation(self):
        """Verify plate crop base64 encoding returns valid base64 data string under 30KB."""
        bbox = [200, 300, 440, 380] # [x1, y1, x2, y2]
        base64_str = self.pipeline.crop_plate_base64(self.test_img, bbox, quality=85)
        
        self.assertIsNotNone(base64_str)
        self.assertTrue(base64_str.startswith("data:image/jpeg;base64,"))
        
        # Verify base64 data size is lightweight (< 35,000 chars)
        self.assertLess(len(base64_str), 35000)

    def test_detection_result_data_url_helper(self):
        """Verify DetectionResult.get_data_url() handles both prefixed and raw base64."""
        det = DetectionResult(
            success=True,
            plate_number="81머2072",
            vehicle_type=VehicleTypeEnum.EV,
            is_ev=True,
            plate_region_base64="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
        )
        self.assertEqual(det.get_data_url(), "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...")

        # Test raw base64 without prefix
        det_raw = DetectionResult(
            success=True,
            plate_number="81머2072",
            plate_region_base64="RAWBASE64STRING12345"
        )
        self.assertEqual(det_raw.get_data_url(), "data:image/jpeg;base64,RAWBASE64STRING12345")


if __name__ == "__main__":
    unittest.main()
