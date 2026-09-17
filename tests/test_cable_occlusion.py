import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.detector.plate_validator import KoreanPlateValidator



class TestCableOcclusion(unittest.TestCase):
    def test_plate_components_decomposition(self):
        p1 = KoreanPlateValidator.parse_plate_components("47호6633")
        self.assertEqual(p1, ("47", "호", "6633"))

        p2 = KoreanPlateValidator.parse_plate_components("123가4567")
        self.assertEqual(p2, ("123", "가", "4567"))

        p3 = KoreanPlateValidator.parse_plate_components("서울12가3456")
        self.assertEqual(p3, ("서울12", "가", "3456"))

    def test_cable_occlusion_exact_match(self):
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match("47호6633", "47호6633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match("47호 6633", "47호6633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match("123가4567", "123가 4567"))

    def test_cable_occlusion_digit_variations(self):
        anchor = "47호6633"

        # Variations caused by vertical cable and shadow cutting digits
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47오3633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47오8633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47오9633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47호3633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47호8633"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor, "47호0633"))

    def test_cable_occlusion_other_ev_plates(self):
        # 312버6132 variations
        anchor2 = "312버6132"
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor2, "312버8132"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor2, "312버0132"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor2, "312보6132"))

        # 52어0586 variations
        anchor3 = "52어0586"
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor3, "52어8586"))
        self.assertTrue(KoreanPlateValidator.is_cable_occlusion_match(anchor3, "52머0586"))

    def test_different_plates_rejected(self):
        self.assertFalse(KoreanPlateValidator.is_cable_occlusion_match("47호6633", "81머2072"))
        self.assertFalse(KoreanPlateValidator.is_cable_occlusion_match("123가4567", "987나1234"))
        self.assertFalse(KoreanPlateValidator.is_cable_occlusion_match("52어0586", "312버6132"))
        self.assertFalse(KoreanPlateValidator.is_cable_occlusion_match(None, "47호6633"))
        self.assertFalse(KoreanPlateValidator.is_cable_occlusion_match("47호6633", None))


if __name__ == "__main__":
    unittest.main()
