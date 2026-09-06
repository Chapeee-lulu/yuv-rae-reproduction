import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yuv_rae.metrics import compare_uint8_images
from yuv_rae.yuv_conversion import rgb_to_yuv, yuv_to_rgb


class YuvConversionTests(unittest.TestCase):
    def test_black_and_white_roundtrip_exactly(self) -> None:
        image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        recovered = yuv_to_rgb(rgb_to_yuv(image)).astype(np.uint8)
        self.assertTrue(np.array_equal(image, recovered))

    def test_integer_roundtrip_error_is_at_most_one(self) -> None:
        colors = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[12, 34, 56], [123, 45, 210], [250, 128, 3]],
            ],
            dtype=np.uint8,
        )
        recovered = yuv_to_rgb(rgb_to_yuv(colors)).astype(np.uint8)
        report = compare_uint8_images(colors, recovered)
        self.assertLessEqual(report.max_absolute_error, 1)

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rgb_to_yuv(np.zeros((8, 8), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

