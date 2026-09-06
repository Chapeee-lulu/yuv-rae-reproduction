import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yuv_rae.pixel_comparison import compare_rgb_pixels


class PixelComparisonTests(unittest.TestCase):
    def test_channel_and_pixel_counts_have_different_meanings(self) -> None:
        reference_rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        candidate_rgb = np.array([[[1, 0, 0], [1, 1, 1]]], dtype=np.uint8)

        result = compare_rgb_pixels(reference_rgb, candidate_rgb)

        self.assertEqual(result.changed_rgb_channel_value_count, 4)
        self.assertEqual(result.changed_rgb_pixel_count, 2)
        self.assertEqual(result.total_rgb_pixel_count, 2)
        self.assertEqual(result.changed_rgb_pixel_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()

