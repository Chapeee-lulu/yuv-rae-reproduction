"""图像逐像素检查量。

这些检查量用于审计“是否逐像素无损”，并不是论文组合提出的五项指标。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PixelErrorReport:
    exactly_equal: bool
    max_absolute_error: int
    mean_absolute_error: float
    changed_channel_values: int
    changed_pixels: int
    total_pixels: int

    @property
    def changed_pixel_ratio(self) -> float:
        return self.changed_pixels / self.total_pixels


def compare_uint8_images(reference: np.ndarray, candidate: np.ndarray) -> PixelErrorReport:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"两张图像形状必须一致：{reference_array.shape} != {candidate_array.shape}"
        )
    if reference_array.ndim != 3 or reference_array.shape[-1] != 3:
        raise ValueError("图像必须是形状为(H, W, 3)的RGB数组")

    difference = np.abs(
        reference_array.astype(np.int16) - candidate_array.astype(np.int16)
    )
    changed_pixel_mask = np.any(difference != 0, axis=-1)
    return PixelErrorReport(
        exactly_equal=bool(np.array_equal(reference_array, candidate_array)),
        max_absolute_error=int(difference.max()),
        mean_absolute_error=float(difference.mean()),
        changed_channel_values=int(np.count_nonzero(difference)),
        changed_pixels=int(np.count_nonzero(changed_pixel_mask)),
        total_pixels=int(changed_pixel_mask.size),
    )
