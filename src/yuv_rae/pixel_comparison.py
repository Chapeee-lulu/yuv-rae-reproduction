"""RGB图像的逐像素一致性检查。

本模块只回答“两个RGB数组是否逐像素相同，以及误差如何分布”。
论文的PSNR、SSIM、CIEDE2000和攻击成功率将在后续评价模块实现。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RgbPixelComparisonResult:
    rgb_arrays_exactly_equal: bool
    max_rgb_channel_absolute_error: int
    mean_rgb_channel_absolute_error: float
    changed_rgb_channel_value_count: int
    changed_rgb_pixel_count: int
    total_rgb_pixel_count: int

    @property
    def changed_rgb_pixel_ratio(self) -> float:
        return self.changed_rgb_pixel_count / self.total_rgb_pixel_count


def compare_rgb_pixels(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
) -> RgbPixelComparisonResult:
    reference_array = np.asarray(reference_rgb)
    candidate_array = np.asarray(candidate_rgb)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"两张图像形状必须一致：{reference_array.shape} != {candidate_array.shape}"
        )
    if reference_array.ndim != 3 or reference_array.shape[-1] != 3:
        raise ValueError("图像必须是形状为(H, W, 3)的RGB数组")

    rgb_channel_absolute_errors = np.abs(
        reference_array.astype(np.int16) - candidate_array.astype(np.int16)
    )
    changed_rgb_pixel_mask = np.any(rgb_channel_absolute_errors != 0, axis=-1)
    return RgbPixelComparisonResult(
        rgb_arrays_exactly_equal=bool(
            np.array_equal(reference_array, candidate_array)
        ),
        max_rgb_channel_absolute_error=int(rgb_channel_absolute_errors.max()),
        mean_rgb_channel_absolute_error=float(rgb_channel_absolute_errors.mean()),
        changed_rgb_channel_value_count=int(
            np.count_nonzero(rgb_channel_absolute_errors)
        ),
        changed_rgb_pixel_count=int(np.count_nonzero(changed_rgb_pixel_mask)),
        total_rgb_pixel_count=int(changed_rgb_pixel_mask.size),
    )

