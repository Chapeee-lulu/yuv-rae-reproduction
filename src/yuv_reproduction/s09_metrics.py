# 图像质量、攻击和恢复指标

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity


def _val_rgb_pair(  # 检查同形状RGB图片
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first_rgb)
    second = np.asarray(second_rgb)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("两张RGB图像的HWC形状必须一致")
    return first, second


def cal_psnr(first_rgb: np.ndarray, second_rgb: np.ndarray) -> float:  # 计算PSNR
    first, second = _val_rgb_pair(first_rgb, second_rgb)
    difference = first.astype(np.float64) - second.astype(np.float64)
    mean_squared_error = float(np.mean(difference**2))
    if mean_squared_error == 0:
        return float("inf")
    return float(10 * np.log10((255**2) / mean_squared_error))


def cal_ssim(first_rgb: np.ndarray, second_rgb: np.ndarray) -> float:  # 计算SSIM
    first, second = _val_rgb_pair(first_rgb, second_rgb)
    return float(
        structural_similarity(
            first,
            second,
            channel_axis=2,
            data_range=255,
        )
    )


def cal_mean_ciede2000(  # 计算平均CIEDE2000
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
) -> float:
    first, second = _val_rgb_pair(first_rgb, second_rgb)
    first_lab = rgb2lab(first.astype(np.float64) / 255)
    second_lab = rgb2lab(second.astype(np.float64) / 255)
    return float(np.mean(deltaE_ciede2000(first_lab, second_lab)))


def cal_recovery_metrics(  # 统计逐像素恢复结果
    original_rgb: np.ndarray,
    recovered_rgb: np.ndarray,
) -> dict[str, float | int | bool]:
    original, recovered = _val_rgb_pair(original_rgb, recovered_rgb)
    absolute_error = np.abs(original.astype(np.int16) - recovered.astype(np.int16))
    equal_pixels = np.all(absolute_error == 0, axis=2)
    return {
        "exact_rgb_recovery": bool(np.array_equal(original, recovered)),
        "equal_pixel_ratio": float(equal_pixels.mean()),
        "maximum_channel_error": int(absolute_error.max()),
    }


def cal_attack_success_rate(  # 计算论文或严格ASR
    clean_predictions: list[int],
    adversarial_predictions: list[int],
    labels: list[int],
    clean_correct_only: bool = False,
) -> dict[str, float | int]:
    if not (
        len(clean_predictions) == len(adversarial_predictions) == len(labels)
    ):
        raise ValueError("预测结果与标签数量必须一致")
    valid_indices = [
        index
        for index, (clean_prediction, label) in enumerate(zip(clean_predictions, labels))
        if not clean_correct_only or clean_prediction == label
    ]
    success_count = sum(
        adversarial_predictions[index] != labels[index] for index in valid_indices
    )
    denominator = len(valid_indices)
    return {
        "success_count": int(success_count),
        "sample_count": denominator,
        "attack_success_rate": float(success_count / denominator) if denominator else 0.0,
    }


def cal_white_box_attack_success_rate(  # 计算白盒ASR
    clean_predictions: list[int],
    adversarial_predictions: list[int],
    labels: list[int],
    clean_correct_only: bool = False,
) -> dict[str, float | int]:
    return cal_attack_success_rate(
        clean_predictions,
        adversarial_predictions,
        labels,
        clean_correct_only,
    )


def cal_black_box_attack_success_rate(  # 计算黑盒ASR
    clean_predictions: list[int],
    adversarial_predictions: list[int],
    labels: list[int],
    clean_correct_only: bool = False,
) -> dict[str, float | int]:
    return cal_attack_success_rate(
        clean_predictions,
        adversarial_predictions,
        labels,
        clean_correct_only,
    )
