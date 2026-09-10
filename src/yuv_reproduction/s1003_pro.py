# Step 10 Pro的CPU并行任务

import numpy as np

from .s02_rgb_yuv import rgb_to_yuv
from .s08_rae import (
    cal_rgb_residual_sup,
    generate_rae,
    recover_rgb_exact_sup,
    recover_yuv_payload_sup,
    yuv_to_rgb_uint8,
)
from .s09_metrics import (
    cal_mean_ciede2000,
    cal_psnr,
    cal_recovery_metrics,
    cal_ssim,
)
from .s1002_ablation import apply_binary_mask_to_integer_y_perturbation


def prepare_rae_candidate_pro(  # 并行生成单张RAE候选
    original_rgb: np.ndarray,
    adversarial_rgb_255: np.ndarray,
    binary_mask: np.ndarray,
) -> dict[str, object]:
    original = np.asarray(original_rgb, dtype=np.uint8)
    original_yuv = rgb_to_yuv(original)
    adversarial_yuv = rgb_to_yuv(
        np.asarray(adversarial_rgb_255, dtype=np.float32)
    )
    y_perturbation = (
        adversarial_yuv[:, :, 0].astype(np.int32)
        - original_yuv[:, :, 0].astype(np.int32)
    )
    masked_perturbation = apply_binary_mask_to_integer_y_perturbation(
        y_perturbation,
        binary_mask,
    )
    try:
        reversible_yuv, embedding = generate_rae(
            original_yuv,
            masked_perturbation,
        )
    except ValueError as error:
        return {"succeeded": False, "error": str(error)}

    attacked_yuv = original_yuv.copy()
    attacked_yuv[:, :, 0] = (
        original_yuv[:, :, 0].astype(np.int32) + masked_perturbation
    )
    return {
        "succeeded": True,
        "rae_rgb": yuv_to_rgb_uint8(reversible_yuv),
        "attacked_rgb": yuv_to_rgb_uint8(attacked_yuv),
        "reversible_yuv": reversible_yuv,
        "embedding": embedding,
        "masked_perturbation": masked_perturbation,
    }


def finalize_rae_candidate_pro(  # 并行计算恢复和图像指标
    original_rgb: np.ndarray,
    candidate: dict[str, object],
    label: int,
    rae_prediction: int,
    outer_iteration: int,
) -> dict[str, object]:
    if not candidate["succeeded"]:
        return candidate

    original = np.asarray(original_rgb, dtype=np.uint8)
    rae_rgb = np.asarray(candidate["rae_rgb"], dtype=np.uint8)
    reversible_yuv = np.asarray(candidate["reversible_yuv"])
    masked_perturbation = np.asarray(
        candidate["masked_perturbation"],
        dtype=np.int32,
    )
    embedding = dict(candidate["embedding"])
    recovered_yuv, recovered_perturbation = recover_yuv_payload_sup(
        reversible_yuv,
        int(embedding["threshold"]),
        embedding["arithmetic_metadata"],
    )
    original_yuv = rgb_to_yuv(original)
    residual = cal_rgb_residual_sup(original, original_yuv)
    recovered_rgb = recover_rgb_exact_sup(recovered_yuv, residual)
    recovery = cal_recovery_metrics(original, recovered_rgb)

    return {
        "succeeded": True,
        "rae_rgb": rae_rgb,
        "rae_prediction_inception": int(rae_prediction),
        "inception_attack_success": int(rae_prediction) != int(label),
        "outer_iterations": int(outer_iteration),
        "embedding_threshold": int(embedding["threshold"]),
        "payload_length": int(embedding["payload_length"]),
        "masked_nonzero_count": int(np.count_nonzero(masked_perturbation)),
        "perturbation_recovered": bool(
            np.array_equal(recovered_perturbation, masked_perturbation)
        ),
        "exact_recovery": recovery,
        "paper_exact_recovery": None,
        "psnr": cal_psnr(original, rae_rgb),
        "ssim": cal_ssim(original, rae_rgb),
        "ciede2000": cal_mean_ciede2000(original, rae_rgb),
        "_reversible_yuv": reversible_yuv,
        "_rgb_conversion_residual": residual,
        "_masked_perturbation": masked_perturbation,
        "_arithmetic_metadata": embedding["arithmetic_metadata"],
    }
