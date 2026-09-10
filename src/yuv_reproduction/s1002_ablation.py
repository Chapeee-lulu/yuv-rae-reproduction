# RDH、YUV、YACK、ENS和OURS统一实验

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from .s1001_rgb_attack import (
    rgb_fgsm_attack,
    rgb_ifgsm_attack,
    rgb_mifgsm_attack,
    rgb_nifgsm_attack,
    rgb_pgd_attack,
)
from .s02_rgb_yuv import rgb_to_yuv
from .s04_y_attack import (
    yfgsm_attack,
    yifgsm_attack,
    ymifgsm_attack,
    ynifgsm_attack,
    ypgd_attack,
)


METHOD_NAMES = ("RDH", "YUV", "YACK", "ENS", "OURS")
ATTACK_NAMES = ("FGSM", "I-FGSM", "PGD", "MI-FGSM", "NI-FGSM")

Y_ATTACK_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "FGSM": yfgsm_attack,
    "I-FGSM": yifgsm_attack,
    "PGD": ypgd_attack,
    "MI-FGSM": ymifgsm_attack,
    "NI-FGSM": ynifgsm_attack,
}
RGB_ATTACK_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "FGSM": rgb_fgsm_attack,
    "I-FGSM": rgb_ifgsm_attack,
    "PGD": rgb_pgd_attack,
    "MI-FGSM": rgb_mifgsm_attack,
    "NI-FGSM": rgb_nifgsm_attack,
}


def generate_method_adversarial_rgb(  # 按消融方法生成攻击图
    method_name: str,
    attack_name: str,
    attack_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if method_name not in METHOD_NAMES:
        raise ValueError(f"未知方法：{method_name}")
    if attack_name not in ATTACK_NAMES:
        raise ValueError(f"未知攻击：{attack_name}")
    if method_name in ("RDH", "YUV", "ENS"):
        return RGB_ATTACK_FUNCTIONS[attack_name](
            attack_model,
            images,
            labels,
            epsilon,
        )
    adversarial_rgb, _ = Y_ATTACK_FUNCTIONS[attack_name](
        attack_model,
        images,
        labels,
        epsilon,
    )
    return adversarial_rgb


def cal_integer_y_perturbation(  # 计算取整后的Y通道差值
    original_rgb: np.ndarray,
    adversarial_rgb: torch.Tensor,
) -> np.ndarray:
    if adversarial_rgb.ndim != 4 or adversarial_rgb.shape[0] != 1:
        raise ValueError("对抗RGB必须只包含1张NCHW图像")
    adversarial_rgb_255 = (
        adversarial_rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255
    )
    original_yuv = rgb_to_yuv(np.asarray(original_rgb, dtype=np.uint8))
    adversarial_yuv = rgb_to_yuv(adversarial_rgb_255)
    return (
        adversarial_yuv[:, :, 0].astype(np.int32)
        - original_yuv[:, :, 0].astype(np.int32)
    )


def apply_binary_mask_to_integer_y_perturbation(  # 保留CAM区域内的整数扰动
    y_perturbation: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    perturbation = np.asarray(y_perturbation, dtype=np.int32)
    mask = np.asarray(binary_mask)
    if perturbation.shape != mask.shape:
        raise ValueError("Y扰动与CAM掩膜形状必须一致")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("CAM掩膜只能包含0和1")
    return perturbation * mask.astype(np.int32)
