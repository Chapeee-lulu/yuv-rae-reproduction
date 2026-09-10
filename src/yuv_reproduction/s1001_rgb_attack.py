# 五种RGB通道攻击

import torch
from torch import nn
from torch.nn import functional as torch_functional


DEFAULT_ALPHA = 1 / 255
DEFAULT_ITERATIVE_STEPS = 20
DEFAULT_PGD_STEPS = 10


def _cal_rgb_gradient(  # 计算RGB输入梯度
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    gradient_images = images.detach().requires_grad_(True)
    loss = torch_functional.cross_entropy(model(gradient_images), labels)
    return torch.autograd.grad(loss, gradient_images)[0].detach()


def _project_rgb_update(  # 裁剪RGB扰动
    original_img: torch.Tensor,
    candidate_images: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    delta = (candidate_images - original_img).clamp(-epsilon, epsilon)
    return (original_img + delta).clamp(0, 1).detach()


def rgb_fgsm_attack(  # 生成RGB FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    gradient = _cal_rgb_gradient(model, images, labels)
    return _project_rgb_update(images, images + epsilon * gradient.sign(), epsilon)


def rgb_ifgsm_attack(  # 生成RGB I-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
) -> torch.Tensor:
    adversarial_images = images
    for _ in range(steps):
        gradient = _cal_rgb_gradient(model, adversarial_images, labels)
        adversarial_images = _project_rgb_update(
            images,
            adversarial_images + alpha * gradient.sign(),
            epsilon,
        )
    return adversarial_images


def rgb_pgd_attack(  # 生成RGB PGD
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_PGD_STEPS,
) -> torch.Tensor:
    random_delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    adversarial_images = _project_rgb_update(images, images + random_delta, epsilon)
    for _ in range(steps):
        gradient = _cal_rgb_gradient(model, adversarial_images, labels)
        adversarial_images = _project_rgb_update(
            images,
            adversarial_images + alpha * gradient.sign(),
            epsilon,
        )
    return adversarial_images


def rgb_mifgsm_attack(  # 生成RGB MI-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> torch.Tensor:
    adversarial_images = images
    momentum = torch.zeros_like(images)
    for _ in range(steps):
        gradient = _cal_rgb_gradient(model, adversarial_images, labels)
        scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = momentum_decay * momentum + gradient / scale
        adversarial_images = _project_rgb_update(
            images,
            adversarial_images + alpha * momentum.sign(),
            epsilon,
        )
    return adversarial_images


def rgb_nifgsm_attack(  # 生成RGB NI-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> torch.Tensor:
    adversarial_images = images
    momentum = torch.zeros_like(images)
    for _ in range(steps):
        lookahead = _project_rgb_update(
            images,
            adversarial_images + momentum_decay * alpha * momentum,
            epsilon,
        )
        gradient = _cal_rgb_gradient(model, lookahead, labels)
        scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = momentum_decay * momentum + gradient / scale
        adversarial_images = _project_rgb_update(
            images,
            adversarial_images + alpha * momentum.sign(),
            epsilon,
        )
    return adversarial_images
