# 五种Y通道攻击

import torch
from torch import nn
from torch.nn import functional as torch_functional


DEFAULT_ALPHA = 1 / 255
DEFAULT_ITERATIVE_STEPS = 20
DEFAULT_PGD_STEPS = 10


def rgb_to_yuv_tensor(rgb_tensor: torch.Tensor) -> torch.Tensor:  # 公式13
    red = rgb_tensor[:, 0:1]
    green = rgb_tensor[:, 1:2]
    blue = rgb_tensor[:, 2:3]

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = -0.1687 * red - 0.3313 * green + 0.5 * blue + 128 / 255
    red_difference = 0.5 * red - 0.4187 * green - 0.0813 * blue + 128 / 255
    return torch.cat((luminance, blue_difference, red_difference), dim=1)


def yuv_to_rgb_tensor(yuv_tensor: torch.Tensor) -> torch.Tensor:  # 公式14
    luminance = yuv_tensor[:, 0:1]
    blue_difference = yuv_tensor[:, 1:2] - 128 / 255
    red_difference = yuv_tensor[:, 2:3] - 128 / 255

    red = luminance + 1.402 * red_difference
    green = luminance - 0.34414 * blue_difference - 0.71414 * red_difference
    blue = luminance + 1.772 * blue_difference
    return torch.cat((red, green, blue), dim=1)


def cal_yuv_grad(  # 复现：计算完整YUV梯度
    model: nn.Module,
    yuv_tensor: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    gradient_yuv = yuv_tensor.detach().requires_grad_(True)
    rgb_tensor = yuv_to_rgb_tensor(gradient_yuv)
    loss = torch_functional.cross_entropy(model(rgb_tensor), labels)
    return torch.autograd.grad(loss, gradient_yuv)[0].detach()


def clip_rgb_update(  # 复现：裁剪RGB扰动
    original_rgb: torch.Tensor,
    candidate_rgb: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    rgb_delta = (candidate_rgb - original_rgb).clamp(-epsilon, epsilon)
    return (original_rgb + rgb_delta).clamp(0, 1).detach()


def build_attack_result(  # 复现：返回结果
    original_rgb: torch.Tensor,
    adversarial_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_yuv = rgb_to_yuv_tensor(original_rgb)
    adversarial_yuv = rgb_to_yuv_tensor(adversarial_rgb)
    y_perturbation = adversarial_yuv[:, 0:1] - original_yuv[:, 0:1]
    return adversarial_rgb.detach(), y_perturbation.detach()


def yfgsm_attack(  # 复现：生成YFGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    yuv_images = rgb_to_yuv_tensor(images).clamp(0, 1)
    gradient = cal_yuv_grad(model, yuv_images, labels)
    candidate_yuv = yuv_images.clone()
    candidate_yuv[:, 0:1] += epsilon * gradient[:, 0:1].sign()
    candidate_rgb = yuv_to_rgb_tensor(candidate_yuv)
    adversarial_rgb = clip_rgb_update(images, candidate_rgb, epsilon)
    return build_attack_result(images, adversarial_rgb)


def yifgsm_attack(  # 复现：生成YI-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    yuv_images = rgb_to_yuv_tensor(images).clamp(0, 1)
    adversarial_rgb = images
    for _ in range(steps):
        gradient = cal_yuv_grad(model, yuv_images, labels)
        candidate_yuv = yuv_images.clone()
        candidate_yuv[:, 0:1] += alpha * gradient[:, 0:1].sign()
        candidate_rgb = yuv_to_rgb_tensor(candidate_yuv)
        adversarial_rgb = clip_rgb_update(images, candidate_rgb, epsilon)
        yuv_images = rgb_to_yuv_tensor(adversarial_rgb)
    return build_attack_result(images, adversarial_rgb)


def ypgd_attack(  # 复现：生成YPGD
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_PGD_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    yuv_images = rgb_to_yuv_tensor(images)
    yuv_images[:, 0:1] += torch.empty_like(yuv_images[:, 0:1]).uniform_(-epsilon, epsilon)
    yuv_images = yuv_images.clamp(0, 1)
    adversarial_rgb = images
    for _ in range(steps):
        gradient = cal_yuv_grad(model, yuv_images, labels)
        candidate_yuv = yuv_images.clone()
        candidate_yuv[:, 0:1] += alpha * gradient[:, 0:1].sign()
        candidate_rgb = yuv_to_rgb_tensor(candidate_yuv)
        adversarial_rgb = clip_rgb_update(images, candidate_rgb, epsilon)
        yuv_images = rgb_to_yuv_tensor(adversarial_rgb)
    return build_attack_result(images, adversarial_rgb)


def ymifgsm_attack(  # 复现：生成YMI-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    yuv_images = rgb_to_yuv_tensor(images).clamp(0, 1)
    momentum = torch.zeros_like(yuv_images[:, 0:1])
    adversarial_rgb = images
    for _ in range(steps):
        gradient = cal_yuv_grad(model, yuv_images, labels)[:, 0:1]
        gradient_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = momentum_decay * momentum + gradient / gradient_scale
        candidate_yuv = yuv_images.clone()
        candidate_yuv[:, 0:1] += alpha * momentum.sign()
        candidate_rgb = yuv_to_rgb_tensor(candidate_yuv)
        adversarial_rgb = clip_rgb_update(images, candidate_rgb, epsilon)
        yuv_images = rgb_to_yuv_tensor(adversarial_rgb)
    return build_attack_result(images, adversarial_rgb)


def ynifgsm_attack(  # 复现：生成YNI-FGSM
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    yuv_images = rgb_to_yuv_tensor(images).clamp(0, 1)
    momentum = torch.zeros_like(yuv_images[:, 0:1])
    adversarial_rgb = images
    for _ in range(steps):
        gradient_yuv = yuv_images.detach().requires_grad_(True)
        lookahead_yuv = gradient_yuv.clone()
        lookahead_yuv[:, 0:1] = (
            gradient_yuv[:, 0:1] + momentum_decay * alpha * momentum
        )
        lookahead_yuv = lookahead_yuv.clamp(0, 1)
        lookahead_rgb = yuv_to_rgb_tensor(lookahead_yuv)
        loss = torch_functional.cross_entropy(model(lookahead_rgb), labels)
        gradient = torch.autograd.grad(loss, gradient_yuv)[0][:, 0:1].detach()
        gradient_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = momentum_decay * momentum + gradient / gradient_scale
        candidate_yuv = yuv_images.clone()
        candidate_yuv[:, 0:1] += alpha * momentum.sign()
        candidate_rgb = yuv_to_rgb_tensor(candidate_yuv)
        adversarial_rgb = clip_rgb_update(images, candidate_rgb, epsilon)
        yuv_images = rgb_to_yuv_tensor(adversarial_rgb)
    return build_attack_result(images, adversarial_rgb)


def cal_y_grad_cmp(  # 对比：只计算Y通道梯度
    model: nn.Module,
    luminance: torch.Tensor,
    fixed_chrominance: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    gradient_luminance = luminance.detach().requires_grad_(True)
    yuv_tensor = torch.cat((gradient_luminance, fixed_chrominance), dim=1)
    rgb_tensor = yuv_to_rgb_tensor(yuv_tensor)
    loss = torch_functional.cross_entropy(model(rgb_tensor), labels)
    return torch.autograd.grad(loss, gradient_luminance)[0].detach()


def project_y_update_cmp(  # 对比：限制RGB扰动和像素范围
    original_rgb: torch.Tensor,
    original_yuv: torch.Tensor,
    candidate_luminance: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    reconstructed_rgb = yuv_to_rgb_tensor(original_yuv)
    lower_bound = torch.maximum(
        -reconstructed_rgb,
        original_rgb - epsilon - reconstructed_rgb,
    ).amax(dim=1, keepdim=True)
    upper_bound = torch.minimum(
        1 - reconstructed_rgb,
        original_rgb + epsilon - reconstructed_rgb,
    ).amin(dim=1, keepdim=True)
    candidate_delta = candidate_luminance - original_yuv[:, 0:1]
    projected_delta = torch.maximum(torch.minimum(candidate_delta, upper_bound), lower_bound)
    return original_yuv[:, 0:1] + projected_delta


def build_attack_result_cmp(  # 对比：生成RGB结果和Y扰动
    original_yuv: torch.Tensor,
    adversarial_luminance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adversarial_yuv = torch.cat((adversarial_luminance, original_yuv[:, 1:3]), dim=1)
    adversarial_rgb = yuv_to_rgb_tensor(adversarial_yuv).clamp(0, 1)
    y_perturbation = adversarial_luminance - original_yuv[:, 0:1]
    return adversarial_rgb.detach(), y_perturbation.detach()


def yfgsm_attack_cmp(  # 对比：严格固定U/V的单步攻击
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:  
    original_yuv = rgb_to_yuv_tensor(images)
    original_luminance = original_yuv[:, 0:1]
    gradient = cal_y_grad_cmp(
        model,
        original_luminance,
        original_yuv[:, 1:3],
        labels,
    )
    candidate_luminance = original_luminance + epsilon * gradient.sign()
    adversarial_luminance = project_y_update_cmp(
        images,
        original_yuv,
        candidate_luminance,
        epsilon,
    )
    return build_attack_result_cmp(original_yuv, adversarial_luminance)


def yifgsm_attack_cmp(  # 对比：严格固定U/V的迭代攻击
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:  
    original_yuv = rgb_to_yuv_tensor(images)
    fixed_chrominance = original_yuv[:, 1:3]
    adversarial_luminance = original_yuv[:, 0:1]

    for _ in range(steps):
        gradient = cal_y_grad_cmp(
            model,
            adversarial_luminance,
            fixed_chrominance,
            labels,
        )
        candidate_luminance = adversarial_luminance + alpha * gradient.sign()
        adversarial_luminance = project_y_update_cmp(
            images,
            original_yuv,
            candidate_luminance,
            epsilon,
        ).detach()

    return build_attack_result_cmp(original_yuv, adversarial_luminance)


def ypgd_attack_cmp(  # 对比：严格固定U/V的随机攻击
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_PGD_STEPS,
    random_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:  
    original_yuv = rgb_to_yuv_tensor(images)
    fixed_chrominance = original_yuv[:, 1:3]
    random_generator = torch.Generator(device=images.device).manual_seed(random_seed)
    random_delta = torch.rand(
        original_yuv[:, 0:1].shape,
        dtype=images.dtype,
        device=images.device,
        generator=random_generator,
    ) * (2 * epsilon) - epsilon
    adversarial_luminance = project_y_update_cmp(
        images,
        original_yuv,
        original_yuv[:, 0:1] + random_delta,
        epsilon,
    )

    for _ in range(steps):
        gradient = cal_y_grad_cmp(
            model,
            adversarial_luminance,
            fixed_chrominance,
            labels,
        )
        candidate_luminance = adversarial_luminance + alpha * gradient.sign()
        adversarial_luminance = project_y_update_cmp(
            images,
            original_yuv,
            candidate_luminance,
            epsilon,
        ).detach()

    return build_attack_result_cmp(original_yuv, adversarial_luminance)


def ymifgsm_attack_cmp(  # 对比：严格固定U/V的动量攻击
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:  
    original_yuv = rgb_to_yuv_tensor(images)
    fixed_chrominance = original_yuv[:, 1:3]
    adversarial_luminance = original_yuv[:, 0:1]
    momentum = torch.zeros_like(adversarial_luminance)

    for _ in range(steps):
        gradient = cal_y_grad_cmp(
            model,
            adversarial_luminance,
            fixed_chrominance,
            labels,
        )
        gradient_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        momentum = momentum_decay * momentum + gradient / gradient_scale
        candidate_luminance = adversarial_luminance + alpha * momentum.sign()
        adversarial_luminance = project_y_update_cmp(
            images,
            original_yuv,
            candidate_luminance,
            epsilon,
        ).detach()

    return build_attack_result_cmp(original_yuv, adversarial_luminance)

 
def ynifgsm_attack_cmp(  # 对比：严格固定U/V的前瞻攻击
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = DEFAULT_ALPHA,
    steps: int = DEFAULT_ITERATIVE_STEPS,
    momentum_decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:  
    original_yuv = rgb_to_yuv_tensor(images)
    fixed_chrominance = original_yuv[:, 1:3]
    adversarial_luminance = original_yuv[:, 0:1]
    momentum = torch.zeros_like(adversarial_luminance)

    for _ in range(steps):
        lookahead_luminance = project_y_update_cmp(
            images,
            original_yuv,
            adversarial_luminance + momentum_decay * alpha * momentum,
            epsilon,
        )
        gradient = cal_y_grad_cmp(
            model,
            lookahead_luminance,
            fixed_chrominance,
            labels,
        )
        gradient_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        momentum = momentum_decay * momentum + gradient / gradient_scale
        candidate_luminance = adversarial_luminance + alpha * momentum.sign()
        adversarial_luminance = project_y_update_cmp(
            images,
            original_yuv,
            candidate_luminance,
            epsilon,
        ).detach()

    return build_attack_result_cmp(original_yuv, adversarial_luminance)
