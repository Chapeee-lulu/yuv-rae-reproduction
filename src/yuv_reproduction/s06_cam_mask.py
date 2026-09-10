# CAM二值区域掩膜

import cv2
import torch
from torch import nn
from torch.nn import functional as torch_functional


def cal_gradcam_plus_plus_low_resolution_map(  # 计算低分辨率Grad-CAM++
    model: nn.Module,
    target_layer: nn.Module,
    input_batch: torch.Tensor,
    target_categories: torch.Tensor | None = None,
    gradient_method: str = "input",
) -> tuple[torch.Tensor, torch.Tensor]:
    saved_activation: dict[str, torch.Tensor] = {}
    saved_gradient: dict[str, torch.Tensor] = {}

    def save_activation(  # 保存目标层输出
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        saved_activation["value"] = output

    def save_input_gradient(  # 保存作者使用的输入侧梯度
        module: nn.Module,
        gradient_inputs: tuple[torch.Tensor | None, ...],
        gradient_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        if gradient_inputs[0] is not None:
            saved_gradient["value"] = gradient_inputs[0]

    forward_hook = target_layer.register_forward_hook(save_activation)
    backward_hook = None
    if gradient_method == "input":
        backward_hook = target_layer.register_backward_hook(save_input_gradient)
    elif gradient_method != "output":
        forward_hook.remove()
        raise ValueError(f"未知梯度方法：{gradient_method}")
    try:
        logits = model(input_batch)
        if target_categories is None:
            target_categories = logits.argmax(dim=1)
        target_scores = logits.gather(1, target_categories[:, None]).sum()
        activations = saved_activation["value"]
        if gradient_method == "input":
            model.zero_grad(set_to_none=True)
            target_scores.backward()
            gradients = saved_gradient["value"]
        else:
            gradients = torch.autograd.grad(target_scores, activations)[0]
    finally:
        forward_hook.remove()
        if backward_hook is not None:
            backward_hook.remove()

    gradient_squared = gradients.square()
    gradient_cubed = gradient_squared * gradients
    activation_sum = activations.sum(dim=(2, 3), keepdim=True)
    denominator = 2 * gradient_squared + activation_sum * gradient_cubed
    alpha = gradient_squared / (denominator + 1e-8)
    alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))
    weights = (gradients.clamp_min(0) * alpha).sum(dim=(2, 3), keepdim=True)
    cam_map = (weights * activations).sum(dim=1, keepdim=True).clamp_min(0)
    return cam_map.detach(), target_categories.detach()


def normalize_cam_map(cam_map: torch.Tensor) -> torch.Tensor:  # 归一化到0至1
    flat_map = cam_map.flatten(start_dim=1)
    minimum = flat_map.min(dim=1).values[:, None, None, None]
    maximum = flat_map.max(dim=1).values[:, None, None, None]
    return (cam_map - minimum) / (maximum - minimum).clamp_min(1e-8)


def resize_cam_map(  # 复现：使用OpenCV双线性插值
    cam_map: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    output_height, output_width = output_size
    resized_maps = []
    for sample_map in cam_map[:, 0]:
        resized_array = cv2.resize(
            sample_map.cpu().numpy(),
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_maps.append(torch.from_numpy(resized_array))
    resized_map = torch.stack(resized_maps, dim=0).unsqueeze(1)
    resized_map = resized_map.to(device=cam_map.device, dtype=cam_map.dtype)
    return normalize_cam_map(resized_map)


def generate_gradcam_plus_plus_map(  # 生成指定插值的热力图
    model: nn.Module,
    target_layer: nn.Module,
    input_batch: torch.Tensor,
    target_categories: torch.Tensor | None = None,
    interpolation_method: str = "opencv",
    gradient_method: str = "input",
) -> tuple[torch.Tensor, torch.Tensor]:
    low_resolution_map, target_categories = cal_gradcam_plus_plus_low_resolution_map(
        model,
        target_layer,
        input_batch,
        target_categories,
        gradient_method,
    )
    if interpolation_method == "opencv":
        cam_map = resize_cam_map(low_resolution_map, input_batch.shape[2:])
    elif interpolation_method == "pytorch":
        cam_map = resize_cam_map_cmp(low_resolution_map, input_batch.shape[2:])
    else:
        raise ValueError(f"未知插值方法：{interpolation_method}")
    return cam_map.detach(), target_categories


def cam_mask(  # 按阈值生成二值掩膜
    cam_map: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    return (cam_map > threshold).to(cam_map.dtype)


def apply_cam_mask_to_y_perturbation(  # 保留掩膜内的Y扰动
    y_perturbation: torch.Tensor,
    binary_mask: torch.Tensor,
) -> torch.Tensor:
    return y_perturbation * binary_mask


def resize_cam_map_cmp(  # 对比：使用PyTorch双线性插值
    cam_map: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    resized_map = torch_functional.interpolate(
        cam_map,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return normalize_cam_map(resized_map)
