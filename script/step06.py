# 第6步：验证CAM二值区域掩膜

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s06_cam_mask import (
    apply_cam_mask_to_y_perturbation,
    cam_mask,
    generate_gradcam_plus_plus_map,
)
from yuv_reproduction.s05_imgn_ensemble import (
    ENSEMBLE_MODEL_NAMES,
    EqualWeightLogitsEnsemble,
)
from yuv_reproduction.s03_imgn_inference import (
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
)
from yuv_reproduction.s04_y_attack import ymifgsm_attack


def save_map_image(values: torch.Tensor, path: Path, color_map: str) -> None:  # 保存二维图
    plt.imsave(path, values.squeeze().cpu().numpy(), cmap=color_map, vmin=0, vmax=1)


def save_perturbation_image(  # 保存Y扰动图
    perturbation: torch.Tensor,
    path: Path,
    epsilon: float,
) -> None:
    plt.imsave(
        path,
        perturbation.squeeze().cpu().numpy(),
        cmap="coolwarm",
        vmin=-epsilon,
        vmax=epsilon,
    )


def main() -> None:  # 用一张真实图片验证CAM掩膜
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 0.5
    epsilon = 2 / 255
    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    labels = load_generated_imgn_labels(PROJECT_ROOT / "results" / "step03_labels.csv")
    label = labels[image_path.name]
    with Image.open(image_path) as image:
        input_batch = add_batch_dimension(convert_image_to_unit_tensor(image)).to(device)
    labels = torch.tensor([label], device=device)

    white_box_models = [
        load_imgn_model(model_name, device)
        for model_name in ENSEMBLE_MODEL_NAMES
    ]
    ensemble_model = EqualWeightLogitsEnsemble(white_box_models).eval()
    _, y_perturbation = ymifgsm_attack(
        ensemble_model,
        input_batch,
        labels,
        epsilon,
    )
    del ensemble_model, white_box_models
    torch.cuda.empty_cache()

    cam_model = load_imgn_model("resnet50", device)
    opencv_cam_map, target_categories = generate_gradcam_plus_plus_map(
        cam_model,
        cam_model.layer4[-1],
        input_batch,
        interpolation_method="opencv",
    )
    pytorch_cam_map, _ = generate_gradcam_plus_plus_map(
        cam_model,
        cam_model.layer4[-1],
        input_batch,
        target_categories=target_categories,
        interpolation_method="pytorch",
    )
    output_gradient_cam_map, _ = generate_gradcam_plus_plus_map(
        cam_model,
        cam_model.layer4[-1],
        input_batch,
        target_categories=target_categories,
        interpolation_method="opencv",
        gradient_method="output",
    )
    opencv_binary_mask = cam_mask(opencv_cam_map, threshold)
    pytorch_binary_mask = cam_mask(pytorch_cam_map, threshold)
    output_gradient_binary_mask = cam_mask(
        output_gradient_cam_map,
        threshold,
    )
    opencv_masked_perturbation = apply_cam_mask_to_y_perturbation(
        y_perturbation,
        opencv_binary_mask,
    )
    pytorch_masked_perturbation = apply_cam_mask_to_y_perturbation(
        y_perturbation,
        pytorch_binary_mask,
    )
    outside_maximum = float(
        opencv_masked_perturbation[opencv_binary_mask == 0].abs().max().item()
    )
    opencv_mask_pixel_count = int(opencv_binary_mask.sum().item())
    pytorch_mask_pixel_count = int(pytorch_binary_mask.sum().item())
    total_pixel_count = int(opencv_binary_mask.numel())
    different_mask_pixel_count = int(
        (opencv_binary_mask != pytorch_binary_mask).sum().item()
    )
    output_gradient_different_mask_pixel_count = int(
        (opencv_binary_mask != output_gradient_binary_mask).sum().item()
    )

    reproduction_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step06"
    cmp_directory = PROJECT_ROOT / "outputs" / "cmp" / "step06"
    reproduction_directory.mkdir(parents=True, exist_ok=True)
    cmp_directory.mkdir(parents=True, exist_ok=True)
    save_map_image(opencv_cam_map, reproduction_directory / "cam_heatmap.png", "jet")
    save_map_image(pytorch_cam_map, cmp_directory / "cam_heatmap_pytorch.png", "jet")
    save_map_image(
        output_gradient_cam_map,
        cmp_directory / "cam_heatmap_output_gradient.png",
        "jet",
    )
    save_map_image(opencv_binary_mask, reproduction_directory / "cam_binary_mask.png", "gray")
    save_map_image(pytorch_binary_mask, cmp_directory / "cam_binary_mask_pytorch.png", "gray")
    save_map_image(
        output_gradient_binary_mask,
        cmp_directory / "cam_binary_mask_output_gradient.png",
        "gray",
    )
    save_perturbation_image(
        y_perturbation,
        reproduction_directory / "y_perturbation_before_mask.png",
        epsilon,
    )
    save_perturbation_image(
        opencv_masked_perturbation,
        reproduction_directory / "y_perturbation_after_mask.png",
        epsilon,
    )
    save_perturbation_image(
        pytorch_masked_perturbation,
        cmp_directory / "y_perturbation_after_mask_pytorch.png",
        epsilon,
    )

    result = {
        "step": "6_s06_cam_mask",
        "device": str(device),
        "image": image_path.name,
        "label": label,
        "cam_model": "resnet50",
        "target_category": int(target_categories.item()),
        "threshold": threshold,
        "primary_interpolation_method": "opencv",
        "primary_gradient_method": "input",
        "opencv_mask_pixel_count": opencv_mask_pixel_count,
        "pytorch_mask_pixel_count": pytorch_mask_pixel_count,
        "total_pixel_count": total_pixel_count,
        "opencv_mask_area_ratio": opencv_mask_pixel_count / total_pixel_count,
        "pytorch_mask_area_ratio": pytorch_mask_pixel_count / total_pixel_count,
        "cam_maximum_absolute_difference": float(
            (opencv_cam_map - pytorch_cam_map).abs().max().item()
        ),
        "different_mask_pixel_count": different_mask_pixel_count,
        "different_mask_pixel_ratio": different_mask_pixel_count / total_pixel_count,
        "output_gradient_cam_maximum_absolute_difference": float(
            (opencv_cam_map - output_gradient_cam_map).abs().max().item()
        ),
        "output_gradient_different_mask_pixel_count": (
            output_gradient_different_mask_pixel_count
        ),
        "output_gradient_different_mask_pixel_ratio": (
            output_gradient_different_mask_pixel_count / total_pixel_count
        ),
        "y_linf_before_mask": float(y_perturbation.abs().max().item()),
        "y_linf_after_opencv_mask": float(opencv_masked_perturbation.abs().max().item()),
        "y_linf_after_pytorch_mask": float(pytorch_masked_perturbation.abs().max().item()),
        "outside_mask_maximum": outside_maximum,
        "outside_mask_is_zero": outside_maximum == 0,
    }
    result_path = PROJECT_ROOT / "results" / "step06.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
