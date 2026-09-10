# 第4步：验证五种Y通道攻击

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as transform_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s03_imgn_inference import (
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
    predict_class_indices,
)
from yuv_reproduction.s04_y_attack import (
    rgb_to_yuv_tensor,
    yfgsm_attack,
    yifgsm_attack,
    ymifgsm_attack,
    ynifgsm_attack,
    ypgd_attack,
)


def main() -> None:  # 用一张真实图片验证五种攻击
    torch.manual_seed(2022)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    labels = load_generated_imgn_labels(PROJECT_ROOT / "results" / "step03_labels.csv")
    label = labels[image_path.name]
    with Image.open(image_path) as image:
        image_tensor = convert_image_to_unit_tensor(image)
    images = add_batch_dimension(image_tensor).to(device)
    labels = torch.tensor([label], device=device)
    model = load_imgn_model("inception_v3", device)
    clean_prediction = int(predict_class_indices(model, images).item())

    attacks = {
        "YFGSM": yfgsm_attack,
        "YI-FGSM": yifgsm_attack,
        "YPGD": ypgd_attack,
        "YMI-FGSM": ymifgsm_attack,
        "YNI-FGSM": ynifgsm_attack,
    }
    output_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step04"
    output_directory.mkdir(parents=True, exist_ok=True)
    original_yuv = rgb_to_yuv_tensor(images)
    reports: list[dict[str, object]] = []

    for epsilon in (2 / 255, 4 / 255):
        for attack_name, attack in attacks.items():
            start_time = time.perf_counter()
            adversarial_rgb, y_perturbation = attack(model, images, labels, epsilon)
            adversarial_prediction = int(predict_class_indices(model, adversarial_rgb).item())
            adversarial_yuv = rgb_to_yuv_tensor(adversarial_rgb)
            uv_max_absolute_change = float(
                (adversarial_yuv[:, 1:3] - original_yuv[:, 1:3]).abs().max().item()
            )
            rgb_linf = float((adversarial_rgb - images).abs().max().item())
            y_linf = float(y_perturbation.abs().max().item())
            epsilon_pixels = round(epsilon * 255)
            output_path = output_directory / f"{attack_name.lower()}_eps_{epsilon_pixels}.png"
            transform_function.to_pil_image(adversarial_rgb[0].cpu()).save(output_path)
            reports.append(
                {
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "alpha": 1 / 255,
                    "iterations": 1 if attack_name == "YFGSM" else 10 if attack_name == "YPGD" else 20,
                    "clean_prediction": clean_prediction,
                    "adversarial_prediction": adversarial_prediction,
                    "attack_success": adversarial_prediction != label,
                    "y_linf": y_linf,
                    "rgb_linf": rgb_linf,
                    "uv_max_absolute_change": uv_max_absolute_change,
                    "rgb_constraint_passed": rgb_linf <= epsilon + 1e-6,
                    "uv_is_unchanged": uv_max_absolute_change == 0,
                    "runtime_seconds": time.perf_counter() - start_time,
                    "output": str(output_path),
                }
            )

    result = {
        "step": "4_y_channel_attacks",
        "device": str(device),
        "image": image_path.name,
        "label": label,
        "clean_prediction": clean_prediction,
        "reports": reports,
        "author_rgb_constraints_passed": all(
            report["rgb_constraint_passed"] for report in reports
        ),
    }
    result_path = PROJECT_ROOT / "results" / "step04.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
