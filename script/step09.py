# 第9步：计算论文与恢复指标

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s09_metrics import (
    cal_attack_success_rate,
    cal_mean_ciede2000,
    cal_psnr,
    cal_recovery_metrics,
    cal_ssim,
)
from yuv_reproduction.s03_imgn_inference import (
    BLACK_BOX_MODEL_NAMES,
    WHITE_BOX_MODEL_NAMES,
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
    predict_class_indices,
)


def main() -> None:  # 计算一张真实RAE的全部指标
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    reproduction_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step08"
    sup_directory = PROJECT_ROOT / "outputs" / "sup" / "step08"
    rae_path = reproduction_directory / "rae.png"
    recovered_path = sup_directory / "recovered_rgb.png"
    if not rae_path.exists() or not recovered_path.exists():
        raise FileNotFoundError("请先运行step08.py")

    with Image.open(image_path) as image:
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        original_tensor = add_batch_dimension(convert_image_to_unit_tensor(image)).to(device)
    with Image.open(rae_path) as image:
        rae_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        rae_tensor = add_batch_dimension(convert_image_to_unit_tensor(image)).to(device)
    with Image.open(recovered_path) as image:
        recovered_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    labels = load_generated_imgn_labels(PROJECT_ROOT / "results" / "step03_labels.csv")
    label = labels[image_path.name]
    model_predictions: dict[str, dict[str, int | bool]] = {}
    strict_clean_values: list[int] = []
    strict_adversarial_values: list[int] = []
    strict_label_values: list[int] = []
    for model_name in WHITE_BOX_MODEL_NAMES + BLACK_BOX_MODEL_NAMES:
        model = load_imgn_model(model_name, device)
        clean_prediction = int(predict_class_indices(model, original_tensor).item())
        rae_prediction = int(predict_class_indices(model, rae_tensor).item())
        model_predictions[model_name] = {
            "clean_prediction": clean_prediction,
            "rae_prediction": rae_prediction,
            "clean_correct": clean_prediction == label,
            "attack_success": rae_prediction != label,
        }
        strict_clean_values.append(clean_prediction)
        strict_adversarial_values.append(rae_prediction)
        strict_label_values.append(label)
        del model

    white_count = len(WHITE_BOX_MODEL_NAMES)
    white_clean = strict_clean_values[:white_count]
    white_rae = strict_adversarial_values[:white_count]
    white_labels = strict_label_values[:white_count]
    black_clean = strict_clean_values[white_count:]
    black_rae = strict_adversarial_values[white_count:]
    black_labels = strict_label_values[white_count:]

    visual_metrics = {
        "psnr": cal_psnr(original_rgb, rae_rgb),
        "ssim": cal_ssim(original_rgb, rae_rgb),
        "mean_ciede2000": cal_mean_ciede2000(original_rgb, rae_rgb),
    }
    result = {
        "step": "9_paper_metrics",
        "device": str(device),
        "image": image_path.name,
        "label": label,
        "sample_count": 1,
        "visual_metrics": visual_metrics,
        "model_predictions": model_predictions,
        "white_box_asr_paper_denominator": cal_attack_success_rate(
            white_clean,
            white_rae,
            white_labels,
        ),
        "white_box_asr_clean_correct_only": cal_attack_success_rate(
            white_clean,
            white_rae,
            white_labels,
            clean_correct_only=True,
        ),
        "black_box_asr_paper_denominator": cal_attack_success_rate(
            black_clean,
            black_rae,
            black_labels,
        ),
        "black_box_asr_clean_correct_only": cal_attack_success_rate(
            black_clean,
            black_rae,
            black_labels,
            clean_correct_only=True,
        ),
        "exact_container_recovery": cal_recovery_metrics(
            original_rgb,
            recovered_rgb,
        ),
        "runtime_seconds": time.perf_counter() - start_time,
    }
    result_path = PROJECT_ROOT / "results" / "step09.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
