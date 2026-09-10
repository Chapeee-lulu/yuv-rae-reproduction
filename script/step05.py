# 第5步：验证等权logits集成

import json
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s05_imgn_ensemble import (
    ENSEMBLE_MODEL_NAMES,
    EqualWeightLogitsEnsemble,
    cal_ensemble_cross_entropy_loss,
)
from yuv_reproduction.s03_imgn_inference import (
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
)


@torch.inference_mode()
def main() -> None:  # 用一张真实图片验证集成结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    labels = load_generated_imgn_labels(PROJECT_ROOT / "results" / "step03_labels.csv")
    label = labels[image_path.name]
    with Image.open(image_path) as image:
        input_batch = add_batch_dimension(convert_image_to_unit_tensor(image)).to(device)

    models = [
        load_imgn_model(model_name, device)
        for model_name in ENSEMBLE_MODEL_NAMES
    ]
    individual_logits = [model(input_batch) for model in models]
    expected_logits = ( # 计算预期
        individual_logits[0] / 3
        + individual_logits[1] / 3
        + individual_logits[2] / 3
    )
    ensemble_logits = EqualWeightLogitsEnsemble(models)(input_batch) # 调用集成模型计算
    labels = torch.tensor([label], device=device)

    result = {
        "step": "5_s05_imgn_ensemble",
        "device": str(device),
        "image": image_path.name,
        "label": label,
        "models": list(ENSEMBLE_MODEL_NAMES),
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "individual_predictions": {
            model_name: int(logits.argmax(dim=1).item())
            for model_name, logits in zip(ENSEMBLE_MODEL_NAMES, individual_logits)
        },
        "ensemble_prediction": int(ensemble_logits.argmax(dim=1).item()),
        "ensemble_loss": float(
            cal_ensemble_cross_entropy_loss(ensemble_logits, labels).item()
        ),
        "strictly_equal_to_arithmetic_mean": torch.equal(
            ensemble_logits,
            expected_logits,
        ),
        "maximum_absolute_difference": float(
            (ensemble_logits - expected_logits).abs().max().item()
        ),
    }
    result_path = PROJECT_ROOT / "results" / "step05.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
