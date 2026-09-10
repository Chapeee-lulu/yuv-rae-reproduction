# 第3步：执行ImageNet模型推理

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


# 定位目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s03_imgn_inference import (
    BLACK_BOX_MODEL_NAMES,
    MODEL_NAMES,
    WHITE_BOX_MODEL_NAMES,
    ImgnPngDataset,
    add_batch_dimension,
    convert_image_to_unit_tensor,
    generate_imgn_labels,
    load_imgn_val_labels,
    load_imgn_model,
    imgn_input_cmp,
    predict_class_indices,
    predict_imgn_dataset,
)


def parse_arguments() -> argparse.Namespace:  # 选择是否归一化处理
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-cmp", action="store_true")
    return parser.parse_args()


def main() -> None:  # 执行第3步检查
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))
    if len(image_paths) != 1000:
        raise RuntimeError(f"应有1000张，实际找到{len(image_paths)}")

    dataset = ImgnPngDataset(
        image_paths,
        normalize_input=arguments.normalized_cmp,
    )

    # 检查首张图片的预处理
    with Image.open(image_paths[0]) as image:
        unit_tensor = convert_image_to_unit_tensor(image)
    model_input_tensor = (
        imgn_input_cmp(unit_tensor)
        if arguments.normalized_cmp
        else unit_tensor
    )
    first_input_batch = add_batch_dimension(model_input_tensor).to(device)

    model_predictions: dict[str, list[int]] = {}
    model_reports: dict[str, dict[str, object]] = {}

    # 依次加载模型
    for model_name in MODEL_NAMES:
        start_time = time.perf_counter()
        model = load_imgn_model(model_name, device)
        first_prediction = int(predict_class_indices(model, first_input_batch).item())
        predictions, filenames = predict_imgn_dataset(
            model,
            dataset,
            device,
        )
        if filenames != [path.name for path in image_paths]:
            raise RuntimeError("推理期间数据集顺序发生变化")

        model_predictions[model_name] = predictions
        model_reports[model_name] = {
            "role": "white_box" if model_name in WHITE_BOX_MODEL_NAMES else "black_box",
            "first_image_prediction": first_prediction,
            "runtime_seconds": time.perf_counter() - start_time,
        }
        del model
        torch.cuda.empty_cache()

    generated_labels = generate_imgn_labels(model_predictions)
    val_labels_by_name = load_imgn_val_labels(
        PROJECT_ROOT / "data" / "val" / "imgn_labels.csv"
    )
    expected_names = [path.name for path in image_paths]
    if set(val_labels_by_name) != set(expected_names):
        raise RuntimeError("验证集标签与原始图片不完整对应")
    val_labels = [val_labels_by_name[name] for name in expected_names]
    white_box_consensus = [label is not None for label in generated_labels]
    generated_label_matches_val = [
        generated_label == val_label
        for generated_label, val_label in zip(generated_labels, val_labels)
    ]
    jointly_correct = generated_label_matches_val

    for model_name in MODEL_NAMES:
        predictions = model_predictions[model_name]
        correct_count = sum(
            prediction == val_label
            for prediction, val_label in zip(predictions, val_labels)
        )
        model_reports[model_name].update(
            {
                "first_image_val_label": val_labels[0],
                "correct_count": correct_count,
                "accuracy": correct_count / len(val_labels),
            }
        )

    # 保存逐图预测
    result_suffix = "_normalized_cmp" if arguments.normalized_cmp else ""
    result_directory = PROJECT_ROOT / "results"
    result_directory.mkdir(parents=True, exist_ok=True)
    csv_path = result_directory / f"step03_predictions{result_suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "filename",
            "generated_label",
            "val_label",
            *MODEL_NAMES,
            "white_box_consensus",
            "generated_label_matches_val",
            "white_box_jointly_correct",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            row = {
                "filename": image_path.name,
                "generated_label": (
                    generated_labels[index]
                    if generated_labels[index] is not None
                    else ""
                ),
                "val_label": val_labels[index],
                "white_box_consensus": white_box_consensus[index],
                "generated_label_matches_val": generated_label_matches_val[index],
                "white_box_jointly_correct": jointly_correct[index],
            }
            row.update(
                {
                    model_name: model_predictions[model_name][index]
                    for model_name in MODEL_NAMES
                }
            )
            writer.writerow(row)

    generated_label_path = result_directory / f"step03_labels{result_suffix}.csv"
    with generated_label_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "filename",
            "generated_label",
            "val_label",
            "white_box_consensus",
            "matches_val",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            writer.writerow(
                {
                    "filename": image_path.name,
                    "generated_label": (
                        generated_labels[index]
                        if generated_labels[index] is not None
                        else ""
                    ),
                    "val_label": val_labels[index],
                    "white_box_consensus": white_box_consensus[index],
                    "matches_val": generated_label_matches_val[index],
                }
            )

    # 保存汇总结果
    result = {
        "step": "3_imgn_inference",
        "device": str(device),
        "mode": "normalized_cmp" if arguments.normalized_cmp else "reproduction",
        "image_count": len(image_paths),
        "image_size": [299, 299],
        "model_input_tensor_range": [
            float(model_input_tensor.min()),
            float(model_input_tensor.max()),
        ],
        "normalization": "imgn" if arguments.normalized_cmp else "none",
        "single_image_batch_shape": list(first_input_batch.shape),
        "white_box_models": list(WHITE_BOX_MODEL_NAMES),
        "black_box_models": list(BLACK_BOX_MODEL_NAMES),
        "models": model_reports,
        "generated_label_count": sum(white_box_consensus),
        "generated_label_matches_val_count": sum(generated_label_matches_val),
        "white_box_jointly_correct_count": sum(jointly_correct),
        "all_images_white_box_jointly_correct": all(jointly_correct),
    }
    json_path = result_directory / f"step03{result_suffix}.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"汇总已保存：{json_path}")
    print(f"预测结果已保存：{csv_path}")
    print(f"生成标签已保存：{generated_label_path}")


if __name__ == "__main__":
    main()
