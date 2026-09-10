# 第10步：运行论文方法实验

import argparse
import csv
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

from yuv_reproduction.s06_cam_mask import (
    cam_mask,
    generate_gradcam_plus_plus_map,
)
from yuv_reproduction.s09_metrics import (
    cal_attack_success_rate,
    cal_mean_ciede2000,
    cal_psnr,
    cal_recovery_metrics,
    cal_ssim,
)
from yuv_reproduction.s05_imgn_ensemble import (
    ENSEMBLE_MODEL_NAMES,
    EqualWeightLogitsEnsemble,
)
from yuv_reproduction.s03_imgn_inference import (
    BLACK_BOX_MODEL_NAMES,
    MODEL_NAMES,
    WHITE_BOX_MODEL_NAMES,
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
    predict_class_indices,
)
from yuv_reproduction.s1002_ablation import (
    ATTACK_NAMES,
    METHOD_NAMES,
    apply_binary_mask_to_integer_y_perturbation,
    cal_integer_y_perturbation,
    generate_method_adversarial_rgb,
)
from yuv_reproduction.s08_rae import (
    cal_rgb_residual_sup,
    yuv_to_rgb_uint8,
    generate_rae,
    load_recovery_png_sup,
    recover_png_sup,
    recover_rgb_exact_sup,
    recover_yuv_payload_sup,
    save_recovery_png_sup,
)
from yuv_reproduction.s02_rgb_yuv import rgb_to_yuv


def parse_arguments() -> argparse.Namespace:  # 读取实验参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--method", choices=("all",) + METHOD_NAMES, default="all")
    parser.add_argument("--attack", choices=("all",) + ATTACK_NAMES, default="all")
    parser.add_argument(
        "--epsilon-pixels",
        type=int,
        choices=(2, 4),
        nargs="+",
        default=[2, 4],
    )
    parser.add_argument("--max-outer-iterations", type=int, default=50)
    parser.add_argument("--saved-artifact-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_experiment_images(  # 读取实验图片
    sample_count: int,
) -> tuple[list[Path], list[np.ndarray], list[torch.Tensor], list[int]]:
    image_paths = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))
    labels_by_name = load_generated_imgn_labels(
        PROJECT_ROOT / "results" / "step03_labels.csv"
    )
    if not 1 <= sample_count <= len(image_paths):
        raise ValueError(f"样本数应在1至{len(image_paths)}")
    selected_paths = image_paths[:sample_count]
    original_arrays = []
    input_tensors = []
    labels = []
    for image_path in selected_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            original_arrays.append(np.asarray(rgb_image, dtype=np.uint8))
            input_tensors.append(add_batch_dimension(convert_image_to_unit_tensor(rgb_image)))
        labels.append(labels_by_name[image_path.name])
    return selected_paths, original_arrays, input_tensors, labels


def generate_cam_masks(  # 生成统一作者CAM掩膜
    input_tensors: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[np.ndarray], list[int]]:
    cam_model = load_imgn_model("resnet50", device)
    masks = []
    target_categories = []
    for input_tensor in input_tensors:
        cam_map, categories = generate_gradcam_plus_plus_map(
            cam_model,
            cam_model.layer4[-1],
            input_tensor.to(device),
            interpolation_method="opencv",
            gradient_method="input",
        )
        mask = cam_mask(cam_map, 0.5)
        masks.append(mask[0, 0].cpu().numpy().astype(np.uint8))
        target_categories.append(int(categories.item()))
    del cam_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return masks, target_categories


def load_attack_model(  # 加载单模型或集成模型
    method_name: str,
    device: torch.device,
    source_model_name: str | None = None,
) -> tuple[torch.nn.Module, dict[str, torch.nn.Module]]:
    if source_model_name is not None:
        source_model = load_imgn_model(source_model_name, device)
        return source_model, {source_model_name: source_model}
    if method_name in ("RDH", "YUV", "YACK"):
        inception = load_imgn_model("inception_v3", device)
        return inception, {"inception_v3": inception}
    models = {
        model_name: load_imgn_model(model_name, device)
        for model_name in ENSEMBLE_MODEL_NAMES
    }
    ensemble = EqualWeightLogitsEnsemble(
        [models[model_name] for model_name in ENSEMBLE_MODEL_NAMES]
    ).eval()
    return ensemble, models


def create_reversible_example(  # 按作者循环生成单张RAE
    method_name: str,
    attack_name: str,
    epsilon: float,
    attack_model: torch.nn.Module,
    inception_model: torch.nn.Module,
    original_rgb: np.ndarray,
    input_tensor: torch.Tensor,
    label: int,
    cam_mask: np.ndarray,
    device: torch.device,
    maximum_outer_iterations: int,
) -> dict[str, object]:
    original_yuv = rgb_to_yuv(original_rgb)
    current_input = input_tensor.to(device)
    labels = torch.tensor([label], device=device)
    last_error = ""
    for outer_iteration in range(1, maximum_outer_iterations + 1):
        adversarial_rgb = generate_method_adversarial_rgb(
            method_name,
            attack_name,
            attack_model,
            current_input,
            labels,
            epsilon,
        )
        y_perturbation = cal_integer_y_perturbation(original_rgb, adversarial_rgb)
        masked_perturbation = apply_binary_mask_to_integer_y_perturbation(
            y_perturbation,
            cam_mask,
        )
        try:
            reversible_yuv, embedding = generate_rae(
                original_yuv,
                masked_perturbation,
            )
        except ValueError as error:
            last_error = str(error)
            break
        rae_rgb = yuv_to_rgb_uint8(reversible_yuv)
        rae_tensor = add_batch_dimension(
            convert_image_to_unit_tensor(Image.fromarray(rae_rgb))
        ).to(device)
        rae_prediction = int(predict_class_indices(inception_model, rae_tensor).item())
        if rae_prediction != label:
            break
        attacked_yuv = original_yuv.copy()
        attacked_yuv[:, :, 0] = (
            original_yuv[:, :, 0].astype(np.int32) + masked_perturbation
        )
        attacked_rgb = yuv_to_rgb_uint8(attacked_yuv)
        current_input = add_batch_dimension(
            convert_image_to_unit_tensor(Image.fromarray(attacked_rgb))
        ).to(device)
    if last_error:
        return {"succeeded": False, "error": last_error}

    recovered_yuv, recovered_perturbation = (
        recover_yuv_payload_sup(
            reversible_yuv,
            int(embedding["threshold"]),
            embedding["arithmetic_metadata"],
        )
    )
    residual = cal_rgb_residual_sup(original_rgb, original_yuv)
    recovered_rgb = recover_rgb_exact_sup(recovered_yuv, residual)
    recovery = cal_recovery_metrics(original_rgb, recovered_rgb)
    perturbation_recovered = bool(
        np.array_equal(recovered_perturbation, masked_perturbation)
    )
    return {
        "succeeded": True,
        "rae_rgb": rae_rgb,
        "rae_prediction_inception": rae_prediction,
        "inception_attack_success": rae_prediction != label,
        "outer_iterations": outer_iteration,
        "embedding_threshold": int(embedding["threshold"]),
        "payload_length": int(embedding["payload_length"]),
        "masked_nonzero_count": int(np.count_nonzero(masked_perturbation)),
        "perturbation_recovered": perturbation_recovered,
        "exact_recovery": recovery,
        "paper_exact_recovery": None,
        "psnr": cal_psnr(original_rgb, rae_rgb),
        "ssim": cal_ssim(original_rgb, rae_rgb),
        "ciede2000": cal_mean_ciede2000(original_rgb, rae_rgb),
        "_reversible_yuv": reversible_yuv,
        "_rgb_conversion_residual": residual,
        "_masked_perturbation": masked_perturbation,
        "_arithmetic_metadata": embedding["arithmetic_metadata"],
    }


def predict_rgb_arrays(  # 分批预测RGB数组
    model: torch.nn.Module,
    rgb_arrays: list[np.ndarray],
    device: torch.device,
    batch_size: int = 8,
) -> list[int]:
    predictions = []
    for start in range(0, len(rgb_arrays), batch_size):
        batch_arrays = np.stack(rgb_arrays[start : start + batch_size])
        batch = torch.from_numpy(batch_arrays).permute(0, 3, 1, 2).to(torch.float32) / 255
        predictions.extend(predict_class_indices(model, batch.to(device)).cpu().tolist())
    return predictions


def load_clean_prediction_rows() -> dict[str, dict[str, str]]:  # 读取第3步干净预测
    prediction_path = PROJECT_ROOT / "results" / "step03_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("请先运行script/step03.py")
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["filename"]: row for row in csv.DictReader(file)}


def cal_checkpoint_rows(  # 汇总1、10和最终样本结果
    method_name: str,
    attack_name: str,
    epsilon_pixels: int,
    image_paths: list[Path],
    labels: list[int],
    generated_results: list[dict[str, object]],
    predictions: dict[str, list[int]],
    clean_rows: dict[str, dict[str, str]],
    runtime_seconds: float,
    source_model_name: str | None = None,
) -> list[dict[str, object]]:
    checkpoints = [count for count in (1, 10, 1000) if count <= len(image_paths)]
    if len(image_paths) not in checkpoints:
        checkpoints.append(len(image_paths))
    rows = []
    for checkpoint in checkpoints:
        subset_results = generated_results[:checkpoint]
        valid_indices = [
            index for index, result in enumerate(subset_results) if result["succeeded"]
        ]
        if not valid_indices:
            continue
        visual_keys = ("psnr", "ssim", "ciede2000")
        visual_means = {
            key: float(np.mean([subset_results[index][key] for index in valid_indices]))
            for key in visual_keys
        }
        exact_recovery_rate = float(
            np.mean(
                [
                    bool(subset_results[index]["exact_recovery"]["exact_rgb_recovery"])
                    for index in valid_indices
                ]
            )
        )
        paper_recovery_values = [
            bool(subset_results[index]["paper_exact_recovery"])
            for index in valid_indices
            if subset_results[index].get("paper_exact_recovery") is not None
        ]
        paper_recovery_rate = (
            float(np.mean(paper_recovery_values))
            if paper_recovery_values
            else None
        )
        sup_indices = [
            index
            for index in valid_indices
            if subset_results[index].get("sup_png_exact_recovery") is not None
        ]
        sup_recovery_rate = (
            float(
                np.mean(
                    [
                        bool(
                            subset_results[index][
                                "sup_png_exact_recovery"
                            ]
                        )
                        for index in sup_indices
                    ]
                )
            )
            if sup_indices
            else None
        )
        maximum_channel_error = max(
            int(subset_results[index]["exact_recovery"]["maximum_channel_error"])
            for index in valid_indices
        )
        for model_name in MODEL_NAMES:
            clean = [
                int(clean_rows[image_paths[index].name][model_name]) for index in valid_indices
            ]
            adversarial = [predictions[model_name][index] for index in valid_indices]
            valid_labels = [labels[index] for index in valid_indices]
            paper_asr = cal_attack_success_rate(clean, adversarial, valid_labels)
            strict_asr = cal_attack_success_rate(
                clean,
                adversarial,
                valid_labels,
                clean_correct_only=True,
            )
            row = {
                    "method": method_name,
                    "attack": attack_name,
                    "epsilon": f"{epsilon_pixels}/255",
                    "model": model_name,
                    "model_role": (
                        "white_box" if model_name in WHITE_BOX_MODEL_NAMES else "black_box"
                    ),
                    "sample_count": checkpoint,
                    "generated_count": len(valid_indices),
                    "psnr": visual_means["psnr"],
                    "ssim": visual_means["ssim"],
                    "ciede2000": visual_means["ciede2000"],
                    "asr_paper_denominator": paper_asr["attack_success_rate"],
                    "asr_clean_correct_only": strict_asr["attack_success_rate"],
                    "strict_asr_sample_count": strict_asr["sample_count"],
                    "exact_container_recovery_rate": exact_recovery_rate,
                    "paper_rgb_recovery_rate": paper_recovery_rate,
                    "sup_png_verified_count": len(sup_indices),
                    "sup_png_recovery_rate": sup_recovery_rate,
                    "maximum_channel_error": maximum_channel_error,
                    "runtime_seconds": runtime_seconds,
                }
            if source_model_name is not None:
                row["source_model"] = source_model_name
            rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:  # 保存汇总表
    if not rows:
        raise ValueError("未生成实验结果")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_results(  # 读取断点结果
    summary_path: Path,
    detail_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not summary_path.exists() and not detail_path.exists():
        return [], []
    if not summary_path.exists() or not detail_path.exists():
        raise ValueError("断点续跑需要汇总CSV和明细JSON")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    details = json.loads(detail_path.read_text(encoding="utf-8"))
    return rows, details


def main() -> None:  # 运行选定消融实验
    arguments = parse_arguments()
    if arguments.max_outer_iterations <= 0:
        raise ValueError("最大外循环次数必须为正数")
    if not 0 <= arguments.saved_artifact_count <= arguments.sample_count:
        raise ValueError("保存文件数应在0至样本数之间")
    torch.manual_seed(2022)
    np.random.seed(2022)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = METHOD_NAMES if arguments.method == "all" else (arguments.method,)
    attacks = ATTACK_NAMES if arguments.attack == "all" else (arguments.attack,)
    image_paths, original_arrays, input_tensors, labels = load_experiment_images(
        arguments.sample_count
    )
    clean_rows = load_clean_prediction_rows()
    missing_clean_rows = [path.name for path in image_paths if path.name not in clean_rows]
    if missing_clean_rows:
        raise ValueError(f"缺少原图预测：{missing_clean_rows[0]}")

    print(f"设备：{device}；样本数：{len(image_paths)}", flush=True)
    print("正在生成共用的作者CAM掩膜", flush=True)
    cam_masks, cam_categories = generate_cam_masks(input_tensors, device)
    reproduction_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step10"
    sup_directory = PROJECT_ROOT / "outputs" / "sup" / "step10"
    reproduction_directory.mkdir(parents=True, exist_ok=True)
    sup_directory.mkdir(parents=True, exist_ok=True)
    result_directory = PROJECT_ROOT / "results"
    summary_path = result_directory / (
        f"step10_{arguments.sample_count}_images.csv"
    )
    detail_path = result_directory / (
        f"step10_{arguments.sample_count}_images.json"
    )
    if arguments.resume:
        all_rows, all_details = load_existing_results(summary_path, detail_path)
    else:
        all_rows, all_details = [], []
    completed_configurations = {
        (detail["method"], detail["attack"], detail["epsilon"])
        for detail in all_details
    }

    for method_name in methods:
        attack_model, available_models = load_attack_model(method_name, device)
        inception_model = available_models["inception_v3"]
        for epsilon_pixels in arguments.epsilon_pixels:
            epsilon = epsilon_pixels / 255
            for attack_name in attacks:
                configuration_key = (
                    method_name,
                    attack_name,
                    f"{epsilon_pixels}/255",
                )
                if configuration_key in completed_configurations:
                    print(
                        f"跳过已完成的{method_name} {attack_name} "
                        f"{epsilon_pixels}/255",
                        flush=True,
                    )
                    continue
                configuration_start = time.perf_counter()
                print(
                    f"正在运行{method_name} {attack_name} {epsilon_pixels}/255",
                    flush=True,
                )
                generated_results = []
                rae_arrays = []
                for image_index, (original_rgb, input_tensor, label, cam_mask) in enumerate(
                    zip(original_arrays, input_tensors, labels, cam_masks)
                ):
                    generated = create_reversible_example(
                        method_name,
                        attack_name,
                        epsilon,
                        attack_model,
                        inception_model,
                        original_rgb,
                        input_tensor,
                        label,
                        cam_mask,
                        device,
                        arguments.max_outer_iterations,
                    )
                    if generated["succeeded"]:
                        reversible_yuv = generated.pop("_reversible_yuv")
                        conversion_residual = generated.pop(
                            "_rgb_conversion_residual"
                        )
                        masked_perturbation = generated.pop("_masked_perturbation")
                        arithmetic_metadata = generated.pop("_arithmetic_metadata")
                        generated["reproduction_rgb_path"] = None
                        generated["sup_png_path"] = None
                        generated["sup_visible_rgb_matches_author"] = None
                        generated["sup_perturbation_recovered"] = None
                        generated["sup_png_exact_recovery"] = None
                        if image_index < arguments.saved_artifact_count:
                            artifact_stem = (
                                f"{method_name}_{attack_name}_{epsilon_pixels}_255_"
                                f"{image_paths[image_index].stem}"
                            )
                            reproduction_path = reproduction_directory / f"{artifact_stem}.png"
                            sup_path = sup_directory / f"{artifact_stem}.png"
                            Image.fromarray(generated["rae_rgb"]).save(reproduction_path)
                            save_recovery_png_sup(
                                sup_path,
                                generated["rae_rgb"],
                                reversible_yuv,
                                int(generated["embedding_threshold"]),
                                conversion_residual,
                                arithmetic_metadata,
                            )
                            sup_visible_rgb, _, _, _ = (
                                load_recovery_png_sup(sup_path)
                            )
                            sup_recovered_rgb, sup_perturbation = (
                                recover_png_sup(
                                    sup_path
                                )
                            )
                            generated["reproduction_rgb_path"] = str(reproduction_path)
                            generated["sup_png_path"] = str(
                                sup_path
                            )
                            generated["sup_visible_rgb_matches_author"] = bool(
                                np.array_equal(
                                    sup_visible_rgb,
                                    generated["rae_rgb"],
                                )
                            )
                            generated["sup_perturbation_recovered"] = bool(
                                np.array_equal(
                                    sup_perturbation,
                                    masked_perturbation,
                                )
                            )
                            generated["sup_png_exact_recovery"] = bool(
                                np.array_equal(
                                    sup_recovered_rgb,
                                    original_rgb,
                                )
                            )
                    generated_results.append(generated)
                    rae_arrays.append(
                        generated["rae_rgb"] if generated["succeeded"] else original_rgb
                    )
                predictions: dict[str, list[int]] = {}
                for model_name in MODEL_NAMES:
                    temporary_model = model_name not in available_models
                    model = (
                        load_imgn_model(model_name, device)
                        if temporary_model
                        else available_models[model_name]
                    )
                    predictions[model_name] = predict_rgb_arrays(model, rae_arrays, device)
                    if temporary_model:
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                runtime_seconds = time.perf_counter() - configuration_start
                rows = cal_checkpoint_rows(
                    method_name,
                    attack_name,
                    epsilon_pixels,
                    image_paths,
                    labels,
                    generated_results,
                    predictions,
                    clean_rows,
                    runtime_seconds,
                )
                all_rows.extend(rows)
                detail_results = []
                for image_path, label, cam_category, generated in zip(
                    image_paths,
                    labels,
                    cam_categories,
                    generated_results,
                ):
                    detail_results.append(
                        {
                            "image": image_path.name,
                            "label": label,
                            "cam_target_category": cam_category,
                            **{
                                key: value
                                for key, value in generated.items()
                                if key != "rae_rgb"
                            },
                        }
                    )
                all_details.append(
                    {
                        "method": method_name,
                        "attack": attack_name,
                        "epsilon": f"{epsilon_pixels}/255",
                        "runtime_seconds": runtime_seconds,
                        "samples": detail_results,
                    }
                )
                write_summary_csv(summary_path, all_rows)
                detail_path.write_text(
                    json.dumps(all_details, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"用时{runtime_seconds:.2f}秒", flush=True)
        del attack_model, available_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"汇总已保存：{summary_path}", flush=True)
    print(f"明细已保存：{detail_path}", flush=True)


if __name__ == "__main__":
    main()
