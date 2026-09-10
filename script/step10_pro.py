# 第10步Pro：GPU批处理与CPU并行

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from step10 import (
    cal_checkpoint_rows,
    load_attack_model,
    load_clean_prediction_rows,
    load_existing_results,
    predict_rgb_arrays,
    write_summary_csv,
)
from yuv_reproduction.s03_imgn_inference import (
    MODEL_NAMES,
    WHITE_BOX_MODEL_NAMES,
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_generated_imgn_labels,
    load_imgn_model,
)
from yuv_reproduction.s06_cam_mask import cam_mask, generate_gradcam_plus_plus_map
from yuv_reproduction.s08_rae import (
    load_recovery_png_sup,
    recover_png_sup,
    save_recovery_png_sup,
)
from yuv_reproduction.s1002_ablation import (
    ATTACK_NAMES,
    METHOD_NAMES,
    generate_method_adversarial_rgb,
)
from yuv_reproduction.s1003_pro import (
    finalize_rae_candidate_pro,
    prepare_rae_candidate_pro,
)
from yuv_reproduction.s1005_rdh import (
    finalize_rdh_candidate_pro,
    prepare_rdh_candidate_pro,
    select_rdh_superpixel_shape,
)


DEFAULT_CPU_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 2))


def parse_arguments_pro() -> argparse.Namespace:  # 读取Pro实验参数
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
    parser.add_argument("--saved-artifact-count", type=int, default=0)
    parser.add_argument("--gpu-batch-size", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=DEFAULT_CPU_WORKERS)
    parser.add_argument(
        "--source-model",
        choices=("auto",) + WHITE_BOX_MODEL_NAMES,
        default="auto",
    )
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_image_pro(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:  # 并行读取图片
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
        input_tensor = add_batch_dimension(convert_image_to_unit_tensor(rgb_image))
    return rgb_array, input_tensor


def load_experiment_images_pro(  # 多线程读取实验图片
    sample_count: int,
    cpu_workers: int,
) -> tuple[list[Path], list[np.ndarray], list[torch.Tensor], list[int]]:
    image_paths = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))
    labels_by_name = load_generated_imgn_labels(
        PROJECT_ROOT / "results" / "step03_labels.csv"
    )
    if not 1 <= sample_count <= len(image_paths):
        raise ValueError(f"样本数应在1至{len(image_paths)}")
    selected_paths = image_paths[:sample_count]
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        loaded = list(executor.map(load_image_pro, selected_paths))
    original_arrays = [item[0] for item in loaded]
    input_tensors = [item[1] for item in loaded]
    labels = [labels_by_name[path.name] for path in selected_paths]
    return selected_paths, original_arrays, input_tensors, labels


def generate_cam_masks_pro(  # GPU批量生成作者CAM掩膜
    input_tensors: list[torch.Tensor],
    device: torch.device,
    gpu_batch_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    cam_model = load_imgn_model("resnet50", device)
    masks: list[np.ndarray] = []
    target_categories: list[int] = []
    start = 0
    current_batch_size = gpu_batch_size
    while start < len(input_tensors):
        end = min(start + current_batch_size, len(input_tensors))
        try:
            input_batch = torch.cat(input_tensors[start:end], dim=0).to(device)
            cam_map, categories = generate_gradcam_plus_plus_map(
                cam_model,
                cam_model.layer4[-1],
                input_batch,
                interpolation_method="opencv",
                gradient_method="input",
            )
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            print(f"CAM显存不足，GPU批次降为{current_batch_size}", flush=True)
            continue
        binary_masks = cam_mask(cam_map, 0.5)[:, 0].cpu().numpy().astype(np.uint8)
        masks.extend([mask for mask in binary_masks])
        target_categories.extend([int(category) for category in categories.cpu().tolist()])
        start = end
    del cam_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return masks, target_categories


def create_reversible_batch_pro(  # 批量攻击并并行生成RAE
    method_name: str,
    attack_name: str,
    epsilon: float,
    attack_model: torch.nn.Module,
    inception_model: torch.nn.Module,
    original_arrays: list[np.ndarray],
    input_tensors: list[torch.Tensor],
    labels: list[int],
    cam_masks: list[np.ndarray],
    device: torch.device,
    maximum_outer_iterations: int,
    gpu_batch_size: int,
    cpu_executor: ProcessPoolExecutor,
    prepare_candidate_function=prepare_rae_candidate_pro,
    finalize_candidate_function=finalize_rae_candidate_pro,
) -> list[dict[str, object]]:
    batch_count = len(original_arrays)
    current_inputs = torch.cat(input_tensors, dim=0).to(device)
    label_tensor = torch.tensor(labels, device=device)
    active_indices = list(range(batch_count))
    final_candidates: list[dict[str, object] | None] = [None] * batch_count
    final_predictions = [int(label) for label in labels]
    final_iterations = [0] * batch_count

    for outer_iteration in range(1, maximum_outer_iterations + 1):
        if not active_indices:
            break
        active_tensor = torch.tensor(active_indices, device=device)
        adversarial_batch = generate_method_adversarial_rgb(
            method_name,
            attack_name,
            attack_model,
            current_inputs.index_select(0, active_tensor),
            label_tensor.index_select(0, active_tensor),
            epsilon,
        )
        adversarial_arrays = (
            adversarial_batch.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
        )
        futures = [
            cpu_executor.submit(
                prepare_candidate_function,
                original_arrays[index],
                adversarial_arrays[position],
                cam_masks[index],
            )
            for position, index in enumerate(active_indices)
        ]
        candidates = [future.result() for future in futures]
        valid_pairs = [
            (index, candidate)
            for index, candidate in zip(active_indices, candidates)
            if candidate["succeeded"]
        ]
        rae_predictions = predict_rgb_arrays(
            inception_model,
            [np.asarray(candidate["rae_rgb"]) for _, candidate in valid_pairs],
            device,
            batch_size=gpu_batch_size,
        ) if valid_pairs else []

        predictions_by_index = {
            index: int(prediction)
            for (index, _), prediction in zip(valid_pairs, rae_predictions)
        }
        next_active_indices = []
        for index, candidate in zip(active_indices, candidates):
            final_candidates[index] = candidate
            final_iterations[index] = outer_iteration
            if not candidate["succeeded"]:
                continue
            prediction = predictions_by_index[index]
            final_predictions[index] = prediction
            if prediction != labels[index]:
                continue
            attacked_tensor = (
                torch.from_numpy(np.asarray(candidate["attacked_rgb"]))
                .permute(2, 0, 1)
                .to(torch.float32)
                / 255
            )
            current_inputs[index] = attacked_tensor.to(device)
            next_active_indices.append(index)
        active_indices = next_active_indices

    final_futures = []
    final_results: list[dict[str, object] | None] = [None] * batch_count
    for index, candidate in enumerate(final_candidates):
        if candidate is None:
            final_results[index] = {
                "succeeded": False,
                "error": "未生成RAE候选",
            }
        elif not candidate["succeeded"]:
            final_results[index] = candidate
        else:
            future = cpu_executor.submit(
                finalize_candidate_function,
                original_arrays[index],
                candidate,
                labels[index],
                final_predictions[index],
                final_iterations[index],
            )
            final_futures.append((index, future))
    for index, future in final_futures:
        final_results[index] = future.result()
    return [result for result in final_results if result is not None]


def create_reversible_examples_pro(  # 显存自适应分批生成RAE
    method_name: str,
    attack_name: str,
    epsilon: float,
    attack_model: torch.nn.Module,
    inception_model: torch.nn.Module,
    original_arrays: list[np.ndarray],
    input_tensors: list[torch.Tensor],
    labels: list[int],
    cam_masks: list[np.ndarray],
    device: torch.device,
    maximum_outer_iterations: int,
    gpu_batch_size: int,
    cpu_executor: ProcessPoolExecutor,
    prepare_candidate_function=prepare_rae_candidate_pro,
    finalize_candidate_function=finalize_rae_candidate_pro,
) -> list[dict[str, object]]:
    generated_results = []
    start = 0
    current_batch_size = gpu_batch_size
    while start < len(original_arrays):
        end = min(start + current_batch_size, len(original_arrays))
        try:
            generated_results.extend(
                create_reversible_batch_pro(
                    method_name,
                    attack_name,
                    epsilon,
                    attack_model,
                    inception_model,
                    original_arrays[start:end],
                    input_tensors[start:end],
                    labels[start:end],
                    cam_masks[start:end],
                    device,
                    maximum_outer_iterations,
                    current_batch_size,
                    cpu_executor,
                    prepare_candidate_function,
                    finalize_candidate_function,
                )
            )
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            print(f"显存不足，GPU批次降为{current_batch_size}", flush=True)
            continue
        start = end
    return generated_results


def save_artifacts_pro(  # 保存选定的Pro产物
    method_name: str,
    attack_name: str,
    epsilon_pixels: int,
    image_paths: list[Path],
    original_arrays: list[np.ndarray],
    generated_results: list[dict[str, object]],
    saved_artifact_count: int,
    reproduction_directory: Path,
    sup_directory: Path,
) -> None:
    for image_index, generated in enumerate(generated_results):
        if not generated["succeeded"]:
            continue
        artifact_kind = generated.pop("_artifact_kind", "yuv")
        if artifact_kind == "rdh":
            recovered_rgb = generated.pop("_recovered_rgb")
            generated["reproduction_rgb_path"] = None
            generated["recovered_rgb_path"] = None
            generated["sup_png_path"] = None
            generated["sup_visible_rgb_matches_author"] = None
            generated["sup_perturbation_recovered"] = None
            generated["sup_png_exact_recovery"] = None
            if image_index < saved_artifact_count:
                artifact_stem = (
                    f"{method_name}_{attack_name}_{epsilon_pixels}_255_"
                    f"{image_paths[image_index].stem}"
                )
                reproduction_path = reproduction_directory / f"{artifact_stem}.png"
                recovered_path = reproduction_directory / f"{artifact_stem}_recovered.png"
                Image.fromarray(generated["rae_rgb"]).save(reproduction_path)
                Image.fromarray(recovered_rgb).save(recovered_path)
                generated["reproduction_rgb_path"] = str(reproduction_path)
                generated["recovered_rgb_path"] = str(recovered_path)
            continue
        reversible_yuv = generated.pop("_reversible_yuv")
        conversion_residual = generated.pop("_rgb_conversion_residual")
        masked_perturbation = generated.pop("_masked_perturbation")
        arithmetic_metadata = generated.pop("_arithmetic_metadata")
        generated["reproduction_rgb_path"] = None
        generated["sup_png_path"] = None
        generated["sup_visible_rgb_matches_author"] = None
        generated["sup_perturbation_recovered"] = None
        generated["sup_png_exact_recovery"] = None
        if image_index >= saved_artifact_count:
            continue
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
        sup_visible_rgb, _, _, _ = load_recovery_png_sup(sup_path)
        sup_recovered_rgb, sup_perturbation = recover_png_sup(sup_path)
        generated["reproduction_rgb_path"] = str(reproduction_path)
        generated["sup_png_path"] = str(sup_path)
        generated["sup_visible_rgb_matches_author"] = bool(
            np.array_equal(sup_visible_rgb, generated["rae_rgb"])
        )
        generated["sup_perturbation_recovered"] = bool(
            np.array_equal(sup_perturbation, masked_perturbation)
        )
        generated["sup_png_exact_recovery"] = bool(
            np.array_equal(sup_recovered_rgb, original_arrays[image_index])
        )


def main() -> None:  # 运行Pro并行实验
    arguments = parse_arguments_pro()
    if arguments.max_outer_iterations <= 0:
        raise ValueError("最大外循环次数必须为正数")
    if arguments.gpu_batch_size <= 0:
        raise ValueError("GPU批次必须为正数")
    if arguments.cpu_workers <= 0:
        raise ValueError("CPU进程数必须为正数")
    if not 0 <= arguments.saved_artifact_count <= arguments.sample_count:
        raise ValueError("保存文件数应在0至样本数之间")
    if arguments.result_suffix and not arguments.result_suffix.replace("_", "").isalnum():
        raise ValueError("结果后缀只能包含字母、数字和下划线")
    if arguments.source_model != "auto" and not arguments.result_suffix:
        raise ValueError("指定单源模型时必须设置独立的结果后缀")

    torch.manual_seed(2022)
    np.random.seed(2022)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = METHOD_NAMES if arguments.method == "all" else (arguments.method,)
    attacks = ATTACK_NAMES if arguments.attack == "all" else (arguments.attack,)
    image_paths, original_arrays, input_tensors, labels = load_experiment_images_pro(
        arguments.sample_count,
        arguments.cpu_workers,
    )
    clean_rows = load_clean_prediction_rows()
    missing_clean_rows = [path.name for path in image_paths if path.name not in clean_rows]
    if missing_clean_rows:
        raise ValueError(f"缺少原图预测：{missing_clean_rows[0]}")

    print(
        f"设备：{device}；样本数：{len(image_paths)}；"
        f"GPU批次：{arguments.gpu_batch_size}；CPU进程：{arguments.cpu_workers}；"
        f"源模型：{arguments.source_model}",
        flush=True,
    )
    print("正在批量生成共用的作者CAM掩膜", flush=True)
    cam_masks, cam_categories = generate_cam_masks_pro(
        input_tensors,
        device,
        arguments.gpu_batch_size,
    )
    reproduction_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step10_pro"
    sup_directory = PROJECT_ROOT / "outputs" / "sup" / "step10_pro"
    reproduction_directory.mkdir(parents=True, exist_ok=True)
    sup_directory.mkdir(parents=True, exist_ok=True)
    result_directory = PROJECT_ROOT / "results"
    result_stem = f"step10_pro_{arguments.sample_count}_images"
    if arguments.result_suffix:
        result_stem = f"{result_stem}_{arguments.result_suffix}"
    summary_path = result_directory / f"{result_stem}.csv"
    detail_path = result_directory / f"{result_stem}.json"
    if arguments.resume:
        all_rows, all_details = load_existing_results(summary_path, detail_path)
    else:
        all_rows, all_details = [], []
    completed_configurations = {
        (
            detail["method"],
            detail["attack"],
            detail["epsilon"],
            detail.get("source_model", "auto"),
        )
        for detail in all_details
    }

    process_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=arguments.cpu_workers,
        mp_context=process_context,
    ) as cpu_executor:
        for method_name in methods:
            selected_source_model = (
                None if arguments.source_model == "auto" else arguments.source_model
            )
            attack_model, available_models = load_attack_model(
                method_name,
                device,
                selected_source_model,
            )
            success_model_name = selected_source_model or "inception_v3"
            success_model = available_models[success_model_name]
            for epsilon_pixels in arguments.epsilon_pixels:
                epsilon = epsilon_pixels / 255
                for attack_name in attacks:
                    configuration_key = (
                        method_name,
                        attack_name,
                        f"{epsilon_pixels}/255",
                        arguments.source_model,
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
                    prepare_candidate_function = prepare_rae_candidate_pro
                    finalize_candidate_function = finalize_rae_candidate_pro
                    if method_name == "RDH":
                        block_shape = select_rdh_superpixel_shape(
                            attack_name,
                            epsilon_pixels,
                        )
                        prepare_candidate_function = partial(
                            prepare_rdh_candidate_pro,
                            block_shape=block_shape,
                        )
                        finalize_candidate_function = finalize_rdh_candidate_pro
                    generated_results = create_reversible_examples_pro(
                        method_name,
                        attack_name,
                        epsilon,
                        attack_model,
                        success_model,
                        original_arrays,
                        input_tensors,
                        labels,
                        cam_masks,
                        device,
                        arguments.max_outer_iterations,
                        arguments.gpu_batch_size,
                        cpu_executor,
                        prepare_candidate_function,
                        finalize_candidate_function,
                    )
                    for generated in generated_results:
                        if generated["succeeded"]:
                            generated["source_model"] = success_model_name
                            generated["rae_prediction_source"] = generated[
                                "rae_prediction_inception"
                            ]
                            generated["source_attack_success"] = generated[
                                "inception_attack_success"
                            ]
                    save_artifacts_pro(
                        method_name,
                        attack_name,
                        epsilon_pixels,
                        image_paths,
                        original_arrays,
                        generated_results,
                        arguments.saved_artifact_count,
                        reproduction_directory,
                        sup_directory,
                    )
                    rae_arrays = [
                        generated["rae_rgb"]
                        if generated["succeeded"]
                        else original_rgb
                        for generated, original_rgb in zip(
                            generated_results,
                            original_arrays,
                        )
                    ]
                    predictions: dict[str, list[int]] = {}
                    for model_name in MODEL_NAMES:
                        temporary_model = model_name not in available_models
                        model = (
                            load_imgn_model(model_name, device)
                            if temporary_model
                            else available_models[model_name]
                        )
                        predictions[model_name] = predict_rgb_arrays(
                            model,
                            rae_arrays,
                            device,
                            batch_size=arguments.gpu_batch_size,
                        )
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
                        selected_source_model,
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
                            "gpu_batch_size": arguments.gpu_batch_size,
                            "cpu_workers": arguments.cpu_workers,
                            "source_model": arguments.source_model,
                            "samples": detail_results,
                        }
                    )
                    detail_path.write_text(
                        json.dumps(all_details, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if all_rows:
                        write_summary_csv(summary_path, all_rows)
                    print(f"用时{runtime_seconds:.2f}秒", flush=True)
            del attack_model, available_models, success_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if summary_path.exists():
        print(f"Pro汇总已保存：{summary_path}", flush=True)
    else:
        print("没有成功样本，未生成Pro汇总CSV", flush=True)
    print(f"Pro明细已保存：{detail_path}", flush=True)


if __name__ == "__main__":
    main()
