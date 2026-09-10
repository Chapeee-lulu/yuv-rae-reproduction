# 第8步：验证RAE生成与恢复闭环

import argparse
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
from yuv_reproduction.s05_imgn_ensemble import (
    ENSEMBLE_MODEL_NAMES,
    EqualWeightLogitsEnsemble,
)
from yuv_reproduction.s03_imgn_inference import (
    add_batch_dimension,
    convert_image_to_unit_tensor,
    load_imgn_model,
    load_generated_imgn_labels,
    predict_class_indices,
)
from yuv_reproduction.s08_rae import (
    cal_rgb_residual_sup,
    yuv_to_rgb_uint8,
    generate_rae,
    generate_rae_sup,
    load_recovery_npz_sup,
    load_recovery_png_sup,
    recover_rgb_cmp,
    recover_png_sup,
    recover_rgb_exact_sup,
    recover_yuv_sup,
    save_recovery_npz_sup,
    save_recovery_png_sup,
)
from yuv_reproduction.s02_rgb_yuv import rgb_to_yuv
from yuv_reproduction.s04_y_attack import ymifgsm_attack


def parse_arguments() -> argparse.Namespace:  # 读取生成或独立恢复参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recover-from",
        type=Path,
        help="从自包含RAE PNG独立恢复原图",
    )
    parser.add_argument(
        "--recovered-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sup" / "step08" / "recovered_from_file.png",
        help="恢复图像的保存位置",
    )
    parser.add_argument(
        "--verify-original",
        type=Path,
        help="可选原图，用于验证是否逐像素一致",
    )
    return parser.parse_args()


def recover_saved_rae_sup(  # 仅凭自包含PNG恢复原图
    input_path: Path,
    output_path: Path,
    original_path: Path | None = None,
) -> dict[str, object]:
    recovered_rgb, recovered_perturbation = recover_png_sup(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(recovered_rgb, mode="RGB").save(output_path)
    result: dict[str, object] = {
        "step": "8_independent_recovery",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "image_shape": list(recovered_rgb.shape),
        "recovered_perturbation_nonzero_count": int(
            np.count_nonzero(recovered_perturbation)
        ),
        "verification_original": None,
        "exact_rgb_recovery": None,
        "maximum_channel_error": None,
    }
    if original_path is not None:
        with Image.open(original_path) as image:
            original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if original_rgb.shape != recovered_rgb.shape:
            raise ValueError("验证原图与恢复图尺寸不一致")
        error = np.abs(
            recovered_rgb.astype(np.int16) - original_rgb.astype(np.int16)
        )
        result["verification_original"] = str(original_path.resolve())
        result["exact_rgb_recovery"] = bool(
            np.array_equal(recovered_rgb, original_rgb)
        )
        result["maximum_channel_error"] = int(error.max())
    return result


def main() -> None:  # 用一张真实图片验证两条恢复路径
    arguments = parse_arguments()
    if arguments.recover_from is not None:
        result = recover_saved_rae_sup(
            arguments.recover_from,
            arguments.recovered_output,
            arguments.verify_original,
        )
        result_path = PROJECT_ROOT / "results" / "step08_recovery.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"恢复结果已保存：{result_path}")
        return

    start_time = time.perf_counter()
    torch.manual_seed(2022)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = 2 / 255
    threshold = 0.5
    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    labels = load_generated_imgn_labels(PROJECT_ROOT / "results" / "step03_labels.csv")
    label = labels[image_path.name]
    with Image.open(image_path) as image:
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        input_batch = add_batch_dimension(convert_image_to_unit_tensor(image)).to(device)
    labels = torch.tensor([label], device=device)

    cam_model = load_imgn_model("resnet50", device)
    cam_map, target_categories = generate_gradcam_plus_plus_map(
        cam_model,
        cam_model.layer4[-1],
        input_batch,
        interpolation_method="opencv",
        gradient_method="input",
    )
    binary_mask = cam_mask(cam_map, threshold)
    del cam_model
    torch.cuda.empty_cache()

    original_yuv = rgb_to_yuv(original_rgb)
    mask_array = binary_mask[0, 0].cpu().numpy().astype(np.int32)
    white_box_models = [
        load_imgn_model(model_name, device)
        for model_name in ENSEMBLE_MODEL_NAMES
    ]
    ensemble_model = EqualWeightLogitsEnsemble(white_box_models).eval()
    evaluation_model = white_box_models[1]
    current_input = input_batch
    rae_prediction = label
    for attack_iteration in range(1, 51):
        adversarial_rgb, _ = ymifgsm_attack(
            ensemble_model,
            current_input,
            labels,
            epsilon,
        )
        adversarial_rgb_255 = (
            adversarial_rgb[0].detach().cpu().permute(1, 2, 0).numpy() * 255
        )
        adversarial_yuv = rgb_to_yuv(adversarial_rgb_255)
        y_perturbation = (
            adversarial_yuv[:, :, 0].astype(np.int32)
            - original_yuv[:, :, 0].astype(np.int32)
        )
        masked_y_perturbation = y_perturbation * mask_array
        reversible_yuv, embedding_metadata = generate_rae(
            original_yuv,
            masked_y_perturbation,
        )
        paper_rae_rgb = yuv_to_rgb_uint8(reversible_yuv)
        rae_tensor = add_batch_dimension(
            convert_image_to_unit_tensor(Image.fromarray(paper_rae_rgb))
        )
        rae_prediction = int(
            predict_class_indices(evaluation_model, rae_tensor.to(device)).item()
        )
        if rae_prediction != label:
            break

        attacked_yuv = original_yuv.copy()
        attacked_yuv[:, :, 0] = (
            original_yuv[:, :, 0].astype(np.int32) + masked_y_perturbation
        )
        attacked_rgb = yuv_to_rgb_uint8(attacked_yuv)
        current_input = add_batch_dimension(
            convert_image_to_unit_tensor(Image.fromarray(attacked_rgb))
        ).to(device)
    del ensemble_model, white_box_models, evaluation_model
    torch.cuda.empty_cache()

    reproduction_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step08"
    cmp_directory = PROJECT_ROOT / "outputs" / "cmp" / "step08"
    sup_directory = PROJECT_ROOT / "outputs" / "sup" / "step08"
    reproduction_directory.mkdir(parents=True, exist_ok=True)
    cmp_directory.mkdir(parents=True, exist_ok=True)
    sup_directory.mkdir(parents=True, exist_ok=True)
    paper_rae_path = reproduction_directory / "rae.png"
    Image.fromarray(paper_rae_rgb, mode="RGB").save(paper_rae_path)

    residual = cal_rgb_residual_sup(original_rgb, original_yuv)
    sup_png_path = sup_directory / "recovery_rae.png"
    save_recovery_png_sup(
        sup_png_path,
        paper_rae_rgb,
        reversible_yuv,
        int(embedding_metadata["threshold"]),
        residual,
        embedding_metadata["arithmetic_metadata"],
    )
    sup_visible_rgb, _, _, _ = load_recovery_png_sup(
        sup_png_path
    )
    sup_recovered_rgb, sup_recovered_perturbation = (
        recover_png_sup(sup_png_path)
    )
    Image.fromarray(sup_recovered_rgb, mode="RGB").save(
        sup_directory / "recovered_rgb.png"
    )

    exact_reversible_yuv, exact_embedding_metadata = generate_rae_sup(
        original_yuv,
        masked_y_perturbation,
    )
    exact_container_path = sup_directory / "recovery.npz"
    save_recovery_npz_sup(
        exact_container_path,
        exact_reversible_yuv,
        int(exact_embedding_metadata["threshold"]),
        residual,
    )
    loaded_yuv, loaded_threshold, loaded_residual = load_recovery_npz_sup(
        exact_container_path
    )
    recovered_yuv, recovered_perturbation = recover_yuv_sup(
        loaded_yuv,
        loaded_threshold,
    )
    exact_recovered_rgb = recover_rgb_exact_sup(
        recovered_yuv,
        loaded_residual,
    )
    Image.fromarray(exact_recovered_rgb, mode="RGB").save(
        sup_directory / "recovered_rgb_npz.png"
    )

    paper_recovery_succeeded = False
    paper_recovery_error = ""
    paper_exact_rgb_recovery = False
    paper_maximum_channel_error = None
    try:
        paper_recovered_rgb, _ = recover_rgb_cmp(
        paper_rae_rgb,
        int(embedding_metadata["threshold"]),
        embedding_metadata["arithmetic_metadata"],
        )
        paper_recovery_succeeded = True
        paper_exact_rgb_recovery = bool(np.array_equal(paper_recovered_rgb, original_rgb))
        paper_maximum_channel_error = int(
            np.abs(
                paper_recovered_rgb.astype(np.int16) - original_rgb.astype(np.int16)
            ).max()
        )
        Image.fromarray(paper_recovered_rgb, mode="RGB").save(
            cmp_directory / "recovered_rgb.png"
        )
    except ValueError as error:
        paper_recovery_error = str(error)

    reconstructed_paper_yuv = rgb_to_yuv(paper_rae_rgb).astype(np.int32)
    paper_round_trip_difference = (
        reconstructed_paper_yuv - reversible_yuv.astype(np.int32)
    )

    exact_error = np.abs(
        exact_recovered_rgb.astype(np.int16) - original_rgb.astype(np.int16)
    )
    result = {
        "step": "8_rae_generation_and_recovery",
        "device": str(device),
        "image": image_path.name,
        "label": label,
        "attack": "YMI-FGSM",
        "attack_iterations": attack_iteration,
        "epsilon": epsilon,
        "cam_target_category": int(target_categories.item()),
        "cam_mask_pixel_count": int(binary_mask.sum().item()),
        "masked_y_nonzero_count": int(np.count_nonzero(masked_y_perturbation)),
        "encoded_recovery_correction": "original_y_minus_adversarial_y",
        "author_embedding": {
            key: value
            for key, value in embedding_metadata.items()
            if key != "arithmetic_metadata"
        },
        "sup_exact_container_embedding": exact_embedding_metadata,
        "paper_rgb_path": {
            "rae_prediction": rae_prediction,
            "attack_success": rae_prediction != label,
            "recovery_process_succeeded": paper_recovery_succeeded,
            "exact_rgb_recovery": paper_exact_rgb_recovery,
            "maximum_channel_error": paper_maximum_channel_error,
            "error": paper_recovery_error,
            "changed_yuv_value_count": [
                int(np.count_nonzero(paper_round_trip_difference[:, :, channel]))
                for channel in range(3)
            ],
            "maximum_yuv_value_error": [
                int(np.abs(paper_round_trip_difference[:, :, channel]).max())
                for channel in range(3)
            ],
        },
        "sup_png_path": {
            "container_size_bytes": sup_png_path.stat().st_size,
            "visible_rgb_matches_reproduction": bool(
                np.array_equal(sup_visible_rgb, paper_rae_rgb)
            ),
            "perturbation_exactly_recovered": bool(
                np.array_equal(
                    sup_recovered_perturbation,
                    masked_y_perturbation,
                )
            ),
            "exact_rgb_recovery": bool(
                np.array_equal(sup_recovered_rgb, original_rgb)
            ),
            "maximum_channel_error": int(
                np.abs(
                    sup_recovered_rgb.astype(np.int16)
                    - original_rgb.astype(np.int16)
                ).max()
            ),
        },
        "exact_container_path": {
            "container_size_bytes": exact_container_path.stat().st_size,
            "yuv_exactly_recovered": bool(
                np.array_equal(recovered_yuv, original_yuv.astype(np.int32))
            ),
            "perturbation_exactly_recovered": bool(
                np.array_equal(recovered_perturbation, masked_y_perturbation)
            ),
            "exact_rgb_recovery": bool(np.array_equal(exact_recovered_rgb, original_rgb)),
            "maximum_channel_error": int(exact_error.max()),
        },
        "runtime_seconds": time.perf_counter() - start_time,
    }
    result_path = PROJECT_ROOT / "results" / "step08.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
