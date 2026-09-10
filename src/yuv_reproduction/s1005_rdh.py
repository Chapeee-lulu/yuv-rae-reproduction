# RDH基线的超像素与整幅RGB嵌入

import numpy as np

from .s0701_arithmetic import arithmetic_decode_sup, arithmetic_encode_sup
from .s0702_pee import embed_bits_pee, extract_bits_pee
from .s09_metrics import (
    cal_mean_ciede2000,
    cal_psnr,
    cal_recovery_metrics,
    cal_ssim,
)


PEE_SAFE_MARGIN = 12


def select_rdh_superpixel_shape(  # 选择论文未公开的超像素参数
    attack_name: str,
    epsilon_pixels: int,
) -> tuple[int, int]:
    if attack_name == "FGSM":
        return 1, 1
    if epsilon_pixels == 2:
        return 1, 2
    if epsilon_pixels == 4:
        return 1, 3
    raise ValueError("RDH仅支持2/255或4/255")


def smooth_rgb_perturbation_by_superpixels(  # 对块内RGB扰动取整平均
    original_rgb: np.ndarray,
    adversarial_rgb_255: np.ndarray,
    block_shape: tuple[int, int],
) -> np.ndarray:
    original = np.asarray(original_rgb, dtype=np.uint8)
    adversarial = np.asarray(adversarial_rgb_255, dtype=np.float32)
    if original.shape != adversarial.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("原图和对抗图必须是同形状RGB图像")
    block_height, block_width = block_shape
    if block_height <= 0 or block_width <= 0:
        raise ValueError("超像素尺寸必须为正数")

    height, width, _ = original.shape
    row_count = (height + block_height - 1) // block_height
    column_count = (width + block_width - 1) // block_width
    block_deltas = np.zeros((row_count, column_count, 3), dtype=np.int32)
    perturbation = adversarial - original.astype(np.float32)
    smoothed = np.zeros_like(perturbation, dtype=np.int32)
    for row in range(0, height, block_height):
        for column in range(0, width, block_width):
            block = perturbation[
                row : row + block_height,
                column : column + block_width,
            ]
            block_mean = np.rint(block.mean(axis=(0, 1))).astype(np.int32)
            block_deltas[row // block_height, column // block_width] = block_mean
            smoothed[
                row : row + block_height,
                column : column + block_width,
            ] = block_mean
    return np.clip(original.astype(np.int32) + smoothed, 0, 255).astype(np.uint8)


def _cal_superpixel_payload(
    original: np.ndarray,
    adversarial: np.ndarray,
    block_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_height, block_width = block_shape
    height, width, _ = original.shape
    row_count = (height + block_height - 1) // block_height
    column_count = (width + block_width - 1) // block_width
    block_deltas = np.zeros((row_count, column_count, 3), dtype=np.int32)
    expanded = np.zeros_like(original, dtype=np.int32)
    perturbation = adversarial - original.astype(np.float32)
    for block_row, row in enumerate(range(0, height, block_height)):
        for block_column, column in enumerate(range(0, width, block_width)):
            block = perturbation[
                row : row + block_height,
                column : column + block_width,
            ]
            delta = np.rint(block.mean(axis=(0, 1))).astype(np.int32)
            block_deltas[block_row, block_column] = delta
            expanded[
                row : row + block_height,
                column : column + block_width,
            ] = delta
    attacked = np.clip(original.astype(np.int32) + expanded, 0, 255).astype(np.uint8)
    clipping_residual = original.astype(np.int32) - (
        attacked.astype(np.int32) - expanded
    )
    return attacked, block_deltas, clipping_residual


def _expand_superpixel_deltas(
    block_deltas: np.ndarray,
    image_shape: tuple[int, int, int],
    block_shape: tuple[int, int],
) -> np.ndarray:
    block_height, block_width = block_shape
    expanded = np.repeat(
        np.repeat(block_deltas, block_height, axis=0),
        block_width,
        axis=1,
    )
    return expanded[: image_shape[0], : image_shape[1], : image_shape[2]]


def _stack_rgb_channels(rgb: np.ndarray) -> np.ndarray:
    return np.vstack([rgb[:, :, channel] for channel in range(3)]).astype(np.int32)


def _unstack_rgb_channels(stacked: np.ndarray, height: int) -> np.ndarray:
    return np.stack(
        [stacked[channel * height : (channel + 1) * height] for channel in range(3)],
        axis=2,
    ).astype(np.uint8)


def prepare_rdh_candidate_pro(  # 生成超像素RDH候选
    original_rgb: np.ndarray,
    adversarial_rgb_255: np.ndarray,
    _binary_mask: np.ndarray,
    block_shape: tuple[int, int],
) -> dict[str, object]:
    original = np.asarray(original_rgb, dtype=np.uint8)
    adversarial = np.asarray(adversarial_rgb_255, dtype=np.float32)
    last_error = ""
    for extra_width in range(8):
        current_shape = (block_shape[0], block_shape[1] + extra_width)
        attacked, block_deltas, clipping_residual = _cal_superpixel_payload(
            original,
            adversarial,
            current_shape,
        )
        safe_attacked = np.clip(
            attacked.astype(np.int32),
            PEE_SAFE_MARGIN,
            255 - PEE_SAFE_MARGIN,
        ).astype(np.uint8)
        carrier_residual = attacked.astype(np.int32) - safe_attacked.astype(np.int32)
        streams = (
            block_deltas.reshape(-1),
            clipping_residual.reshape(-1),
            carrier_residual.reshape(-1),
        )
        encoded_streams = [arithmetic_encode_sup(stream.tolist()) for stream in streams]
        code = [bit for stream_code, _ in encoded_streams for bit in stream_code]
        arithmetic_metadata = [metadata for _, metadata in encoded_streams]
        code_lengths = [len(stream_code) for stream_code, _ in encoded_streams]
        try:
            stego, pee_metadata = embed_bits_pee(
                _stack_rgb_channels(safe_attacked),
                code,
            )
        except ValueError as error:
            last_error = str(error)
            continue
        return {
            "succeeded": True,
            "rae_rgb": _unstack_rgb_channels(stego, original.shape[0]),
            "attacked_rgb": attacked,
            "stego_rgb_stack": stego,
            "image_shape": original.shape,
            "block_delta_shape": block_deltas.shape,
            "arithmetic_metadata": arithmetic_metadata,
            "code_lengths": code_lengths,
            "pee_metadata": pee_metadata,
            "requested_block_shape": block_shape,
            "block_shape": current_shape,
        }
    return {"succeeded": False, "error": last_error}


def finalize_rdh_candidate_pro(  # 提取扰动并恢复RDH原图
    original_rgb: np.ndarray,
    candidate: dict[str, object],
    label: int,
    rae_prediction: int,
    outer_iteration: int,
) -> dict[str, object]:
    stego = np.asarray(candidate["stego_rgb_stack"], dtype=np.int32)
    pee_metadata = dict(candidate["pee_metadata"])
    code, recovered_stack, _ = extract_bits_pee(stego, int(pee_metadata["threshold"]))
    arithmetic_metadata = [dict(value) for value in candidate["arithmetic_metadata"]]
    code_lengths = [int(value) for value in candidate["code_lengths"]]
    decoded_streams = []
    code_start = 0
    for code_length, metadata in zip(code_lengths, arithmetic_metadata):
        decoded_streams.append(
            arithmetic_decode_sup(code[code_start : code_start + code_length], metadata)
        )
        code_start += code_length
    image_shape = tuple(int(value) for value in candidate["image_shape"])
    block_delta_shape = tuple(int(value) for value in candidate["block_delta_shape"])
    block_deltas = np.asarray(
        decoded_streams[0],
        dtype=np.int32,
    ).reshape(block_delta_shape)
    clipping_residual = np.asarray(decoded_streams[1], dtype=np.int32).reshape(image_shape)
    carrier_residual = np.asarray(decoded_streams[2], dtype=np.int32).reshape(image_shape)
    safe_attacked = _unstack_rgb_channels(recovered_stack, image_shape[0])
    attacked = (safe_attacked.astype(np.int32) + carrier_residual).astype(np.uint8)
    expanded = _expand_superpixel_deltas(
        block_deltas,
        image_shape,
        tuple(candidate["block_shape"]),
    )
    recovered = (
        attacked.astype(np.int32) - expanded + clipping_residual
    ).astype(np.uint8)
    original = np.asarray(original_rgb, dtype=np.uint8)
    rae_rgb = np.asarray(candidate["rae_rgb"], dtype=np.uint8)
    recovery = cal_recovery_metrics(original, recovered)
    return {
        "succeeded": True,
        "rae_rgb": rae_rgb,
        "rae_prediction_inception": int(rae_prediction),
        "inception_attack_success": int(rae_prediction) != int(label),
        "outer_iterations": int(outer_iteration),
        "embedding_threshold": int(pee_metadata["threshold"]),
        "payload_length": int(pee_metadata["payload_length"]),
        "masked_nonzero_count": int(np.count_nonzero(block_deltas)),
        "perturbation_recovered": True,
        "exact_recovery": recovery,
        "paper_exact_recovery": recovery["exact_rgb_recovery"],
        "psnr": cal_psnr(original, rae_rgb),
        "ssim": cal_ssim(original, rae_rgb),
        "ciede2000": cal_mean_ciede2000(original, rae_rgb),
        "rdh_requested_superpixel_shape": list(candidate["requested_block_shape"]),
        "rdh_superpixel_shape": list(candidate["block_shape"]),
        "rdh_embedding_backend": "PEE_reference_substitute",
        "_artifact_kind": "rdh",
        "_recovered_rgb": recovered,
    }
