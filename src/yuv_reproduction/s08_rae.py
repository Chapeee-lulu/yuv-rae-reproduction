# RAE生成与恢复闭环

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from .s0701_arithmetic import (
    arithmetic_decode_sup,
    arithmetic_encode_sup,
    arithmetic_encode,
    build_arithmetic_payload_sup,
    parse_arithmetic_payload_sup,
    restore_integer_perturbation_shape,
    serialize_integer_perturbation_sequence,
)
from .s0702_pee import (
    embed_payload_uv,
    embed_payload_uv_rounds,
    extract_payload_uv,
    extract_payload_uv_rounds_sup,
)
from .s02_rgb_yuv import (
    convert_color_channels_to_uint8,
    rgb_to_yuv,
    yuv_to_rgb,
)


RECOVERY_PNG_METADATA_KEY = "yuv_reproduction_exact_state_v1"
RECOVERY_ROUNDS_PNG_METADATA_KEY = "yuv_reproduction_exact_rounds_state_v1"


def generate_rae(  # 使用作者短载荷生成YUV载体
    original_yuv: np.ndarray,
    masked_y_perturbation: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    yuv = np.asarray(original_yuv)
    perturbation = np.asarray(masked_y_perturbation)
    if yuv.ndim != 3 or yuv.shape[2] != 3:
        raise ValueError("YUV图像必须有3个通道")
    if perturbation.shape != yuv.shape[:2]:
        raise ValueError("Y扰动形状与图像不一致")
    if not np.array_equal(yuv, np.rint(yuv)):
        raise ValueError("YUV图像必须为整数")

    recovery_correction = -perturbation.astype(np.int32)
    sequence = serialize_integer_perturbation_sequence(recovery_correction)
    payload_bits, arithmetic_metadata = (
        arithmetic_encode(sequence)
    )
    stego_u, stego_v, pee_metadata = embed_payload_uv(
        yuv[:, :, 1],
        yuv[:, :, 2],
        payload_bits,
        enforce_uint8_bounds=False,
    )
    reversible_yuv = np.empty(yuv.shape, dtype=np.int32)
    reversible_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + perturbation.astype(
        np.int32
    )
    reversible_yuv[:, :, 1] = stego_u
    reversible_yuv[:, :, 2] = stego_v
    metadata: dict[str, object] = {
        "payload_format": "author_code_only",
        "threshold": pee_metadata["threshold"],
        "capacity": pee_metadata["capacity"],
        "sequence_length": len(sequence),
        "code_length": len(payload_bits),
        "payload_length": len(payload_bits),
        "arithmetic_metadata": arithmetic_metadata,
    }
    return reversible_yuv, metadata


def yuv_to_rgb_uint8(yuv_image: np.ndarray) -> np.ndarray:  # 生成论文RGB路径
    rgb_float = yuv_to_rgb(np.asarray(yuv_image, dtype=np.float32))
    return convert_color_channels_to_uint8(rgb_float)


def recover_rgb_cmp(  # 从论文RGB路径尝试恢复
    reversible_rgb: np.ndarray,
    threshold: int,
    arithmetic_metadata: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed_yuv = rgb_to_yuv(reversible_rgb)
    try:
        if arithmetic_metadata is None:
            recovered_yuv, recovered_perturbation = recover_yuv_sup(
                reconstructed_yuv,
                threshold,
            )
        else:
            recovered_yuv, recovered_perturbation = (
                recover_yuv_payload_sup(
                    reconstructed_yuv,
                    threshold,
                    arithmetic_metadata,
                )
            )
    except (ValueError, ArithmeticError) as error:
        raise ValueError(
            "RGB/YUV整数往返已破坏嵌入载荷"
        ) from error
    recovered_rgb = yuv_to_rgb_uint8(recovered_yuv)
    return recovered_rgb, recovered_perturbation
def encode_y_payload_sup(  # 压缩整数Y扰动
    y_perturbation: np.ndarray,
) -> tuple[list[int], dict[str, object]]:
    sequence = serialize_integer_perturbation_sequence(y_perturbation)
    code, metadata = arithmetic_encode_sup(sequence)
    payload = build_arithmetic_payload_sup(code, metadata)
    return payload, {
        "sequence_length": len(sequence),
        "code_length": len(code),
        "payload_length": len(payload),
        "frequencies": metadata["frequencies"],
    }


def decode_y_payload_sup(  # 解码并恢复Y扰动形状
    payload_bits: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    code, metadata = parse_arithmetic_payload_sup(payload_bits)
    sequence = arithmetic_decode_sup(code, metadata)
    return restore_integer_perturbation_shape(sequence, shape)


def generate_rae_sup(  # 按Algorithm 2生成YUV载体
    original_yuv: np.ndarray,
    masked_y_perturbation: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    yuv = np.asarray(original_yuv)
    perturbation = np.asarray(masked_y_perturbation)
    if yuv.ndim != 3 or yuv.shape[2] != 3:
        raise ValueError("YUV图像必须有3个通道")
    if perturbation.shape != yuv.shape[:2]:
        raise ValueError("Y扰动形状与图像不一致")
    if not np.array_equal(yuv, np.rint(yuv)):
        raise ValueError("YUV图像必须为整数")

    recovery_correction = -perturbation.astype(np.int32)
    payload_bits, coding_metadata = encode_y_payload_sup(recovery_correction)
    stego_u, stego_v, pee_metadata = embed_payload_uv(
        yuv[:, :, 1],
        yuv[:, :, 2],
        payload_bits,
    )
    reversible_yuv = np.empty(yuv.shape, dtype=np.int32)
    reversible_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + perturbation.astype(np.int32)
    reversible_yuv[:, :, 1] = stego_u
    reversible_yuv[:, :, 2] = stego_v
    metadata: dict[str, object] = {
        "threshold": pee_metadata["threshold"],
        "capacity": pee_metadata["capacity"],
        **coding_metadata,
    }
    return reversible_yuv, metadata


def generate_rae_rounds_sup(  # 使用多轮PEE生成大载荷RAE
    original_yuv: np.ndarray,
    masked_y_perturbation: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    yuv = np.asarray(original_yuv)
    perturbation = np.asarray(masked_y_perturbation)
    if yuv.ndim != 3 or yuv.shape[2] != 3:
        raise ValueError("YUV图像必须有3个通道")
    if perturbation.shape != yuv.shape[:2]:
        raise ValueError("Y扰动形状与图像不一致")
    if not np.array_equal(yuv, np.rint(yuv)):
        raise ValueError("YUV图像必须为整数")

    recovery_correction = -perturbation.astype(np.int32)
    payload_bits, coding_metadata = encode_y_payload_sup(recovery_correction)
    stego_u, stego_v, pee_metadata = embed_payload_uv_rounds(
        yuv[:, :, 1],
        yuv[:, :, 2],
        payload_bits,
    )
    reversible_yuv = np.empty(yuv.shape, dtype=np.int32)
    reversible_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + perturbation.astype(
        np.int32
    )
    reversible_yuv[:, :, 1] = stego_u
    reversible_yuv[:, :, 2] = stego_v
    return reversible_yuv, {
        **coding_metadata,
        **pee_metadata,
    }


def recover_yuv_sup(  # 提取扰动并恢复原YUV
    reversible_yuv: np.ndarray,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    yuv = np.asarray(reversible_yuv)
    if yuv.ndim != 3 or yuv.shape[2] != 3:
        raise ValueError("Reversible YUV图像必须有3个通道")
    payload_bits, recovered_u, recovered_v, _ = (
        extract_payload_uv(
            yuv[:, :, 1],
            yuv[:, :, 2],
            threshold,
        )
    )
    recovered_correction = decode_y_payload_sup(
        payload_bits,
        yuv.shape[:2],
    )
    recovered_perturbation = -recovered_correction
    recovered_yuv = np.empty(yuv.shape, dtype=np.int32)
    recovered_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + recovered_correction
    recovered_yuv[:, :, 1] = recovered_u
    recovered_yuv[:, :, 2] = recovered_v
    return recovered_yuv, recovered_perturbation


def recover_yuv_rounds_sup(  # 逆序提取多轮PEE并恢复原YUV
    reversible_yuv: np.ndarray,
    thresholds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    yuv = np.asarray(reversible_yuv)
    if yuv.ndim != 3 or yuv.shape[2] != 3:
        raise ValueError("Reversible YUV图像必须有3个通道")
    payload_bits, recovered_u, recovered_v = extract_payload_uv_rounds_sup(
        yuv[:, :, 1],
        yuv[:, :, 2],
        thresholds,
    )
    recovered_correction = decode_y_payload_sup(payload_bits, yuv.shape[:2])
    recovered_perturbation = -recovered_correction
    recovered_yuv = np.empty(yuv.shape, dtype=np.int32)
    recovered_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + recovered_correction
    recovered_yuv[:, :, 1] = recovered_u
    recovered_yuv[:, :, 2] = recovered_v
    return recovered_yuv, recovered_perturbation


def recover_yuv_payload_sup(  # 解码作者短载荷
    reversible_yuv: np.ndarray,
    threshold: int,
    arithmetic_metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    yuv = np.asarray(reversible_yuv)
    payload_bits, recovered_u, recovered_v, _ = (
        extract_payload_uv(
            yuv[:, :, 1],
            yuv[:, :, 2],
            threshold,
        )
    )
    correction_sequence = arithmetic_decode_sup(
        payload_bits,
        arithmetic_metadata,
    )
    recovered_correction = restore_integer_perturbation_shape(
        correction_sequence,
        yuv.shape[:2],
    )
    recovered_yuv = np.empty(yuv.shape, dtype=np.int32)
    recovered_yuv[:, :, 0] = yuv[:, :, 0].astype(np.int32) + recovered_correction
    recovered_yuv[:, :, 1] = recovered_u
    recovered_yuv[:, :, 2] = recovered_v
    return recovered_yuv, -recovered_correction


def cal_rgb_residual_sup(  # 保存颜色转换残差
    original_rgb: np.ndarray,
    original_yuv: np.ndarray,
) -> np.ndarray:
    converted_rgb = yuv_to_rgb_uint8(original_yuv)
    return original_rgb.astype(np.int16) - converted_rgb.astype(np.int16)


def recover_rgb_exact_sup(  # 使用残差精确恢复RGB
    recovered_yuv: np.ndarray,
    rgb_conversion_residual: np.ndarray,
) -> np.ndarray:
    converted_rgb = yuv_to_rgb_uint8(recovered_yuv).astype(np.int16)
    recovered_rgb = converted_rgb + np.asarray(rgb_conversion_residual, dtype=np.int16)
    if recovered_rgb.min() < 0 or recovered_rgb.max() > 255:
        raise ValueError("恢复的RGB值超出0至255")
    return recovered_rgb.astype(np.uint8)


def save_recovery_npz_sup(  # 保存精确恢复所需内容
    path: Path,
    reversible_yuv: np.ndarray,
    threshold: int,
    rgb_conversion_residual: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        reversible_yuv=np.asarray(reversible_yuv, dtype=np.int16),
        threshold=np.asarray(threshold, dtype=np.int16),
        rgb_conversion_residual=np.asarray(rgb_conversion_residual, dtype=np.int8),
    )


def load_recovery_npz_sup(  # 读取精确恢复容器
    path: Path,
) -> tuple[np.ndarray, int, np.ndarray]:
    with np.load(path) as package:
        reversible_yuv = package["reversible_yuv"].astype(np.int32)
        threshold = int(package["threshold"])
        rgb_conversion_residual = package["rgb_conversion_residual"].astype(np.int16)
    return reversible_yuv, threshold, rgb_conversion_residual


def save_recovery_png_sup(  # 保存像素不变的自包含恢复PNG
    path: Path,
    reversible_rgb: np.ndarray,
    reversible_yuv: np.ndarray,
    threshold: int,
    rgb_conversion_residual: np.ndarray,
    arithmetic_metadata: dict[str, object] | None = None,
) -> None:
    rgb = np.asarray(reversible_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("可逆RGB图像必须是8位3通道图像")
    archive = BytesIO()
    state: dict[str, np.ndarray] = {
        "reversible_yuv": np.asarray(reversible_yuv, dtype=np.int16),
        "threshold": np.asarray(threshold, dtype=np.int16),
        "rgb_conversion_residual": np.asarray(
            rgb_conversion_residual,
            dtype=np.int8,
        ),
    }
    if arithmetic_metadata is not None:
        encoded_metadata = json.dumps(
            arithmetic_metadata,
            separators=(",", ":"),
        ).encode("utf-8")
        state["arithmetic_metadata"] = np.frombuffer(
            encoded_metadata,
            dtype=np.uint8,
        )
    np.savez_compressed(
        archive,
        **state,
    )
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt(
        RECOVERY_PNG_METADATA_KEY,
        base64.b64encode(archive.getvalue()).decode("ascii"),
        zip=True,
    )
    Image.fromarray(rgb, mode="RGB").save(path, pnginfo=metadata)


def load_recovery_png_sup(  # 读取自包含恢复PNG
    path: Path,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    with Image.open(path) as image:
        image.load()
        reversible_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        encoded_state = image.info.get(RECOVERY_PNG_METADATA_KEY)
    if not isinstance(encoded_state, str):
        raise ValueError("缺少补充恢复元数据")
    try:
        archive_bytes = base64.b64decode(encoded_state, validate=True)
        with np.load(BytesIO(archive_bytes)) as package:
            reversible_yuv = package["reversible_yuv"].astype(np.int32)
            threshold = int(package["threshold"])
            rgb_conversion_residual = package["rgb_conversion_residual"].astype(np.int16)
    except (ValueError, KeyError) as error:
        raise ValueError("补充恢复元数据无效") from error
    return reversible_rgb, reversible_yuv, threshold, rgb_conversion_residual


def _load_arithmetic_metadata_sup(  # 读取作者短载荷解码信息
    path: Path,
) -> dict[str, object] | None:
    with Image.open(path) as image:
        image.load()
        encoded_state = image.info.get(RECOVERY_PNG_METADATA_KEY)
    if not isinstance(encoded_state, str):
        raise ValueError("缺少补充恢复元数据")
    archive_bytes = base64.b64decode(encoded_state, validate=True)
    with np.load(BytesIO(archive_bytes)) as package:
        if "arithmetic_metadata" not in package:
            return None
        metadata_bytes = package["arithmetic_metadata"].astype(np.uint8).tobytes()
    return json.loads(metadata_bytes.decode("utf-8"))


def recover_png_sup(  # 使用补充状态恢复原始RGB
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    _, reversible_yuv, threshold, residual = load_recovery_png_sup(path)
    arithmetic_metadata = _load_arithmetic_metadata_sup(path)
    if arithmetic_metadata is None:
        recovered_yuv, recovered_perturbation = recover_yuv_sup(
            reversible_yuv,
            threshold,
        )
    else:
        recovered_yuv, recovered_perturbation = (
            recover_yuv_payload_sup(
                reversible_yuv,
                threshold,
                arithmetic_metadata,
            )
        )
    recovered_rgb = recover_rgb_exact_sup(recovered_yuv, residual)
    return recovered_rgb, recovered_perturbation


def save_recovery_png_rounds_sup(  # 保存多轮PEE自包含PNG
    path: Path,
    reversible_rgb: np.ndarray,
    reversible_yuv: np.ndarray,
    thresholds: list[int],
    rgb_conversion_residual: np.ndarray,
) -> None:
    rgb = np.asarray(reversible_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("可逆RGB图像必须是8位3通道图像")
    if not thresholds:
        raise ValueError("多轮PEE阈值不能为空")
    archive = BytesIO()
    np.savez_compressed(
        archive,
        reversible_yuv=np.asarray(reversible_yuv, dtype=np.int16),
        thresholds=np.asarray(thresholds, dtype=np.int16),
        rgb_conversion_residual=np.asarray(
            rgb_conversion_residual,
            dtype=np.int8,
        ),
    )
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt(
        RECOVERY_ROUNDS_PNG_METADATA_KEY,
        base64.b64encode(archive.getvalue()).decode("ascii"),
        zip=True,
    )
    Image.fromarray(rgb, mode="RGB").save(path, pnginfo=metadata)


def load_recovery_png_rounds_sup(  # 读取多轮PEE自包含PNG
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    with Image.open(path) as image:
        image.load()
        reversible_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        encoded_state = image.info.get(RECOVERY_ROUNDS_PNG_METADATA_KEY)
    if not isinstance(encoded_state, str):
        raise ValueError("缺少多轮PEE恢复元数据")
    try:
        archive_bytes = base64.b64decode(encoded_state, validate=True)
        with np.load(BytesIO(archive_bytes)) as package:
            reversible_yuv = package["reversible_yuv"].astype(np.int32)
            thresholds = [int(value) for value in package["thresholds"].tolist()]
            residual = package["rgb_conversion_residual"].astype(np.int16)
    except (ValueError, KeyError) as error:
        raise ValueError("多轮PEE恢复元数据无效") from error
    return reversible_rgb, reversible_yuv, thresholds, residual


def recover_png_rounds_sup(  # 从多轮PEE自包含PNG恢复原始RGB
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    _, reversible_yuv, thresholds, residual = load_recovery_png_rounds_sup(path)
    recovered_yuv, recovered_perturbation = recover_yuv_rounds_sup(
        reversible_yuv,
        thresholds,
    )
    recovered_rgb = recover_rgb_exact_sup(recovered_yuv, residual)
    return recovered_rgb, recovered_perturbation

