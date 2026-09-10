# 第7步：验证算术编码与PEE

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s0701_arithmetic import (
    arithmetic_decode_sup,
    arithmetic_encode_sup,
    build_arithmetic_payload_sup,
    parse_arithmetic_payload_sup,
    restore_integer_perturbation_shape,
    serialize_integer_perturbation_sequence,
)
from yuv_reproduction.s0702_pee import (
    embed_bits_pee,
    embed_payload_uv,
    extract_bits_pee,
    extract_payload_uv,
)
from yuv_reproduction.s02_rgb_yuv import rgb_to_yuv


def generate_sparse_integer_perturbation(shape: tuple[int, int]) -> np.ndarray:  # 生成人工稀疏扰动
    perturbation = np.zeros(shape, dtype=np.int32)
    center_row = shape[0] // 2
    center_column = shape[1] // 2
    for row in range(center_row - 8, center_row + 8):
        for column in range(center_column - 8, center_column + 8):
            perturbation[row, column] = 1 if (row + column) % 2 == 0 else -1
    return perturbation


def main() -> None:  # 用人工数组和真实U/V验证闭环
    artificial_carrier = np.full((32, 32), 128, dtype=np.int32)
    artificial_payload = [0, 1, 1, 0] * 16
    artificial_stego, artificial_metadata = embed_bits_pee(
        artificial_carrier,
        artificial_payload,
    )
    artificial_extracted, artificial_recovered, _ = extract_bits_pee(
        artificial_stego,
        artificial_metadata["threshold"],
    )

    image_path = sorted((PROJECT_ROOT / "data" / "original_img").glob("*.png"))[0]
    with Image.open(image_path) as image:
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    original_yuv = rgb_to_yuv(original_rgb)
    original_u = original_yuv[:, :, 1].astype(np.int32)
    original_v = original_yuv[:, :, 2].astype(np.int32)

    perturbation = generate_sparse_integer_perturbation(original_u.shape)
    perturbation_sequence = serialize_integer_perturbation_sequence(perturbation)
    arithmetic_code, arithmetic_metadata = arithmetic_encode_sup(
        perturbation_sequence
    )
    payload_bits = build_arithmetic_payload_sup(arithmetic_code, arithmetic_metadata)

    stego_u, stego_v, pee_metadata = embed_payload_uv(
        original_u,
        original_v,
        payload_bits,
    )
    extracted_payload, recovered_u, recovered_v, extraction_metadata = (
        extract_payload_uv(
            stego_u,
            stego_v,
            pee_metadata["threshold"],
        )
    )
    extracted_code, extracted_arithmetic_metadata = parse_arithmetic_payload_sup(
        extracted_payload
    )
    recovered_sequence = arithmetic_decode_sup(
        extracted_code,
        extracted_arithmetic_metadata,
    )
    recovered_perturbation = restore_integer_perturbation_shape(
        recovered_sequence,
        perturbation.shape,
    )

    output_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step07"
    output_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stego_u.astype(np.uint8), mode="L").save(
        output_directory / "stego_u.png"
    )
    Image.fromarray(stego_v.astype(np.uint8), mode="L").save(
        output_directory / "stego_v.png"
    )

    result = {
        "step": "7_s0701_arithmetic_and_pee",
        "image": image_path.name,
        "artificial_array": {
            "payload_length": len(artificial_payload),
            "threshold": artificial_metadata["threshold"],
            "bits_exactly_extracted": artificial_extracted == artificial_payload,
            "carrier_exactly_recovered": bool(
                np.array_equal(artificial_recovered, artificial_carrier)
            ),
        },
        "s0701_arithmetic": {
            "sequence_length": len(perturbation_sequence),
            "nonzero_symbols": int(np.count_nonzero(perturbation)),
            "frequencies": arithmetic_metadata["frequencies"],
            "code_length": len(arithmetic_code),
            "payload_length_with_frequency_table": len(payload_bits),
            "perturbation_exactly_decoded": bool(
                np.array_equal(recovered_perturbation, perturbation)
            ),
        },
        "uv_pee": {
            "threshold": pee_metadata["threshold"],
            "capacity": pee_metadata["capacity"],
            "payload_length": pee_metadata["payload_length"],
            "extracted_payload_length": extraction_metadata["payload_length"],
            "payload_exactly_extracted": extracted_payload == payload_bits,
            "u_exactly_recovered": bool(np.array_equal(recovered_u, original_u)),
            "v_exactly_recovered": bool(np.array_equal(recovered_v, original_v)),
            "stego_minimum": int(min(stego_u.min(), stego_v.min())),
            "stego_maximum": int(max(stego_u.max(), stego_v.max())),
        },
    }
    result_path = PROJECT_ROOT / "results" / "step07.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")


if __name__ == "__main__":
    main()
