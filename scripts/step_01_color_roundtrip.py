"""第1步：验证论文RGB/YUV公式的整数往返误差。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yuv_rae.pixel_comparison import compare_rgb_pixels
from yuv_rae.yuv_conversion import rgb_to_yuv, yuv_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取一张RGB图像，执行RGB→YUV→RGB，并统计逐像素误差。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入PNG/JPEG路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "step01_color_roundtrip",
        help="实验输出目录",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入图像：{input_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    float_yuv = rgb_to_yuv(original, quantize=False)
    float_recovered = yuv_to_rgb(float_yuv, quantize=False, clip=False)
    float_difference = np.abs(original.astype(np.float64) - float_recovered)

    yuv = rgb_to_yuv(original, quantize=True)
    recovered_float = yuv_to_rgb(yuv, quantize=True, clip=True)
    recovered = recovered_float.astype(np.uint8)
    pixel_comparison = compare_rgb_pixels(original, recovered)

    difference = np.abs(original.astype(np.int16) - recovered.astype(np.int16))
    amplified_difference = np.clip(difference * 80, 0, 255).astype(np.uint8)

    Image.fromarray(original).save(output_dir / "original.png")
    Image.fromarray(recovered).save(output_dir / "roundtrip.png")
    Image.fromarray(amplified_difference).save(output_dir / "difference_x80.png")

    roundtrip_results = {
        "input_file_name": input_path.name,
        "input_sha256": input_sha256,
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
        "float_roundtrip_max_absolute_error": float(float_difference.max()),
        "float_roundtrip_mean_absolute_error": float(float_difference.mean()),
        "integer_roundtrip_rgb_arrays_exactly_equal": (
            pixel_comparison.rgb_arrays_exactly_equal
        ),
        "integer_roundtrip_max_rgb_channel_absolute_error": (
            pixel_comparison.max_rgb_channel_absolute_error
        ),
        "integer_roundtrip_mean_rgb_channel_absolute_error": (
            pixel_comparison.mean_rgb_channel_absolute_error
        ),
        "integer_roundtrip_changed_rgb_channel_value_count": (
            pixel_comparison.changed_rgb_channel_value_count
        ),
        "integer_roundtrip_changed_rgb_pixel_count": (
            pixel_comparison.changed_rgb_pixel_count
        ),
        "total_rgb_pixel_count": pixel_comparison.total_rgb_pixel_count,
        "integer_roundtrip_changed_rgb_pixel_ratio": (
            pixel_comparison.changed_rgb_pixel_ratio
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(roundtrip_results, file, ensure_ascii=False, indent=2)

    print("=== RGB → YUV → RGB 整数往返实验 ===")
    print(f"输入图像：{input_path}")
    print(f"图像尺寸：{original.shape[1]} × {original.shape[0]}")
    print(f"浮点往返最大绝对误差：{float_difference.max():.9f}")
    print(f"浮点往返平均绝对误差：{float_difference.mean():.9f}")
    print(f"RGB数组逐像素完全一致：{pixel_comparison.rgb_arrays_exactly_equal}")
    print(f"RGB单通道最大绝对误差：{pixel_comparison.max_rgb_channel_absolute_error}")
    print(
        "RGB单通道平均绝对误差："
        f"{pixel_comparison.mean_rgb_channel_absolute_error:.6f}"
    )
    print(
        "变化的RGB通道值数量："
        f"{pixel_comparison.changed_rgb_channel_value_count}"
    )
    print(
        "变化的RGB像素数量："
        f"{pixel_comparison.changed_rgb_pixel_count}/"
        f"{pixel_comparison.total_rgb_pixel_count} "
        f"({pixel_comparison.changed_rgb_pixel_ratio:.2%})"
    )
    print(f"指标记录：{output_dir / 'metrics.json'}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
