# 第2步：检查RGB与YUV 4:4:4转换

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 定位目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s02_rgb_yuv import (
    clip_color_channel_values,
    convert_color_channels_to_uint8,
    rgb_to_yuv,
    yuv_to_rgb,
)


def cal_round_trip_error(  # 统计来回转换误差
    original_rgb: np.ndarray,
    recovered_rgb: np.ndarray,
) -> dict[str, object]:
    absolute_error = np.abs(
        original_rgb.astype(np.int16) - recovered_rgb.astype(np.int16)
    )
    return {
        "maximum_channel_error": int(absolute_error.max()),
        "mean_absolute_channel_error": float(absolute_error.mean()),
        "equal_channel_ratio": float(np.mean(absolute_error == 0)),
        "exact_rgb_recovery": bool(np.array_equal(original_rgb, recovered_rgb)),
    }


def summarize_yuv_channels(yuv_image: np.ndarray) -> dict[str, dict[str, float]]:  # 统计YUV通道
    channel_names = ("Y", "U", "V")
    return {
        channel_name: {
            "minimum": int(yuv_image[:, :, channel_index].min()),
            "maximum": int(yuv_image[:, :, channel_index].max()),
            "mean": float(yuv_image[:, :, channel_index].mean()),
        }
        for channel_index, channel_name in enumerate(channel_names)
    }


def check_one_image(image_path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:  # 检查单张图片
    with Image.open(image_path) as image:
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    yuv_image = rgb_to_yuv(original_rgb)
    recovered_rgb = yuv_to_rgb(yuv_image)
    report = {
        "filename": image_path.name,
        "shape": list(original_rgb.shape),
        "yuv_channel_statistics": summarize_yuv_channels(yuv_image),
        "round_trip_error": cal_round_trip_error(original_rgb, recovered_rgb),
    }
    return report, yuv_image, recovered_rgb


def main() -> None:  # 执行第2步检查
    image_directory = PROJECT_ROOT / "data" / "original_img"
    image_paths = sorted(image_directory.glob("*.png"))[:10]
    if len(image_paths) != 10:
        raise RuntimeError(f"应检查10张图片，实际找到{len(image_paths)}")

    # 检查五种人工颜色
    artificial_rgb = np.array(
        [[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )
    artificial_yuv = rgb_to_yuv(artificial_rgb)
    artificial_recovered_rgb = yuv_to_rgb(artificial_yuv)

    # 检查前十张真实图片
    image_reports = []
    first_yuv = None
    first_recovered_rgb = None
    for image_path in image_paths:
        image_report, yuv_image, recovered_rgb = check_one_image(image_path)
        image_reports.append(image_report)
        if first_yuv is None:
            first_yuv = yuv_image
            first_recovered_rgb = recovered_rgb

    maximum_errors = [
        report["round_trip_error"]["maximum_channel_error"]
        for report in image_reports
    ]
    mean_errors = [
        report["round_trip_error"]["mean_absolute_channel_error"]
        for report in image_reports
    ]
    result = {
        "step": "2_rgb_yuv_444_conversion",
        "artificial_colors": {
            "rgb": artificial_rgb.tolist(),
            "yuv": artificial_yuv.tolist(),
            "recovered_rgb": artificial_recovered_rgb.tolist(),
            "round_trip_error": cal_round_trip_error(
                artificial_rgb, artificial_recovered_rgb
            ),
        },
        "one_real_image": image_reports[0],
        "ten_real_images": {
            "image_count": len(image_reports),
            "maximum_channel_error": int(max(maximum_errors)),
            "mean_absolute_channel_error": float(np.mean(mean_errors)),
            "per_image": image_reports,
        },
    }

    # 保存首张图片的通道预览
    output_directory = PROJECT_ROOT / "outputs" / "reproduction" / "step02"
    output_directory.mkdir(parents=True, exist_ok=True)
    first_stem = image_paths[0].stem
    preview_yuv = convert_color_channels_to_uint8(clip_color_channel_values(first_yuv))
    preview_rgb = convert_color_channels_to_uint8(first_recovered_rgb)
    Image.fromarray(preview_yuv[:, :, 0], mode="L").save(
        output_directory / f"{first_stem}_y_channel.png"
    )
    Image.fromarray(preview_yuv[:, :, 1], mode="L").save(
        output_directory / f"{first_stem}_u_channel.png"
    )
    Image.fromarray(preview_yuv[:, :, 2], mode="L").save(
        output_directory / f"{first_stem}_v_channel.png"
    )
    Image.fromarray(preview_rgb, mode="RGB").save(
        output_directory / f"{first_stem}_recovered_rgb.png"
    )

    # 保存数值结果
    result_path = PROJECT_ROOT / "results" / "step02.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{result_path}")
    print(f"通道预览已保存：{output_directory}")


if __name__ == "__main__":
    main()
