"""RGB与YUV颜色空间转换。

本模块使用论文公式(13)、(14)，三个通道始终保持相同的高和宽，因此
数据表示属于YUV 4:4:4。文件名和函数名不重复写444，具体采样方式在
文档和函数说明中声明。
"""

import numpy as np


def _validate_three_channel_image(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name}必须是形状为(H, W, 3)的数组，当前为{array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}包含NaN或无穷大")
    return array


def rgb_to_yuv(rgb: np.ndarray, *, quantize: bool = True) -> np.ndarray:
    """按照论文公式(13)将RGB转换为YUV 4:4:4。

    quantize=True表示模拟8位图像处理中对Y/U/V取整数的过程；
    quantize=False保留浮点值，用于分析矩阵转换本身的数值误差。
    """

    rgb_float = _validate_three_channel_image(rgb, "rgb")
    red = rgb_float[..., 0]
    green = rgb_float[..., 1]
    blue = rgb_float[..., 2]

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    chroma_u = -0.1687 * red - 0.3313 * green + 0.500 * blue + 128.0
    chroma_v = 0.500 * red - 0.4187 * green - 0.0813 * blue + 128.0

    yuv = np.stack((luminance, chroma_u, chroma_v), axis=-1)
    return np.rint(yuv) if quantize else yuv


def yuv_to_rgb(
    yuv: np.ndarray,
    *,
    quantize: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """按照论文公式(14)将YUV 4:4:4转换回RGB。"""

    yuv_float = _validate_three_channel_image(yuv, "yuv")
    luminance = yuv_float[..., 0]
    chroma_u = yuv_float[..., 1]
    chroma_v = yuv_float[..., 2]

    red = luminance + 1.402 * (chroma_v - 128.0)
    green = luminance - 0.34414 * (chroma_u - 128.0) - 0.71414 * (chroma_v - 128.0)
    blue = luminance + 1.772 * (chroma_u - 128.0)

    rgb = np.stack((red, green, blue), axis=-1)
    if quantize:
        rgb = np.rint(rgb)
    if clip:
        rgb = np.clip(rgb, 0.0, 255.0)
    return rgb

