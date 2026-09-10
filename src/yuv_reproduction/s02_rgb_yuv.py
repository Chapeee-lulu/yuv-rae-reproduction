# 转换8位RGB与YUV 4:4:4图像

import numpy as np


def _val_three_channel_image(image: np.ndarray) -> None:  # 检查输入形状
    if not isinstance(image, np.ndarray):
        raise TypeError("图像必须是NumPy数组")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("图像形状必须为(H, W, 3)")


def cal_rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:  # 公式13计算浮点YUV
    _val_three_channel_image(rgb_image)
    rgb_float = rgb_image.astype(np.float32)
    red = rgb_float[:, :, 0]
    green = rgb_float[:, :, 1]
    blue = rgb_float[:, :, 2]

    yuv_float = np.empty_like(rgb_float)
    yuv_float[:, :, 0] = 0.299 * red + 0.587 * green + 0.114 * blue
    yuv_float[:, :, 1] = -0.1687 * red - 0.3313 * green + 0.5 * blue + 128
    yuv_float[:, :, 2] = 0.5 * red - 0.4187 * green - 0.0813 * blue + 128
    return yuv_float


def cal_yuv_to_rgb(yuv_image: np.ndarray) -> np.ndarray:  # 公式14计算浮点RGB
    _val_three_channel_image(yuv_image)
    yuv_float = yuv_image.astype(np.float32)
    luminance = yuv_float[:, :, 0]
    blue_difference = yuv_float[:, :, 1] - 128
    red_difference = yuv_float[:, :, 2] - 128

    rgb_float = np.empty_like(yuv_float)
    rgb_float[:, :, 0] = luminance + 1.402 * red_difference
    rgb_float[:, :, 1] = luminance - 0.34414 * blue_difference - 0.71414 * red_difference
    rgb_float[:, :, 2] = luminance + 1.772 * blue_difference
    return rgb_float


def round_color_channel_values(float_image: np.ndarray) -> np.ndarray:  # 取整
    _val_three_channel_image(float_image)
    return np.rint(float_image)


def clip_color_channel_values(rounded_image: np.ndarray) -> np.ndarray:  # 裁剪，便于uint8
    _val_three_channel_image(rounded_image)
    return np.clip(rounded_image, 0, 255)


def convert_color_channels_to_uint8(clipped_image: np.ndarray) -> np.ndarray:  # 转为uint8
    _val_three_channel_image(clipped_image)
    return clipped_image.astype(np.uint8)


def rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:  # RGB转YUV，没转uint8，可能出现256
    yuv_float = cal_rgb_to_yuv(rgb_image)
    return round_color_channel_values(yuv_float)


def yuv_to_rgb(yuv_image: np.ndarray) -> np.ndarray:  # YUV转RGB，保留浮点数
    rgb_float = cal_yuv_to_rgb(yuv_image)
    rgb_rounded = round_color_channel_values(rgb_float)
    return clip_color_channel_values(rgb_rounded).astype(yuv_image.dtype)


def rgb_to_yuv_cmp(  # 对比：裁剪并转uint8
    rgb_image: np.ndarray,
) -> np.ndarray:
    yuv_rounded = rgb_to_yuv(rgb_image)
    yuv_clipped = clip_color_channel_values(yuv_rounded)
    return convert_color_channels_to_uint8(yuv_clipped)


def yuv_to_rgb_cmp(  # 对比：转uint8
    yuv_image: np.ndarray,
) -> np.ndarray:
    rgb_image = yuv_to_rgb(yuv_image)
    return convert_color_channels_to_uint8(rgb_image)
