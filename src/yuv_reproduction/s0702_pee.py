# 预测误差扩展与UV载荷嵌入

import numpy as np


MESSAGE_LENGTH_BITS = 18


def _val_integer_channel(channel: np.ndarray) -> np.ndarray:  # 检查二维整数载体
    values = np.asarray(channel)
    if values.ndim != 2:
        raise ValueError("载体通道必须是二维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("载体通道含非有限值")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("载体通道必须为整数")
    return rounded.astype(np.int64)


def _val_bits(bits: list[int]) -> None:  # 检查二进制序列
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("消息只能包含0和1")


def _integer_to_bits(value: int, width: int) -> list[int]:  # 整数转定宽比特
    if value < 0 or value >= 2**width:
        raise ValueError("消息长度超出头部范围")
    return [int(bit) for bit in f"{value:0{width}b}"]


def _bits_to_integer(bits: list[int]) -> int:  # 定宽比特转整数
    _val_bits(bits)
    return int("".join(str(bit) for bit in bits), 2) if bits else 0


def generate_checkerboard_prediction(  # 使用四邻域预测半数像素
    channel: np.ndarray,
    target_parity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _val_integer_channel(channel)
    if target_parity not in (0, 1):
        raise ValueError("目标奇偶值只能是0或1")
    height, width = values.shape
    padded = np.pad(values, ((1, 1), (1, 1)), mode="constant")
    predicted = values.copy()
    mask = np.zeros_like(values, dtype=bool)
    for row in range(height):
        for column in range(width):
            if (row + column) % 2 == target_parity:
                padded_row = row + 1
                padded_column = column + 1
                neighbor_sum = (
                    padded[padded_row - 1, padded_column]
                    + padded[padded_row + 1, padded_column]
                    + padded[padded_row, padded_column - 1]
                    + padded[padded_row, padded_column + 1]
                )
                predicted[row, column] = neighbor_sum // 4
                mask[row, column] = True
    prediction_error = values - predicted
    return predicted, prediction_error, mask


def embed_bits_in_prediction_errors(  # 按公式3、4扩展误差
    predicted: np.ndarray,
    prediction_error: np.ndarray,
    mask: np.ndarray,
    bits: list[int],
    threshold: int,
) -> tuple[np.ndarray, int]:
    if threshold <= 0:
        raise ValueError("PEE阈值必须为正数")
    _val_bits(bits)
    embedded_error = np.zeros_like(prediction_error, dtype=np.int64)
    bit_index = 0
    for row, column in np.argwhere(mask):
        error = int(prediction_error[row, column])
        if -threshold <= error < threshold:
            if bit_index >= len(bits):
                raise ValueError("嵌入比特少于可用容量")
            embedded_error[row, column] = 2 * error + bits[bit_index]
            bit_index += 1
        elif error >= threshold:
            embedded_error[row, column] = error + threshold
        else:
            embedded_error[row, column] = error - threshold
    return predicted.astype(np.int64) + embedded_error, bit_index


def extract_bits_from_prediction_errors(  # 按公式5、6提取并恢复
    predicted: np.ndarray,
    expanded_error: np.ndarray,
    mask: np.ndarray,
    threshold: int,
) -> tuple[np.ndarray, list[int]]:
    if threshold <= 0:
        raise ValueError("PEE阈值必须为正数")
    recovered_error = np.zeros_like(expanded_error, dtype=np.int64)
    recovered_bits: list[int] = []
    for row, column in np.argwhere(mask):
        error = int(expanded_error[row, column])
        if -2 * threshold <= error < 2 * threshold:
            recovered_bits.append(error % 2)
            recovered_error[row, column] = error // 2
        elif error < -2 * threshold:
            recovered_error[row, column] = error + threshold
        else:
            recovered_error[row, column] = error - threshold
    return predicted.astype(np.int64) + recovered_error, recovered_bits


def _embed_two_checkerboard_passes(  # 先交叉像素再点像素嵌入
    carrier: np.ndarray,
    padded_bits: list[int],
    threshold: int,
) -> tuple[np.ndarray, int]:
    cross_prediction, cross_error, cross_mask = generate_checkerboard_prediction(
        carrier,
        0,
    )
    cross_stego, cross_count = embed_bits_in_prediction_errors(
        cross_prediction,
        cross_error,
        cross_mask,
        padded_bits,
        threshold,
    )
    dot_prediction, dot_error, dot_mask = generate_checkerboard_prediction(cross_stego, 1)
    dot_stego, dot_count = embed_bits_in_prediction_errors(
        dot_prediction,
        dot_error,
        dot_mask,
        padded_bits[cross_count:],
        threshold,
    )
    return dot_stego, cross_count + dot_count


def select_pee_threshold(  # 按作者1、6、11顺序选择阈值
    carrier: np.ndarray,
    padded_bits: list[int],
    required_bits: int,
    enforce_uint8_bounds: bool = True,
) -> tuple[int, np.ndarray, int]:
    values = _val_integer_channel(carrier)
    maximum_threshold = 251
    if not enforce_uint8_bounds:
        maximum_absolute_value = int(np.max(np.abs(values)))
        maximum_threshold = max(251, 6 * maximum_absolute_value + 6)
    for threshold in range(1, maximum_threshold + 1, 5):
        stego, capacity = _embed_two_checkerboard_passes(
            values,
            padded_bits,
            threshold,
        )
        bounds_valid = stego.min() >= 0 and stego.max() <= 255
        if capacity >= required_bits and (bounds_valid or not enforce_uint8_bounds):
            return threshold, stego, capacity
    raise ValueError("载体容量不足或像素将越界")


def embed_bits_pee(  # 添加长度头并执行两遍PEE
    carrier: np.ndarray,
    payload_bits: list[int],
    threshold: int | None = None,
    enforce_uint8_bounds: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    values = _val_integer_channel(carrier)
    _val_bits(payload_bits)
    if len(payload_bits) >= 2**MESSAGE_LENGTH_BITS:
        raise ValueError("载荷长度超出18位头部范围")
    framed_bits = _integer_to_bits(len(payload_bits), MESSAGE_LENGTH_BITS) + payload_bits
    if len(framed_bits) > values.size:
        raise ValueError("载体容量不足，无法嵌入带头载荷")
    padded_bits = framed_bits + [0] * (values.size - len(framed_bits))

    if threshold is None:
        selected_threshold, stego, capacity = select_pee_threshold(
            values,
            padded_bits,
            len(framed_bits),
            enforce_uint8_bounds,
        )
    else:
        selected_threshold = int(threshold)
        stego, capacity = _embed_two_checkerboard_passes(
            values,
            padded_bits,
            selected_threshold,
        )
        if capacity < len(framed_bits):
            raise ValueError("所选阈值下的载体容量不足")
        if enforce_uint8_bounds and (stego.min() < 0 or stego.max() > 255):
            raise ValueError("PEE嵌入会使像素超出0至255")

    metadata = {
        "threshold": selected_threshold,
        "payload_length": len(payload_bits),
        "framed_length": len(framed_bits),
        "capacity": capacity,
    }
    return stego.astype(np.int32), metadata


def extract_bits_pee(  # 逆序提取两遍PEE
    stego: np.ndarray,
    threshold: int,
) -> tuple[list[int], np.ndarray, dict[str, int]]:
    values = _val_integer_channel(stego)
    dot_prediction, dot_error, dot_mask = generate_checkerboard_prediction(values, 1)
    dot_recovered, dot_bits = extract_bits_from_prediction_errors(
        dot_prediction,
        dot_error,
        dot_mask,
        threshold,
    )
    cross_prediction, cross_error, cross_mask = generate_checkerboard_prediction(
        dot_recovered,
        0,
    )
    recovered, cross_bits = extract_bits_from_prediction_errors(
        cross_prediction,
        cross_error,
        cross_mask,
        threshold,
    )
    extracted_bits = cross_bits + dot_bits
    if len(extracted_bits) < MESSAGE_LENGTH_BITS:
        raise ValueError("提取数据缺少完整长度头")
    payload_length = _bits_to_integer(extracted_bits[:MESSAGE_LENGTH_BITS])
    payload_end = MESSAGE_LENGTH_BITS + payload_length
    if payload_end > len(extracted_bits):
        raise ValueError("提取的载荷长度超过可用比特数")
    metadata = {
        "threshold": int(threshold),
        "payload_length": payload_length,
        "extracted_capacity": len(extracted_bits),
    }
    return extracted_bits[MESSAGE_LENGTH_BITS:payload_end], recovered.astype(np.int32), metadata


def embed_payload_uv(  # 将载荷嵌入U和V
    u_channel: np.ndarray,
    v_channel: np.ndarray,
    payload_bits: list[int],
    threshold: int | None = None,
    enforce_uint8_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    u_values = _val_integer_channel(u_channel)
    v_values = _val_integer_channel(v_channel)
    if u_values.shape != v_values.shape:
        raise ValueError("U、V通道形状必须一致")
    stacked_uv = np.vstack((u_values, v_values))
    stego_uv, metadata = embed_bits_pee(
        stacked_uv,
        payload_bits,
        threshold,
        enforce_uint8_bounds,
    )
    height = u_values.shape[0]
    return stego_uv[:height], stego_uv[height:], metadata


def extract_payload_uv(  # 提取载荷并恢复U和V
    stego_u: np.ndarray,
    stego_v: np.ndarray,
    threshold: int,
) -> tuple[list[int], np.ndarray, np.ndarray, dict[str, int]]:
    u_values = _val_integer_channel(stego_u)
    v_values = _val_integer_channel(stego_v)
    if u_values.shape != v_values.shape:
        raise ValueError("U、V通道形状必须一致")
    stacked_uv = np.vstack((u_values, v_values))
    payload, recovered_uv, metadata = extract_bits_pee(
        stacked_uv,
        threshold,
    )
    height = u_values.shape[0]
    return payload, recovered_uv[:height], recovered_uv[height:], metadata


def embed_payload_uv_rounds(  # 按作者方式分轮嵌入长载荷
    u_channel: np.ndarray,
    v_channel: np.ndarray,
    payload_bits: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    u_values = _val_integer_channel(u_channel)
    v_values = _val_integer_channel(v_channel)
    if u_values.shape != v_values.shape:
        raise ValueError("U、V通道形状必须一致")
    carrier = np.vstack((u_values, v_values))
    chunk_capacity = carrier.size - MESSAGE_LENGTH_BITS
    remaining = list(payload_bits)
    thresholds: list[int] = []
    while remaining:
        chunk = remaining[:chunk_capacity]
        carrier, metadata = embed_bits_pee(
            carrier,
            chunk,
            enforce_uint8_bounds=False,
        )
        thresholds.append(int(metadata["threshold"]))
        remaining = remaining[len(chunk) :]
    height = u_values.shape[0]
    return carrier[:height], carrier[height:], {
        "thresholds": thresholds,
        "embedding_rounds": len(thresholds),
        "payload_length": len(payload_bits),
    }


def extract_payload_uv_rounds_sup(  # 逆序恢复作者多轮PEE
    stego_u: np.ndarray,
    stego_v: np.ndarray,
    thresholds: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    u_values = _val_integer_channel(stego_u)
    v_values = _val_integer_channel(stego_v)
    if u_values.shape != v_values.shape:
        raise ValueError("U、V通道形状必须一致")
    if not thresholds:
        raise ValueError("At least one PEE threshold is required")
    carrier = np.vstack((u_values, v_values))
    reversed_chunks: list[list[int]] = []
    for threshold in reversed(thresholds):
        payload, carrier, _ = extract_bits_pee(
            carrier,
            int(threshold),
        )
        reversed_chunks.append(payload)
    payload_bits = [bit for chunk in reversed(reversed_chunks) for bit in chunk]
    height = u_values.shape[0]
    return payload_bits, carrier[:height], carrier[height:]
