# 整数扰动序列的算术编码

import numpy as np


PAYLOAD_MAGIC = 0x5955
PAYLOAD_VERSION = 1


def serialize_integer_perturbation_sequence(  # 按行序列化整数扰动
    perturbation: np.ndarray,
) -> list[int]:
    values = np.asarray(perturbation)
    if not np.all(np.isfinite(values)):
        raise ValueError("扰动含非有限值")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("扰动必须为整数")
    return rounded.astype(np.int32).reshape(-1).tolist()


def restore_integer_perturbation_shape(  # 恢复扰动数组形状
    sequence: list[int],
    shape: tuple[int, ...],
) -> np.ndarray:
    if len(sequence) != int(np.prod(shape)):
        raise ValueError("序列长度与目标形状不匹配")
    return np.asarray(sequence, dtype=np.int32).reshape(shape)


def cal_integer_symbol_frequencies(  # 统计有序频率表
    sequence: list[int],
    end_symbol: int,
) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for symbol in sequence + [end_symbol]:
        counts[symbol] = counts.get(symbol, 0) + 1
    return sorted(counts.items())


def _cal_cumulative_ranges(  # 计算累计频率区间
    frequencies: list[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    cumulative = 0
    ranges: dict[int, tuple[int, int]] = {}
    for symbol, frequency in frequencies:
        if frequency <= 0:
            raise ValueError("符号频率必须为正数")
        ranges[symbol] = (cumulative, cumulative + frequency)
        cumulative += frequency
    return ranges


def arithmetic_encode(  # 复现作者仅返回算术码的路径
    sequence: list[int],
    precision: int = 32,
) -> tuple[list[int], dict[str, object]]:
    if precision < 8:
        raise ValueError("算术编码精度过低")
    if any(symbol < -(2**31) or symbol >= 2**31 for symbol in sequence):
        raise ValueError("符号超出32位有符号整数范围")

    author_end_symbol = "!"
    stream: list[int | str] = [*sequence, author_end_symbol]
    frequencies_by_symbol = {
        symbol: stream.count(symbol) for symbol in set(stream)
    }
    cumulative_ranges: dict[int | str, tuple[int, int]] = {}
    cumulative = 0
    for symbol, frequency in frequencies_by_symbol.items():
        cumulative_ranges[symbol] = (cumulative, cumulative + frequency)
        cumulative += frequency

    full = 2**precision
    half = full // 2
    quarter = half // 2
    low = 0
    high = full
    pending_bits = 0
    code: list[int] = []
    for symbol in stream:
        symbol_low, symbol_high = cumulative_ranges[symbol]
        interval = high - low
        high = low + interval * symbol_high // len(stream)
        low = low + interval * symbol_low // len(stream)
        while True:
            if high < half:
                code.append(0)
                code.extend([1] * pending_bits)
                pending_bits = 0
                low *= 2
                high *= 2
            elif low >= half:
                code.append(1)
                code.extend([0] * pending_bits)
                pending_bits = 0
                low = 2 * (low - half)
                high = 2 * (high - half)
            elif low >= quarter and high < 3 * quarter:
                pending_bits += 1
                low = 2 * (low - quarter)
                high = 2 * (high - quarter)
            else:
                break

    pending_bits += 1
    if low <= quarter:
        code.append(0)
        code.extend([1] * pending_bits)
    else:
        code.append(1)
        code.extend([0] * pending_bits)

    integer_end_symbol = max(sequence, default=-1) + 1
    frequencies = [
        (
            integer_end_symbol if symbol == author_end_symbol else int(symbol),
            frequency,
        )
        for symbol, frequency in frequencies_by_symbol.items()
    ]
    metadata: dict[str, object] = {
        "precision": precision,
        "message_length": len(sequence),
        "code_length": len(code),
        "end_symbol": integer_end_symbol,
        "frequencies": frequencies,
    }
    return code, metadata


def _unsigned_integer_to_bits(value: int, width: int) -> list[int]:  # 定宽无符号整数转比特
    if value < 0 or value >= 2**width:
        raise ValueError("无符号数超出指定位宽")
    return [int(bit) for bit in f"{value:0{width}b}"]


def _signed_integer_to_bits(value: int, width: int) -> list[int]:  # 定宽有符号整数转比特
    if value < -(2 ** (width - 1)) or value >= 2 ** (width - 1):
        raise ValueError("有符号数超出指定位宽")
    return _unsigned_integer_to_bits(value % (2**width), width)


def _bits_to_unsigned_integer(bits: list[int]) -> int:  # 比特转无符号整数
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("比特序列只能包含0和1")
    return int("".join(str(bit) for bit in bits), 2) if bits else 0


def _bits_to_signed_integer(bits: list[int]) -> int:  # 比特转有符号整数
    value = _bits_to_unsigned_integer(bits)
    return value - 2 ** len(bits) if bits and bits[0] == 1 else value


def arithmetic_encode_sup(  # 按作者区间更新方式编码
    sequence: list[int],
    precision: int = 32,
) -> tuple[list[int], dict[str, object]]:
    if precision < 8:
        raise ValueError("算术编码精度过低")
    if any(symbol < -(2**31) or symbol >= 2**31 for symbol in sequence):
        raise ValueError("符号超出32位有符号整数范围")

    end_symbol = max(sequence, default=-1) + 1
    if end_symbol >= 2**31:
        raise ValueError("没有可用的32位有符号结束符")
    stream = sequence + [end_symbol]
    frequencies = cal_integer_symbol_frequencies(sequence, end_symbol)
    cumulative_ranges = _cal_cumulative_ranges(frequencies)
    stream_size = len(stream)

    full = 2**precision
    half = full // 2
    quarter = half // 2
    if stream_size >= quarter:
        raise ValueError("消息过长，超出当前算术编码精度")

    low = 0
    high = full
    pending_bits = 0
    code: list[int] = []

    for symbol in stream:
        symbol_low, symbol_high = cumulative_ranges[symbol]
        interval = high - low
        high = low + interval * symbol_high // stream_size
        low = low + interval * symbol_low // stream_size

        while True:
            if high < half:
                code.append(0)
                code.extend([1] * pending_bits)
                pending_bits = 0
                low *= 2
                high *= 2
            elif low >= half:
                code.append(1)
                code.extend([0] * pending_bits)
                pending_bits = 0
                low = 2 * (low - half)
                high = 2 * (high - half)
            elif low >= quarter and high < 3 * quarter:
                pending_bits += 1
                low = 2 * (low - quarter)
                high = 2 * (high - quarter)
            else:
                break

    pending_bits += 1
    if low <= quarter:
        code.append(0)
        code.extend([1] * pending_bits)
    else:
        code.append(1)
        code.extend([0] * pending_bits)

    metadata: dict[str, object] = {
        "precision": precision,
        "message_length": len(sequence),
        "code_length": len(code),
        "end_symbol": end_symbol,
        "frequencies": frequencies,
    }
    return code, metadata


def arithmetic_decode_sup(  # 使用频率表恢复整数序列
    code: list[int],
    metadata: dict[str, object],
) -> list[int]:
    if any(bit not in (0, 1) for bit in code):
        raise ValueError("算术码只能包含0和1")

    precision = int(metadata["precision"])
    message_length = int(metadata["message_length"])
    end_symbol = int(metadata["end_symbol"])
    frequencies = [(int(a), int(b)) for a, b in metadata["frequencies"]]
    cumulative_ranges = _cal_cumulative_ranges(frequencies)
    stream_size = sum(frequency for _, frequency in frequencies)
    if stream_size != message_length + 1:
        raise ValueError("频率表与消息长度不匹配")

    full = 2**precision
    half = full // 2
    quarter = half // 2
    low = 0
    high = full
    value = 0
    code_index = 0
    for _ in range(precision):
        value = value * 2 + (code[code_index] if code_index < len(code) else 0)
        code_index += 1

    message: list[int] = []
    ordered_symbols = [symbol for symbol, _ in frequencies]
    for _ in range(message_length + 1):
        matched_symbol = None
        for symbol in ordered_symbols:
            symbol_low, symbol_high = cumulative_ranges[symbol]
            interval = high - low
            next_low = low + interval * symbol_low // stream_size
            next_high = low + interval * symbol_high // stream_size
            if next_low <= value < next_high:
                matched_symbol = symbol
                low = next_low
                high = next_high
                break
        if matched_symbol is None:
            raise ValueError("算术码无法解码")
        if matched_symbol == end_symbol:
            if len(message) != message_length:
                raise ValueError("结束符出现早于预期长度")
            return message
        message.append(matched_symbol)

        while True:
            if high < half:
                low *= 2
                high *= 2
                value *= 2
            elif low >= half:
                low = 2 * (low - half)
                high = 2 * (high - half)
                value = 2 * (value - half)
            elif low >= quarter and high < 3 * quarter:
                low = 2 * (low - quarter)
                high = 2 * (high - quarter)
                value = 2 * (value - quarter)
            else:
                break
            if code_index < len(code):
                value += code[code_index]
            code_index += 1

    raise ValueError("未找到算术编码结束符")


def build_arithmetic_payload_sup(  # 将频率表和编码合成载荷
    code: list[int],
    metadata: dict[str, object],
) -> list[int]:
    frequencies = [(int(a), int(b)) for a, b in metadata["frequencies"]]
    header = _unsigned_integer_to_bits(PAYLOAD_MAGIC, 16)
    header += _unsigned_integer_to_bits(PAYLOAD_VERSION, 8)
    header += _unsigned_integer_to_bits(int(metadata["precision"]), 8)
    header += _unsigned_integer_to_bits(int(metadata["message_length"]), 32)
    header += _unsigned_integer_to_bits(len(code), 32)
    header += _signed_integer_to_bits(int(metadata["end_symbol"]), 32)
    header += _unsigned_integer_to_bits(len(frequencies), 16)
    for symbol, frequency in frequencies:
        header += _signed_integer_to_bits(symbol, 32)
        header += _unsigned_integer_to_bits(frequency, 32)
    return header + code


def parse_arithmetic_payload_sup(  # 从载荷读取频率表和编码
    payload_bits: list[int],
) -> tuple[list[int], dict[str, object]]:
    if len(payload_bits) < 144:
        raise ValueError("算术载荷头不完整")
    position = 0

    def read(width: int) -> list[int]:  # 顺序读取定宽字段
        nonlocal position
        field = payload_bits[position : position + width]
        if len(field) != width:
            raise ValueError("算术载荷已截断")
        position += width
        return field

    if _bits_to_unsigned_integer(read(16)) != PAYLOAD_MAGIC:
        raise ValueError("算术载荷标识无效")
    if _bits_to_unsigned_integer(read(8)) != PAYLOAD_VERSION:
        raise ValueError("不支持该算术载荷版本")
    precision = _bits_to_unsigned_integer(read(8))
    message_length = _bits_to_unsigned_integer(read(32))
    code_length = _bits_to_unsigned_integer(read(32))
    end_symbol = _bits_to_signed_integer(read(32))
    symbol_count = _bits_to_unsigned_integer(read(16))
    frequencies = []
    for _ in range(symbol_count):
        symbol = _bits_to_signed_integer(read(32))
        frequency = _bits_to_unsigned_integer(read(32))
        frequencies.append((symbol, frequency))
    code = read(code_length)
    if position != len(payload_bits):
        raise ValueError("算术载荷含多余比特")
    metadata: dict[str, object] = {
        "precision": precision,
        "message_length": message_length,
        "code_length": code_length,
        "end_symbol": end_symbol,
        "frequencies": frequencies,
    }
    return code, metadata
