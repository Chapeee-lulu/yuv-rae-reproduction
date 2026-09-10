# 检查本地ImageNet图片数据

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

ORIGINAL_IMAGE_NAME_PATTERN = re.compile(r"^(?P<index>\d{4})\.png$")


@dataclass(frozen=True)
class LocalImageDatasetReport:
    directory: str
    image_count: int
    filename_pattern: str
    all_images_readable: bool
    image_modes: list[str]
    image_sizes: list[list[int]]

    def to_dictionary(self) -> dict[str, object]:
        return asdict(self)


def val_png(  # 检查数量、格式、命名和可读性
    image_directory: Path,
    expected_image_count: int = 1000,
) -> LocalImageDatasetReport:
    if not image_directory.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_directory}")

    directory_entries = sorted(image_directory.iterdir())
    non_files = [entry.name for entry in directory_entries if not entry.is_file()]
    if non_files:
        raise ValueError(f"发现非文件项：{non_files[:5]}")

    if len(directory_entries) != expected_image_count:
        raise ValueError(
            f"应有{expected_image_count}张图片，实际找到{len(directory_entries)}"
        )

    non_png_files = [path.name for path in directory_entries if path.suffix.lower() != ".png"]
    if non_png_files:
        raise ValueError(f"发现非PNG文件：{non_png_files[:5]}")

    invalid_names = [
        path.name
        for path in directory_entries
        if ORIGINAL_IMAGE_NAME_PATTERN.fullmatch(path.name) is None
    ]
    if invalid_names:
        raise ValueError(f"图片名称无效: {invalid_names[:5]}")

    image_modes: set[str] = set()
    image_sizes: set[tuple[int, int]] = set()
    for image_path in directory_entries:
        try:
            with Image.open(image_path) as image:
                if image.format != "PNG":
                    raise ValueError(
                        f"文件扩展名为PNG，但实际格式为{image.format}: {image_path.name}"
                    )
                image_modes.add(image.mode)
                image_sizes.add(image.size)
                image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError(f"PNG图片无法读取：{image_path.name}") from error

    return LocalImageDatasetReport(
        directory=str(image_directory.resolve()),
        image_count=len(directory_entries),
        filename_pattern=ORIGINAL_IMAGE_NAME_PATTERN.pattern,
        all_images_readable=True,
        image_modes=sorted(image_modes),
        image_sizes=[list(size) for size in sorted(image_sizes)],
    )
