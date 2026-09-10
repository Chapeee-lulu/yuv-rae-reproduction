# 第1步：检查环境和本地数据

import json
import sys
from pathlib import Path

# 定位目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from yuv_reproduction.s0102_data import val_png
from yuv_reproduction.s0101_env import collect_runtime_environment


def main() -> None:  # 执行step01
    environment_report = collect_runtime_environment()
    dataset_report = val_png(
        PROJECT_ROOT / "data" / "original_img",
    )
    complete_report = {
        "step": "1_environment_project_structure_and_local_data",
        "environment": environment_report,
        "dataset": dataset_report.to_dictionary(),
    }

    # 保存检查结果
    result_path = PROJECT_ROOT / "results" / "step01.json"
    result_path.write_text(
        json.dumps(complete_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(complete_report, ensure_ascii=False, indent=2))
    print(f"step01通过，相关结果保存到: {result_path}")


if __name__ == "__main__":
    main()
