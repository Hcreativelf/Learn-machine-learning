"""环境自检脚本：确认 VSCode 里选中的解释器是哪一个、依赖是否齐全、数据是否到位。

在 VSCode 里打开本文件后按 Ctrl+F5（或点击右上角 ▶）运行即可。
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# 交互窗口里没有 __file__，退回到当前工作目录
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

DATA_FILES = ["train.csv", "test.csv", "sample_submission.csv"]

# (模块名, pip 包名)
PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("jupyter", "jupyter"),
    ("ipykernel", "ipykernel"),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 62)
    print("解释器路径 :", sys.executable)
    print("Python 版本:", platform.python_version(), f"({platform.architecture()[0]})")
    print("是否虚拟环境:", sys.prefix != sys.base_prefix)
    print("=" * 62)

    missing = []
    for module_name, pip_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK]   {pip_name:<14} {getattr(module, '__version__', 'n/a')}")
        except Exception as exc:  # noqa: BLE001
            missing.append(pip_name)
            print(f"[缺失] {pip_name:<14} {exc}")

    print("-" * 62)
    print("TF-DF（本机 Windows 无法安装，属正常）:", end=" ")
    try:
        import tensorflow_decision_forests as tfdf  # noqa: F401

        print(f"可用，版本 {tfdf.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"不可用 —— {type(exc).__name__}")

    print("-" * 62)
    data_dir = PROJECT_DIR / "data"
    print(f"数据目录: {data_dir}")
    if data_dir.exists():
        for name in DATA_FILES:
            path = data_dir / name
            print(f"  {'[有]' if path.exists() else '[缺]'} {name}")
    else:
        print("  [缺] data 目录不存在，请把 Kaggle 的数据解压到这里")

    print("-" * 62)
    if missing:
        print("缺少依赖，请在终端执行：")
        print("  uv pip install -r requirements.txt")
        return 1

    print("环境检查通过，可以运行 house_price_sklearn.py 了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
