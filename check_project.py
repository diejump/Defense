"""编译项目文件并运行协议检查，不训练公开数据集。"""
import ast
from pathlib import Path
import unittest
from tqdm.auto import tqdm


def main():
    root = Path(__file__).resolve().parent
    paths = list((root / "src").glob("*.py")) + list((root / "upstream").rglob("*.py"))
    for path in tqdm(paths, desc="检查Python语法", unit="文件"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("语法与协议检查全部通过。")


if __name__ == "__main__":
    main()
