"""每个作者在独立进程中使用自己的原始顶层包。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_directory(method, attack):
    if method in ["none", "abp", "rigbd"]:
        return ROOT / "upstream/RIGBD"
    if method == "prune":
        if attack == "DPGBA":
            return ROOT / "upstream/LoSplit/DPGBA"
        return ROOT / "upstream/LoSplit"
    if method == "losplit":
        if attack == "DPGBA":
            return ROOT / "upstream/LoSplit/DPGBA"
        return ROOT / "upstream/LoSplit"
    if method == "dshield":
        return ROOT / "upstream/DShield/NodeClassificationTasks"
    if method == "mad":
        if attack == "DPGBA":
            return ROOT / "upstream/MAD/MAD/DPGBA/DPGBA/run"
        return ROOT / "upstream/MAD/MAD/GTA_and_UGBA"
    raise ValueError("未知作者方法：" + method)


def activate(method, attack):
    folder = source_directory(method, attack)
    if not folder.is_dir():
        raise FileNotFoundError("作者源码目录不存在：" + str(folder))
    sys.path.insert(0, str(folder))
    return folder
