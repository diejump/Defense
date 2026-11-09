"""设备、随机种子、计时与文件保存。"""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch


def setup(seed, device_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用GPU，请检查环境，或明确设置 --device cpu。")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_clock(device):
    synchronize(device)
    return time.perf_counter()


def elapsed(begin, device):
    synchronize(device)
    return time.perf_counter() - begin


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_training(folder, device):
    # 训练阶段只打开training.pt，不打开包含测试真值的evaluation.pt。
    data = torch.load(Path(folder) / "training.pt", map_location=device)
    return data


def cpu_state(model):
    state = {}
    for name, value in model.state_dict().items():
        state[name] = value.detach().cpu().clone()
    return state


def arguments(args):
    values = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        values[name] = value
    return values


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    result = {}
    for folder in [root / "src", root / "upstream"]:
        for path in sorted(folder.rglob("*.py")):
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result
