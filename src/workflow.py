"""主进程只读固定样本、调度独立作者进程和读取结果。"""
import json
import os
import subprocess
import sys
from pathlib import Path
from tqdm.auto import tqdm
from .common import arguments, save_json, file_hash, code_hashes
from .config import ROOT
from .author_cache import prepare_cache

PROTOCOL = "author_original_20261111"


def run_worker(request, request_path):
    save_json(request_path, request)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run([sys.executable, str(ROOT / "author_worker.py"), str(request_path)],
                   cwd=str(ROOT), env=environment, check=True)


def run_method(bundle, output, method, args):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    record_path = output / "result.json"
    values = arguments(args)
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        expected = previous["arguments"].copy()
        expected["resume"] = values["resume"]
        if args.resume and previous["status"] == "complete" and previous.get("protocol") == PROTOCOL and expected == values:
            if file_hash(output / "model.pt") != previous["checkpoint_sha256"]:
                raise ValueError("已保存模型散列变化。")
            tqdm.write("跳过已完成且参数一致的作者流程：" + str(output))
            return previous
        raise FileExistsError("此目录已有结果，请使用新输出目录，或用--resume继续参数一致的当前流程：" + str(output))
    info = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if info["settings"]["dataset"] != args.dataset or info["settings"]["attack"] != args.attack:
        raise ValueError("样本包的数据集或生成方法与本次参数不一致。")
    for name, digest in info.get("sha256", {}).items():
        if file_hash(bundle / name) != digest:
            raise ValueError("固定样本散列变化：" + name)
    if args.attack is None and method != "none":
        raise ValueError("正常图入口只用于普通分类训练。")
    cache = prepare_cache(bundle) if args.attack is not None else None
    output.mkdir(parents=True, exist_ok=False)
    record = {"protocol": PROTOCOL, "status": "running", "method": method,
        "dataset": args.dataset, "attack": args.attack, "model": args.model, "seed": args.seed,
        "arguments": values, "bundle": str(bundle), "bundle_id": info["bundle_id"],
        "bundle_counts": info.get("counts"), "generator_parameters": info.get("actual_generator_parameters"),
        "budget": info.get("budget"), "class_allocation": info.get("class_allocation"),
        "code_sha256": code_hashes(), "cache": str(cache) if cache else None,
        "cache_sha256": file_hash(cache) if cache else None}
    save_json(record_path, record)
    request = {"operation": "train", "method": method, "arguments": values, "bundle": str(bundle),
               "output": str(output), "cache": str(cache) if cache else None}
    with tqdm(total=1, desc=method + "原流程训练及评估", unit="次") as progress:
        try:
            run_worker(request, output / "request.json")
        except subprocess.CalledProcessError:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["status"] == "running":
                record.update(status="failed", error="作者进程退出，详细错误见终端输出。")
                save_json(record_path, record)
            raise
        progress.update(1)
    return json.loads(record_path.read_text(encoding="utf-8"))


def evaluate_checkpoint(bundle, checkpoint, output, device_name):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    record = json.loads((checkpoint.parent / "result.json").read_text(encoding="utf-8"))
    if file_hash(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("模型文件与运行记录的散列不一致。")
    if record.get("cache_sha256") and file_hash(record["cache"]) != record["cache_sha256"]:
        raise ValueError("原始触发输入缓存散列变化。")
    request = {"operation": "evaluate", "bundle": str(Path(bundle).resolve()), "checkpoint": str(checkpoint),
               "output": str(output), "device": device_name, "cache": record["cache"]}
    run_worker(request, output.with_suffix(".request.json"))
    return json.loads(output.read_text(encoding="utf-8"))
