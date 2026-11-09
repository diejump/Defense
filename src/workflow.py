"""一个方法一次运行：读取训练数据→训练→保存→评估→保存指标。"""
import json
import platform
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_geometric
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .common import setup, arguments, load_training, save_json, cpu_state, file_hash, start_clock, elapsed, code_hashes
from .models import make_model, describe
from .training import fit_classifier
from .abp import filter_edges
from .evaluation import evaluate


def fit_method(method, data, args, device):
    if method == "rigbd":
        from .rigbd import train
        return train(data, args, device)
    if method == "losplit":
        from .losplit import train
        return train(data, args, device)
    if method == "dshield":
        from .dshield import train
        return train(data, args, device)
    working = copy(data)
    processing_seconds = 0.0
    if method == "abp":
        begin = start_clock(device)
        working["edges"], working["weights"] = filter_edges(data["edges"], data["weights"], data["x"].size(0), data["train"])
        processing_seconds = elapsed(begin, device)
    model = make_model(data, args, device, "rigbd")
    begin = start_clock(device)
    history = fit_classifier(model, working, args.epochs, args.lr, args.weight_decay,
                             ("ABP" if method == "abp" else "无防御") + "分类训练")
    details = {"family": "rigbd", "final_model": describe(model), "checkpoint_rule": "best_validation",
               "phases": {"graph_processing": processing_seconds, "classification": elapsed(begin, device)},
               "history": history}
    return model, details


def run_method(bundle, output, method, args):
    output = Path(output)
    record_path = output / "result.json"
    saved_arguments = arguments(args)
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if args.resume and previous["status"] == "complete":
            expected = previous["arguments"].copy()
            expected["resume"] = saved_arguments["resume"]
            if expected != saved_arguments:
                raise ValueError("已完成运行的参数不同，请选择新目录。")
            if file_hash(output / "model.pt") != previous["checkpoint_sha256"]:
                raise ValueError("已保存模型发生变化。")
            tqdm.write("跳过已完成且参数一致的运行：" + str(output))
            return previous
        raise FileExistsError("运行已存在；继续相同实验请使用 --resume：" + str(output))
    output.mkdir(parents=True, exist_ok=False)
    device = setup(args.seed, args.device)
    data = load_training(bundle, device)
    bundle_info = json.loads((Path(bundle) / "metadata.json").read_text(encoding="utf-8"))
    if bundle_info["settings"]["dataset"] != args.dataset or bundle_info["settings"]["attack"] != args.attack:
        raise ValueError("样本包的数据集或生成方法与本次参数不一致。")
    record = {"status": "running", "method": method, "dataset": args.dataset, "model": args.model,
              "attack": args.attack, "seed": args.seed, "arguments": saved_arguments,
              "bundle_id": bundle_info["bundle_id"], "bundle": str(bundle),
              "code_sha256": code_hashes(),
              "environment": {"torch": torch.__version__, "pyg": torch_geometric.__version__,
                              "python": sys.version, "platform": platform.platform(),
                              "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    save_json(record_path, record)
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        begin = start_clock(device)
        with logging_redirect_tqdm():
            model, details = fit_method(method, data, args, device)
        build_seconds = elapsed(begin, device)
        torch.save({"state_dict": cpu_state(model), "family": details["family"], "method": method,
                    "arguments": saved_arguments, "bundle_id": bundle_info["bundle_id"]}, output / "model.pt")
        # 至此训练已完成，下面才打开包含测试标签的文件。
        metrics = evaluate(model, data, bundle, method, details, device)
        metrics["model_build_seconds"] = build_seconds
        metrics["peak_cuda_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        record.update(status="complete", metrics=metrics, details=details,
                      checkpoint_sha256=file_hash(output / "model.pt"),
                      timing="数据已驻设备；含模型创建、必要准备和最终训练；GPU同步；测试和保存单列")
        save_json(record_path, record)
        asr_text = "无触发测试"
        if metrics["ASR"] is not None:
            asr_text = str(round(metrics["ASR"] * 100, 2)) + "%"
        tqdm.write(method + " 完成：CA=" + str(round(metrics["CA"] * 100, 2)) + "%  ASR=" +
                   asr_text + "  建立模型=" + str(round(build_seconds, 3)) + "秒")
        return record
    except Exception as error:
        record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        save_json(record_path, record)
        raise


def evaluate_checkpoint(bundle, checkpoint, output, device_name):
    device = setup(10, device_name)
    saved = torch.load(checkpoint, map_location=device)
    args = SimpleNamespace(**saved["arguments"])
    data = load_training(bundle, device)
    info = json.loads((Path(bundle) / "metadata.json").read_text(encoding="utf-8"))
    if info["bundle_id"] != saved["bundle_id"]:
        raise ValueError("模型与样本包不匹配。")
    model = make_model(data, args, device, saved["family"])
    model.load_state_dict(saved["state_dict"])
    metrics = evaluate(model, data, bundle, saved["method"], {}, device)
    save_json(output, metrics)
    return metrics
