"""独立进程内按作者入口执行、保存模型，并支持原流程重载评估。"""
import importlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch_geometric
from tqdm.auto import tqdm
from .common import setup, save_json, arguments, file_hash, start_clock, elapsed
from .author_paths import activate
from .author_inputs import load_author_inputs
from .author_parameters import get_parameters
from .author_support import pack_model, unpack_model, plain_values, capture_evaluation, restore_evaluation


def method_module(method, attack):
    if method in ["none", "abp"]:
        return importlib.import_module("src.evaluation")
    name = method
    if method in ["losplit", "mad"]:
        name += "_" + attack.lower()
    return importlib.import_module("src.author_methods." + name)


def fit_own(inputs, args, device):
    from .models import make_model
    from .abp import filter_edges
    from .evaluation import evaluate
    data = inputs["training"].copy()
    begin = start_clock(device)
    if args.method == "abp":
        data["edges"], data["weights"] = filter_edges(data["edges"], data["weights"], data["x"].size(0), data["train"])
    model = make_model(inputs, args, device)
    # 与作者普通分类器相同：保留原fit的初始化与模型选择。
    model.fit(data["x"], data["edges"], data["weights"], data["labels"],
              data["train"], data["val"], train_iters=args.epochs, verbose=False)
    seconds = elapsed(begin, device)
    frame = capture_evaluation(model, device)
    metrics = evaluate(inputs, args, device, model)
    return {"models": [model], "source_metrics": metrics, "evaluation_frames": [frame],
            "defended_edges": inputs["poison_edge_index"], "defended_weights": inputs["poison_edge_weights"],
            "auxiliary_models": {}, "source_args": arguments(args), "internal_seeds": [args.seed],
            "author_training_nodes": data["train"].cpu().tolist(),
            "author_attached_nodes": inputs["idx_attach"].cpu().tolist(),
            "training_api": "RIGBD原模型fit：原初始化和原检查点选择",
            "reference_seconds": 0.0, "model_build_seconds": seconds,
            "prediction_seconds": metrics["prediction_seconds"]}


def check_metrics(metrics):
    for name in ["CA", "ASR"]:
        number = metrics[name]
        if number is not None and (not math.isfinite(float(number)) or not 0 <= float(number) <= 1):
            raise ValueError("作者流程返回无效指标：" + name + "=" + str(number))


def execute(request):
    config = SimpleNamespace(**request["arguments"])
    config.method = request["method"]
    output = Path(request["output"])
    device = setup(config.seed, config.device)
    activate(config.method, config.attack)
    inputs = load_author_inputs(request["bundle"], request["cache"], device)
    record_path = output / "result.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["environment"] = {"torch": torch.__version__, "pyg": torch_geometric.__version__,
        "python": sys.version, "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    save_json(record_path, record)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        parameters = config
        parameter_source = "本项目无防御与ABP的公共分类参数"
        if config.method not in ["none", "abp"]:
            parameters, parameter_source = get_parameters(config.method, config, inputs, output)
            parameters.attack = config.attack
        save_json(output / "effective_parameters.json", plain_values(vars(parameters)))
        module = method_module(config.method, config.attack)
        begin = start_clock(device)
        if config.method in ["none", "abp"]:
            result = fit_own(inputs, config, device)
        else:
            result = module.run(inputs, parameters, device)
        execution_seconds = elapsed(begin, device)
        metrics = plain_values(result["source_metrics"])
        check_metrics(metrics)
        metrics["model_build_seconds"] = result["model_build_seconds"]
        metrics["reference_seconds"] = result["reference_seconds"]
        metrics["prediction_seconds"] = result["prediction_seconds"]
        metrics["author_execution_seconds"] = execution_seconds
        metrics["internal_model_count"] = len(result["models"])
        # 检查点保存放在计时之外，MAD保留原五个最终模型和两个辅助模型。
        models = []
        for model in result["models"]:
            models.append(pack_model(model, vars(parameters)))
        auxiliaries = {}
        for name, model in result["auxiliary_models"].items():
            auxiliaries[name] = pack_model(model, vars(parameters))
        saved = {"protocol": "author_original_20261111", "method": config.method,
            "arguments": arguments(config), "source_args": vars(parameters).copy(),
            "bundle_id": inputs["metadata"]["bundle_id"], "models": models, "auxiliary_models": auxiliaries,
            "evaluation_frames": result["evaluation_frames"], "internal_seeds": result["internal_seeds"],
            "defended_edges": result["defended_edges"].detach().cpu(),
            "defended_weights": result["defended_weights"].detach().cpu(),
            "entropy_threshold": plain_values(result.get("entropy_threshold")),
            "cache": request["cache"]}
        for model in models + list(auxiliaries.values()):
            for name, value in model["state_dict"].items():
                if not torch.isfinite(value).all():
                    raise ValueError("作者模型含非有限权重：" + name)
        torch.save(saved, output / "model.pt")
        excluded = ["models", "auxiliary_models", "defended_edges", "defended_weights", "evaluation_frames"]
        details = {}
        for name, value in result.items():
            if name not in excluded:
                details[name] = plain_values(value)
        details["final_models"] = [m["signature"] for m in models]
        details["auxiliary_models"] = {name: m["signature"] for name, m in auxiliaries.items()}
        details["parameter_source"] = parameter_source
        details["label_condition"] = "作者全部原节点标签监督两个辅助模型，用户已确认" if config.method == "mad" else "作者原入口标签与索引；无防御/ABP只监督训练节点"
        details["target_condition"] = "作者原入口使用已知目标类别" if config.method in ["rigbd", "losplit", "prune"] else "按所存原入口参数执行"
        metrics["peak_cuda_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        record.update(status="complete", metrics=metrics, details=details,
                      checkpoint_sha256=file_hash(output / "model.pt"))
        save_json(record_path, record)
        tqdm.write(config.method + " 完成：CA=" + format(metrics["CA"] * 100, ".2f") + "%  ASR=" + str(metrics["ASR"]))
    except Exception as error:
        record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        save_json(record_path, record)
        raise


def aggregate_evaluations(results, method, attack):
    if method != "mad":
        return plain_values(results[0])
    metrics = results[0].copy()
    # 保留作者的聚合次序：GTA用np.average，另外两分支按原Tensor累加。
    for name in ["CA", "ASR"]:
        if attack == "GTA":
            metrics[name] = np.average([float(row[name]) for row in results])
        else:
            total = 0
            for row in results:
                total += row[name]
            metrics[name] = total / len(results)
    metrics["per_model"] = plain_values(results)
    return plain_values(metrics)


def reevaluate(request):
    saved = torch.load(request["checkpoint"], map_location="cpu")
    if saved.get("protocol") != "author_original_20261111":
        raise ValueError("此入口只重载当前作者流程检查点，旧实验保持原记录。")
    config = SimpleNamespace(**saved["arguments"])
    device = setup(config.seed, request["device"])
    activate(saved["method"], config.attack)
    inputs = load_author_inputs(request["bundle"], request["cache"], device)
    if inputs["metadata"]["bundle_id"] != saved["bundle_id"]:
        raise ValueError("检查点与固定样本不匹配。")
    inputs["poison_edge_index"] = saved["defended_edges"].to(device)
    inputs["poison_edge_weights"] = saved["defended_weights"].to(device)
    parameters = SimpleNamespace(**saved["source_args"])
    module = method_module(saved["method"], config.attack)
    auxiliary = {}
    for name, state in saved["auxiliary_models"].items():
        auxiliary[name] = unpack_model(state, device)
    values = []
    for state, frame in tqdm(zip(saved["models"], saved["evaluation_frames"]), total=len(saved["models"]), desc="重载原生模型", unit="个"):
        model = unpack_model(state, device)
        restore_evaluation(model, frame, device)
        values.append(module.evaluate(inputs, parameters, device, model, auxiliary, saved["entropy_threshold"]))
    metrics = aggregate_evaluations(values, saved["method"], config.attack)
    check_metrics(metrics)
    save_json(request["output"], metrics)
