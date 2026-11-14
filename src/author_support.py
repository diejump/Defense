"""作者代码外层的计时、返回格式和状态保存接口。"""
import inspect
import importlib
import random
from types import SimpleNamespace
import numpy as np
import torch
from .common import start_clock, elapsed, cpu_state


def logits_of(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def native_signature(model):
    layers = []
    for module in model.modules():
        if module.__class__.__name__ in ["GCNConv", "GATConv", "SAGEConv"]:
            item = {"type": module.__class__.__name__}
            for name in ["add_self_loops", "heads", "root_weight", "flow"]:
                if hasattr(module, name):
                    item[name] = getattr(module, name)
            layers.append(item)
    return {"class": model.__class__.__name__, "file": inspect.getfile(model.__class__),
            "parameters": sum(p.numel() for p in model.parameters()), "layers": layers}


def model_state(model):
    return {"state_dict": cpu_state(model), "signature": native_signature(model)}


def plain_values(value):
    # 将作者代码的NumPy标量和设备值转为普通JSON值。
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_values(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def capture_evaluation(model, device):
    # 原实现可能在train模式下预测；保存模式和随机数，重载才能重现该次评估。
    return {"training": model.training, "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


def restore_evaluation(model, frame, device):
    model.train(frame["training"])
    torch.set_rng_state(frame["cpu_rng"].cpu())
    if device.type == "cuda" and frame["cuda_rng"] is not None:
        torch.cuda.set_rng_state(frame["cuda_rng"].cpu(), device)
    np.random.set_state(frame["numpy_rng"])
    random.setstate(frame["python_rng"])


def pack_model(model, source_args):
    # 仅保存构造参数和权重；不序列化训练图、优化器或整个作者模型。
    kwargs = {}
    signature = inspect.signature(type(model).__init__)
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name == "device":
            kwargs[name] = "cpu"
        elif name == "args":
            kwargs[name] = source_args.copy()
        elif hasattr(model, name):
            kwargs[name] = plain_values(getattr(model, name))
        elif name in ["nhid", "n_hid"]:
            kwargs[name] = model.hidden_sizes[0]
        elif name == "layer" and hasattr(model, "body"):
            kwargs[name] = len(model.body.convs)
        elif name == "layer" and hasattr(model, "convs"):
            kwargs[name] = len(model.convs) + 1
        elif name == "heads":
            kwargs[name] = model.gc1.heads
        elif name == "arch":
            kwargs[name] = source_args["model"]
        elif parameter.default == inspect.Parameter.empty:
            raise ValueError("缺少作者模型构造参数：" + type(model).__name__ + "." + name)
    return {"module": type(model).__module__, "class": type(model).__name__,
            "constructor": kwargs, "state_dict": cpu_state(model), "signature": native_signature(model),
            "training": model.training}


def unpack_model(saved, device):
    module = importlib.import_module(saved["module"])
    model_class = getattr(module, saved["class"])
    kwargs = saved["constructor"].copy()
    if "device" in kwargs:
        kwargs["device"] = device
    if "args" in kwargs:
        kwargs["args"] = SimpleNamespace(**kwargs["args"])
    model = model_class(**kwargs).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.train(saved["training"])
    return model
