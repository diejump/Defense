"""作者默认参数、公开命令参数和显式覆盖，按普通字典读取。"""
import json
from pathlib import Path
from types import SimpleNamespace
from .author_paths import ROOT


def get_parameters(method, config, inputs, output):
    book = json.loads((ROOT / "configs/author_parameters.json").read_text(encoding="utf-8"))
    item = book["methods"][method]
    attack = config.attack
    if "defaults_by_attack" in item:
        values = item["defaults_by_attack"][attack].copy()
    else:
        values = item["defaults"].copy()
    key = config.dataset + "/" + attack
    values.update(item.get("by_case", {}).get(key, {}))
    source = item.get("case_sources", {}).get(key, "原入口共享默认，未声明该组合最优")
    if config.author_overrides is not None:
        overrides = json.loads(Path(config.author_overrides).read_text(encoding="utf-8"))
        values.update(overrides.get(method, {}))
        source += "；用户显式覆盖见运行参数"
    model = "GraphSage" if config.model == "GraphSAGE" else config.model
    values["dataset"] = config.dataset
    values["seed"] = config.seed
    values["test_model"] = model
    values["target_class"] = inputs["metadata"]["settings"]["target"]
    values["vs_number"] = inputs["idx_attach"].numel()
    values["pre_train_param"] = str(Path(output) / "author_split.pt")
    if method == "rigbd":
        if config.rigbd_repeats is not None:
            values["K"] = config.rigbd_repeats
        if config.rigbd_drop_ratio is not None:
            values["drop_ratio"] = config.rigbd_drop_ratio
    if method == "dshield":
        values["model"] = model
        values["defense_method"] = "DShield"
        values["attack_method"] = attack
    if method == "mad":
        values["defense_mode"] = "MAD"
        values["bkd_model"] = attack
    if method == "prune":
        values["defense_mode"] = "prune"
        if config.prune_threshold is not None:
            values["prune_thr"] = config.prune_threshold
    values["device_id"] = int(config.device.split(":")[-1]) if ":" in config.device else 0
    return SimpleNamespace(**values), source
