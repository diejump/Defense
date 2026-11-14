"""用保存的生成器权重调用原注入接口，冻结含零权边的作者测试输入。"""
from pathlib import Path
import hashlib
import json
import torch
from .author_paths import ROOT, activate
from .author_inputs import attack_test_indices
from .common import file_hash, save_json


def prepare_cache(bundle):
    bundle = Path(bundle).resolve()
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    key = hashlib.sha256(metadata["bundle_id"].encode()).hexdigest()[:20]
    folder = ROOT / "outputs/author_inputs" / key
    destination = folder / "raw_trigger_input.pt"
    note_path = folder / "source.json"
    if destination.exists() and note_path.exists():
        note = json.loads(note_path.read_text(encoding="utf-8"))
        if note["bundle_id"] != metadata["bundle_id"] or note["generator_sha256"] != file_hash(bundle / "generator.pt"):
            raise ValueError("作者输入缓存与样本生成器不匹配。")
        return destination
    training = torch.load(bundle / "training.pt", map_location="cpu")
    evaluation = torch.load(bundle / "evaluation.pt", map_location="cpu")
    saved = torch.load(bundle / "generator.pt", map_location="cpu")
    values = saved["constructor"].copy()
    values["device"] = torch.device("cpu")
    attack = metadata["settings"]["attack"]
    activate("dshield", attack)
    if attack == "UGBA":
        from attack.ugba import UGBA, GraphTrojanNet
        generator = UGBA(**values)
        generator.trojan = GraphTrojanNet(values["device"], training["x"].size(1), values["trigger_size"], layer_num=2)
    elif attack == "GTA":
        from attack.gta import GTA, GraphTrojanNet
        generator = GTA(**values)
        generator.trojan = GraphTrojanNet(values["device"], training["x"].size(1), values["trigger_size"], layer_num=2)
    elif attack == "DPGBA":
        from attack.dpgba import DPGBA, GraphTrojanNet
        generator = DPGBA(**values)
        generator.trojan = GraphTrojanNet(values["device"], training["x"].size(1), values["trigger_size"], layernum=2)
    else:
        raise ValueError("当前作者输入缓存支持GTA、UGBA、DPGBA。")
    generator.trojan.load_state_dict(saved["state_dict"])
    generator.trojan.eval()
    count = evaluation["truth"].numel()
    settings = metadata["settings"]
    seed = settings.get("split_seed", metadata.get("source", {}).get("split_seed", 10))
    indices = attack_test_indices(count, seed)
    with torch.no_grad():
        x, edges, weights = generator.inject_trigger(indices, training["x"], evaluation["reference_edges"], evaluation["reference_weights"])
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x.detach(), "edges": edges.detach(), "weights": weights.detach(), "indices": indices,
                "base_nodes": training["x"].size(0), "trigger_size": values["trigger_size"]}, destination)
    save_json(note_path, {"bundle_id": metadata["bundle_id"], "generator_sha256": file_hash(bundle / "generator.pt"),
        "source": "upstream/DShield/NodeClassificationTasks/attack/" + attack.lower() + ".py::inject_trigger",
        "zero_weight_edges_retained": int((weights == 0).sum()), "training_executed": False})
    return destination
