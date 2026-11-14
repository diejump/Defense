"""把固定样本转为作者入口变量；触发数据复用原生成器的输出。"""
from pathlib import Path
import datetime
import json
import numpy as np
import torch
from torch_geometric.data import Data


def attack_test_indices(count, split_seed):
    order = np.random.RandomState(split_seed).permutation(count)
    start = int(count * 0.2) + int(count * 0.1)
    test = sorted(order[start:start + int(count * 0.2)])
    return torch.tensor(test[len(test) // 2:], dtype=torch.long)


class PacketTrigger:
    """仅移动已冻结触发子图的编号，不重新实现生成算法。"""
    def __init__(self, cache, device):
        self.groups = {}
        self.node_ids = None
        features, edges, weights = cache["x"].to(device), cache["edges"].to(device), cache["weights"].to(device)
        start, size = cache["base_nodes"], cache["trigger_size"]
        # 只扫描新增边一次，避免在大图上为每个测试节点重新扫描整图。
        added = (edges[0] >= start) | (edges[1] >= start)
        trigger_edges, trigger_weights = edges[:, added], weights[added]
        group = (trigger_edges.max(dim=0).values - start) // size
        order = torch.argsort(group, stable=True)
        trigger_edges, trigger_weights = trigger_edges[:, order], trigger_weights[order]
        counts = torch.bincount(group, minlength=cache["indices"].numel()).cpu().tolist()
        offset = 0
        for index, target in enumerate(cache["indices"].tolist()):
            first = start + index * size
            last = first + size
            end = offset + counts[index]
            local = trigger_edges[:, offset:end].clone()
            local[local == target] = -1
            local[local >= first] -= first
            if bool(((local < -1) | (local >= size)).any()):
                raise ValueError("缓存中存在跨触发组连接，无法按原接口移动编号。")
            self.groups[int(target)] = (features[first:last], local, trigger_weights[offset:end])
            offset = end

    def set_node_ids(self, node_ids):
        self.node_ids = node_ids

    def inject_trigger(self, indices, features, edges, weights, device=None):
        if weights is None:
            weights = features.new_ones(edges.size(1))
        feature_parts = [features]
        edge_parts = [edges]
        weight_parts = [weights]
        next_node = features.size(0)
        for target in indices.flatten().tolist():
            original = target if self.node_ids is None else int(self.node_ids[target])
            if original not in self.groups:
                raise ValueError("固定触发缓存中没有目标节点：" + str(original))
            x, local, w = self.groups[original]
            mapped = local.clone()
            original_marker = mapped == -1
            mapped[~original_marker] += next_node
            mapped[original_marker] = target
            feature_parts.append(x)
            edge_parts.append(mapped)
            weight_parts.append(w)
            next_node += x.size(0)
        return torch.cat(feature_parts), torch.cat(edge_parts, dim=1), torch.cat(weight_parts)


def load_author_inputs(bundle, cache_path, device):
    folder = Path(bundle)
    training = torch.load(folder / "training.pt", map_location=device)
    evaluation = torch.load(folder / "evaluation.pt", map_location=device)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    truth = evaluation["truth"]
    original_count = truth.numel()
    attached = evaluation.get("attached", torch.empty(0, dtype=torch.long, device=device))
    original_train = training["train"][~torch.isin(training["train"], attached)]
    known = torch.cat([original_train, attached])
    labels = truth.clone()
    observed_labels = training["labels"][:original_count]
    observed = observed_labels >= 0
    labels[observed] = observed_labels[observed]
    prefix = training["edges"].size(1)
    if not torch.equal(evaluation["reference_edges"][:, :prefix], training["edges"]):
        raise ValueError("参考测试图的前段与已保存训练边不一致。")
    heldout = evaluation["reference_edges"][:, prefix:]
    original_edges = evaluation["reference_edges"]
    original_edges = original_edges[:, (original_edges[0] < original_count) & (original_edges[1] < original_count)]
    original = Data(x=training["x"][:original_count], y=truth, edge_index=original_edges)
    source = metadata.get("source", {})
    split_seed = metadata.get("settings", {}).get("split_seed", source.get("split_seed", 10))
    indices = attack_test_indices(original_count, split_seed).to(device)
    trigger = None
    if cache_path is not None:
        cache = torch.load(cache_path, map_location="cpu")
        trigger = PacketTrigger(cache, device)
    return {"data": original, "training": training, "evaluation": evaluation, "metadata": metadata,
            "poison_x": training["x"], "poison_edge_index": training["edges"],
            "poison_edge_weights": training["weights"], "poison_labels": labels,
            "idx_train": original_train, "idx_attach": attached, "idx_val": training["val"],
            "known_nodes": known, "bkd_tn_nodes": known, "idx_clean_test": evaluation["clean_nodes"],
            "idx_atk": indices, "mask_edge_index": heldout, "trigger": trigger,
            "current_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}
