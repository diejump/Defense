"""RIGBD评分与恢复；三个骨架均使用对应作者模型。"""
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .common import setup, start_clock, elapsed
from .models import make_model, describe
from .training import fit_classifier
from upstream.RIGBD.sampling import sample_noise_all


@torch.no_grad()
def score_nodes(model, data, repeats):
    model.eval()
    reference = F.log_softmax(model(data["x"], data["edges"], data["weights"]), dim=-1)
    probabilities = reference.exp()
    scores = torch.zeros(data["x"].size(0), device=data["x"].device)
    for _ in tqdm(range(repeats), desc="RIGBD随机丢边评分", unit="次", leave=False):
        edges, weights = sample_noise_all(data["edges"], data["weights"], data["x"].device)
        changed = F.log_softmax(model(data["x"], edges, weights), dim=-1)
        scores += (probabilities * (reference - changed)).sum(dim=-1)
    return scores / repeats


def select_nodes(scores, data):
    # 按论文的自动阈值规则推断目标类，不读取生成目标或真实附着名单。
    order = data["train"][torch.argsort(scores[data["train"]], descending=True)]
    target = int(data["labels"][order[0]])
    labels = data["labels"][order]
    stop = None
    for index in range(order.numel() - 1):
        if int(labels[index]) != target and int(labels[index + 1]) != target:
            stop = index
            break
    selected = order
    if stop is not None:
        selected = order[scores[order] > scores[order[stop]]]
    return selected, target


def recover(model, data, selected, target, args):
    # 用节点值做差集，修正作者Tensor集合不能可靠做差的问题。
    normal = data["train"][~torch.isin(data["train"], selected)]
    if normal.numel() == 0:
        raise RuntimeError("RIGBD没有保留普通训练节点，请检查评分与数据。")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in tqdm(range(args.epochs), desc="RIGBD最终恢复训练", unit="轮", leave=False):
        model.train()
        optimizer.zero_grad()
        output = model(data["x"], data["edges"], data["weights"])
        if args.rigbd_loss == "source":
            normal_loss = F.cross_entropy(output[normal], data["labels"][normal])
            probabilities = F.softmax(output[selected], dim=-1)
            penalty = probabilities.gather(1, data["labels"][selected, None]).mean()
            loss = normal_loss + penalty
        else:
            normal_loss = F.cross_entropy(output[normal], data["labels"][normal], reduction="sum")
            log_target = F.log_softmax(output[selected], dim=-1)[:, target]
            log_target = log_target.clamp_min(math.log(1e-8))
            loss = (normal_loss + log_target.sum()) / data["train"].numel()
        if not torch.isfinite(loss):
            raise RuntimeError("RIGBD恢复训练出现非有限损失。")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": loss.item()})
    model.eval()
    return history


def train(data, args, device):
    begin = start_clock(device)
    probe = make_model(data, args, device, "rigbd", scoring=True)
    probe_description = describe(probe)
    fit_classifier(probe, data, args.rigbd_probe_epochs, args.lr, args.weight_decay,
                   "RIGBD评分模型（" + args.model + "）")
    prepare_seconds = elapsed(begin, device)
    begin = start_clock(device)
    scores = score_nodes(probe, data, args.rigbd_repeats)
    selected, target = select_nodes(scores, data)
    score_seconds = elapsed(begin, device)
    del probe
    setup(args.seed, str(device))
    model = make_model(data, args, device, "rigbd")
    begin = start_clock(device)
    if selected.numel() == 0:
        history = fit_classifier(model, data, args.epochs, args.lr, args.weight_decay, "RIGBD空候选普通训练")
    else:
        history = recover(model, data, selected, target, args)
    details = {"family": "rigbd", "scoring_model": probe_description, "final_model": describe(model),
               "selected_nodes": selected.cpu().tolist(), "inferred_target": target,
               "loss": args.rigbd_loss, "checkpoint_rule": "last" if selected.numel() else "best_validation",
               "phases": {"preparation": prepare_seconds, "score_and_selection": score_seconds,
                          "final_training": elapsed(begin, device)}, "history": history}
    return model, details
