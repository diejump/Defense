"""LoSplit：固定GCN分组，所选骨架完成恢复。"""
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from upstream.LoSplit.models.LoSplit import LoSplit
from .common import setup, start_clock, elapsed
from .models import make_model, describe


def parameters(args):
    # 作者defense.sh中的早期轮数、学习率和gamma。
    table = {
        "Cora": {"UGBA": (15, 0.01, 0.2), "GTA": (10, 0.01, 0.3), "DPGBA": (15, 0.01, 1.0)},
        "Citeseer": {"UGBA": (10, 0.01, 0.45), "GTA": (10, 0.05, 0.45), "DPGBA": (10, 0.01, 0.85)},
        "Pubmed": {"UGBA": (15, 0.03, 0.0), "GTA": (15, 0.005, 0.55), "DPGBA": (40, 0.01, 1.0)}}
    if args.dataset in table:
        epochs, lr, gamma = table[args.dataset][args.attack]
        origin = "作者defense.sh"
    else:
        epochs, lr, gamma = 15, 0.01, 0.2
        origin = "Physics/Flickr预设待验证配置；可用显式参数覆盖"
    if args.losplit_epochs is not None:
        epochs = args.losplit_epochs
    if args.losplit_lr is not None:
        lr = args.losplit_lr
    if args.losplit_gamma is not None:
        gamma = args.losplit_gamma
    return epochs, lr, gamma, origin


def recover(model, data, normal, selected, target, gamma, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    choices = []
    for label in range(data["classes"]):
        if label != target:
            choices.append(label)
    choices = torch.tensor(choices, device=data["x"].device)
    positions = torch.randint(choices.numel(), (selected.numel(),), device=choices.device)
    replacements = choices[positions]
    history = []
    for epoch in tqdm(range(args.epochs), desc="LoSplit最终恢复训练", unit="轮", leave=False):
        model.train()
        optimizer.zero_grad()
        output = model(data["x"], data["edges"], data["weights"])
        if args.losplit_loss == "paper":
            positions = torch.randint(choices.numel(), (selected.numel(),), device=choices.device)
            replacements = choices[positions]
        normal_loss = F.cross_entropy(output[normal], data["labels"][normal], reduction="sum")
        relabel_loss = F.cross_entropy(output[selected], replacements, reduction="sum")
        loss = normal_loss + gamma * relabel_loss
        if args.losplit_loss == "paper":
            targets = torch.full_like(selected, target)
            forget_loss = F.cross_entropy(output[selected], targets, reduction="sum")
            loss -= (1 - gamma) * forget_loss
        # source模式等价于作者有限数值下的目标：relu(-NLL)=0，替代标签只抽一次。
        loss = loss / data["train"].numel()
        if not torch.isfinite(loss):
            raise RuntimeError("LoSplit恢复训练出现非有限损失。")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": loss.item()})
    model.eval()
    return history


def train(data, args, device):
    epochs, lr, gamma, origin = parameters(args)
    begin = start_clock(device)
    # 此处不根据args.model改动：三个最终骨架共用作者GCN划分器。
    splitter = LoSplit(args, nfeat=data["x"].size(1), nhid=args.hidden, nclass=data["classes"],
                       dropout=args.dropout, lr=lr, weight_decay=args.weight_decay, device=device)
    splitter = splitter.to(device)
    scoring_description = describe(splitter)
    bad, good, gaps, targets = splitter.fit(data["x"], data["edges"], data["weights"],
        data["labels"].clone(), data["train"], data["val"], split_epoch=epochs, num_attach=0)
    best = None
    for index in range(len(gaps)):
        if math.isfinite(float(gaps[index])) and len(bad[index]) > 0 and len(good[index]) > 0:
            if best is None or float(gaps[index]) > float(gaps[best]):
                best = index
    if best is None:
        raise RuntimeError("LoSplit没有得到有效的两组划分。")
    selected, normal = bad[best], good[best]
    target = int(targets[best])
    split_seconds = elapsed(begin, device)
    del splitter
    setup(args.seed, str(device))
    model = make_model(data, args, device, "losplit")
    begin = start_clock(device)
    history = recover(model, data, normal, selected, target, gamma, args)
    details = {"family": "losplit", "scoring_model": scoring_description, "final_model": describe(model),
               "selected_nodes": selected.cpu().tolist(), "inferred_target": target,
               "loss": args.losplit_loss, "checkpoint_rule": "last", "selected_early_epoch": best + 1,
               "early_parameters": {"epochs": epochs, "lr": lr, "gamma": gamma, "origin": origin},
               "phases": {"early_training_and_selection": split_seconds,
                          "final_training": elapsed(begin, device)}, "history": history}
    return model, details
