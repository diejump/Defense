"""直接调用DShield作者适配，不改变其GCN/GAT/GraphSage内部关系。"""
import torch
from upstream.DShield.NodeClassificationTasks.defense.dshield import dshield
from .models import make_model, describe
from .training import fit_classifier
from .common import start_clock, elapsed


def train(data, args, device):
    begin = start_clock(device)
    probe = make_model(data, args, device, "dshield")
    probe_description = describe(probe)
    fit_classifier(probe, data, args.epochs, args.lr, args.weight_decay,
                   "DShield准备模型（" + args.model + "）")
    prepare_seconds = elapsed(begin, device)
    # 作者接口对全图标签做属性梯度分析；未知位置由准备模型预测补齐。
    # 训练和验证标签照常保留，测试真值始终留在evaluation.pt中。
    with torch.no_grad():
        output = probe(data["x"], data["edges"], data["weights"])
        labels = output[:data["labels"].numel()].argmax(-1)
    known = data["labels"] >= 0
    labels[known] = data["labels"][known]
    arch = args.model
    if arch == "GraphSAGE":
        arch = "GraphSage"
    begin = start_clock(device)
    model, selected = dshield(probe, data["x"], data["edges"], data["weights"], labels,
        data["train"], torch.empty(0, dtype=torch.long, device=device), data["val"], args.hidden,
        lr=args.lr, weight_decay=args.weight_decay, kappa1=args.kappa1, kappa2=args.kappa2,
        kappa3=args.kappa3, thresh=args.dshield_threshold, edge_drop_ratio=args.dshield_edge_drop,
        feature_drop_ratio=args.dshield_feature_drop, tau=args.dshield_temperature,
        pretrain_epochs=args.dshield_pretrain, classify_epochs=args.epochs,
        finetune_epochs=args.dshield_finetune, device=device, balance_factor=args.dshield_balance,
        classify_rounds=args.dshield_rounds, dataset=args.dataset, arch=arch,
        attack_name=args.attack, rtn_node_idx=True)
    model.eval()
    details = {"family": "dshield", "scoring_model": probe_description, "final_model": describe(model),
               "internal_arch": arch, "selected_nodes": selected.cpu().tolist(),
               "unknown_labels": "补齐原节点未知标签；长度保持作者原节点数；新增节点没有标签槽位",
               "phases": {"preparation": prepare_seconds, "dshield": elapsed(begin, device)}}
    return model, details
