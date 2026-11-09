"""明确选择作者模型，保留各作者对同名骨架的实现。"""
from upstream.RIGBD.models.GCN import GCN as RIGBDGCN
from upstream.RIGBD.models.GAT import GAT as RIGBDGAT
from upstream.RIGBD.models.SAGE import GraphSage as RIGBDSAGE
from upstream.LoSplit.models.GCN import GCN as LoSplitGCN
from upstream.LoSplit.models.GAT import GAT as LoSplitGAT
from upstream.LoSplit.models.SAGE import GraphSage as LoSplitSAGE
from upstream.DShield.NodeClassificationTasks.models.GCN import GCN as DShieldGCN
from upstream.DShield.NodeClassificationTasks.models.GAT import GAT as DShieldGAT
from upstream.DShield.NodeClassificationTasks.models.SAGE import GraphSage as DShieldSAGE


def make_model(data, args, device, family="rigbd", scoring=False):
    if family == "dshield":
        values = dict(n_feat=data["x"].size(1), n_hid=args.hidden, n_class=data["classes"],
                      dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, device=device)
        if args.model == "GCN":
            model = DShieldGCN(**values)
        elif args.model == "GAT":
            model = DShieldGAT(**values)
        else:
            model = DShieldSAGE(**values)
    else:
        values = dict(nfeat=data["x"].size(1), nhid=args.hidden, nclass=data["classes"],
                      dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay, device=device)
        if family == "losplit":
            if args.model == "GCN":
                model = LoSplitGCN(args, **values)
            elif args.model == "GAT":
                model = LoSplitGAT(**values)
            else:
                model = LoSplitSAGE(**values)
        elif args.model == "GCN":
            # 作者construct的add_self_loops开关只传给GCN；评分时关闭。
            model = RIGBDGCN(add_self_loops=not scoring, **values)
        elif args.model == "GAT":
            model = RIGBDGAT(**values)
        else:
            model = RIGBDSAGE(**values)
    return model.to(device)


def describe(model):
    layers = []
    for module in model.modules():
        name = module.__class__.__name__
        if name in ["GCNConv", "GATConv", "SAGEConv"]:
            item = {"type": name}
            if hasattr(module, "add_self_loops"):
                item["add_self_loops"] = module.add_self_loops
            if hasattr(module, "heads"):
                item["heads"] = module.heads
            layers.append(item)
    return {"class": model.__class__.__module__ + "." + model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()), "layers": layers}
