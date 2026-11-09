"""公开图数据及固定的20%训练、10%验证、20%测试划分。"""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected, remove_self_loops
from tqdm.auto import tqdm
from .common import file_hash


def load_clean(args, device):
    tqdm.write("读取公开数据集：" + args.dataset)
    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(str(args.data_root), args.dataset, transform=NormalizeFeatures())
    elif args.dataset == "Physics":
        dataset = Coauthor(str(args.data_root), "Physics", transform=NormalizeFeatures())
    elif args.dataset == "Flickr":
        dataset = Flickr(str(args.data_root / "Flickr"))
    else:
        raise ValueError("没有实现此数据集：" + args.dataset)
    graph = dataset[0].to(device)
    normalization = "逐行NormalizeFeatures"
    if args.dataset == "Flickr":
        # 与DShield的ColumnNormalizeFeatures相同，逐特征做min-max。
        graph.x = graph.x - graph.x.min(dim=0).values
        graph.x = graph.x / (graph.x.max(dim=0).values + 1e-6)
        normalization = "逐列min-max，分母加1e-6"
    edges, _ = remove_self_loops(graph.edge_index)
    edges = to_undirected(edges, num_nodes=graph.num_nodes)
    count = graph.num_nodes
    # 三篇作者代码使用固定随机划分；这里把种子作为显式参数。
    order = np.random.RandomState(args.split_seed).permutation(count)
    train_count = int(count * 0.2)
    val_count = int(count * 0.1)
    test_count = int(count * 0.2)
    train = torch.tensor(sorted(order[:train_count]), device=device)
    val = torch.tensor(sorted(order[train_count:train_count + val_count]), device=device)
    test = torch.tensor(sorted(order[train_count + val_count:train_count + val_count + test_count]), device=device)
    clean_test = test[:test.numel() // 2]
    attack_test = test[test.numel() // 2:]
    unavailable = torch.zeros(count, dtype=torch.bool, device=device)
    unavailable[train] = True
    unavailable[val] = True
    unavailable[test] = True
    pool = (~unavailable).nonzero().flatten()
    # 训练图移去测试节点的相邻连接；评估时恢复这些连接。
    is_test = torch.zeros(count, dtype=torch.bool, device=device)
    is_test[test] = True
    keep = ~is_test[edges[0]] & ~is_test[edges[1]]
    labels = torch.full((count,), -1, dtype=torch.long, device=device)
    labels[train] = graph.y[train]
    labels[val] = graph.y[val]
    data = {"x": graph.x, "edges": edges[:, keep], "weights": graph.x.new_ones(int(keep.sum())),
            "labels": labels, "train": train, "val": val, "classes": dataset.num_classes}
    hashes = {}
    raw_names = dataset.raw_file_names
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    for name in raw_names:
        path = Path(dataset.raw_dir) / name
        if path.is_file():
            hashes[name] = file_hash(path)
    info = {"truth": graph.y.clone(), "heldout": edges[:, ~keep], "pool": pool,
            "clean_test": clean_test, "attack_test": attack_test,
            "source": {"loader": dataset.__class__.__name__, "raw_directory": dataset.raw_dir,
                       "raw_sha256": hashes, "feature_normalization": normalization,
                       "split_seed": args.split_seed, "original_nodes": count}}
    return data, info
