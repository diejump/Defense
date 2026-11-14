"""调用DShield仓库中的三种生成算法，保存所有方法共用的固定样本。"""
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .data import load_clean
from .common import setup, save_json, file_hash, start_clock, elapsed
from .poison_budget import validate_budget, resolve_budget
from .class_sampling import select_attached


def bundle_directory(args):
    if args.bundle is not None:
        return args.bundle
    return args.output_root / "data" / args.dataset / args.attack / ("seed" + str(args.seed))


def generation_parameters(args):
    names = ["dataset", "attack", "seed", "split_seed", "rate", "vs_number", "class_sampling",
             "target", "trigger_size", "attack_epochs", "attack_hidden", "attack_inner", "attack_lr",
             "attack_weight_decay", "threshold", "target_weight", "homo_weight", "homo_threshold",
             "gta_loss_factor", "rec_epochs", "detector_updates", "outer_size", "ood_weight", "class_weight"]
    values = {}
    for name in names:
        values[name] = getattr(args, name)
    # 保护旧样本：分母或过滤规则改变时不能复用旧设置的生成结果。
    values["budget_basis"] = "original_train_nodes"
    values["target_filter"] = "after_selection_no_refill"
    return values


def make_generator(args, device, available_count):
    from .author_paths import activate
    activate("dshield", args.attack)
    from attack.ugba import UGBA
    from attack.gta import GTA
    from attack.dpgba import DPGBA
    values = dict(thrd=args.threshold, hidden=args.attack_hidden, trojan_epochs=args.attack_epochs,
                  lr=args.attack_lr, weight_decay=args.attack_weight_decay, trigger_size=args.trigger_size,
                  target_class=args.target, device=device)
    if args.attack == "GTA":
        values["loss_factor"] = args.gta_loss_factor
        model = GTA(**values)
    elif args.attack == "UGBA":
        values["seed"] = args.seed
        values["inner_epochs"] = 5 if args.attack_inner is None else args.attack_inner
        values["target_loss_weight"] = 5.0 if args.target_weight is None else args.target_weight
        values["homo_loss_weight"] = args.homo_weight
        values["homo_boost_thrd"] = args.homo_threshold
        model = UGBA(**values)
    else:
        values["seed"] = args.seed
        values["inner_epochs"] = 1 if args.attack_inner is None else args.attack_inner
        values["rec_epochs"] = args.rec_epochs
        values["k"] = args.detector_updates
        values["outter_size"] = min(args.outer_size, available_count)
        values["target_weight"] = 1.0 if args.target_weight is None else args.target_weight
        values["ood_weight"] = args.ood_weight
        values["target_class_weight"] = args.class_weight
        values["dataset_name"] = args.dataset
        model = DPGBA(**values)
    values["device"] = str(device)
    return model, values


def keep_positive_edges(graph):
    x, edges, weights = graph
    keep = weights > 0
    return x.detach(), edges[:, keep].detach(), weights[keep].detach()


def to_cpu(data):
    result = {}
    for name, value in data.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
        result[name] = value
    return result


def labels_for_generator(data, target):
    # 生成阶段同样只提供训练标签；GTA的目标类样本不能从测试标签筛选。
    labels = torch.full_like(data["labels"], -1)
    labels[data["train"]] = data["labels"][data["train"]]
    if int(labels.max()) + 1 != data["classes"]:
        raise ValueError("训练标签没有覆盖最高类别，当前作者生成器无法确定完整类别数。")
    if not bool((labels[data["train"]] == target).any()):
        raise ValueError("训练集没有目标类样本，请调整目标类别或划分种子。")
    return labels


def generate(args):
    validate_budget(args)
    folder = bundle_directory(args)
    settings = generation_parameters(args)
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError("已有样本的生成参数不同，请设置新的 --output-root 或 --bundle。")
        for name, digest in previous["sha256"].items():
            if file_hash(folder / name) != digest:
                raise ValueError("已保存样本发生变化：" + name)
        tqdm.write("使用已校验的固定样本：" + str(folder))
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("样本目录存在未完成的文件，请选择新的输出目录：" + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    device = setup(args.seed, args.device)
    data, info = load_clean(args, device)
    if not 0 <= args.target < data["classes"]:
        raise ValueError("目标类别超出此数据集的范围。")
    original_train_count = data["train"].numel()
    requested_count = resolve_budget(args, original_train_count, info["pool"].numel())
    attached, available, allocation = select_attached(info["pool"], info["truth"], requested_count,
        data["classes"], args.seed, args.class_sampling, args.target)
    count = attached.numel()
    tqdm.write("原训练节点：" + str(original_train_count) + "，选样预算：" + str(requested_count)
               + "，过滤原目标类：" + str(requested_count - count) + "，实际触发器组数：" + str(count))
    generator, actual_parameters = make_generator(args, device, available.numel())
    attack_labels = labels_for_generator(data, args.target)
    begin = start_clock(device)
    with logging_redirect_tqdm():
        generator.fit(data["x"], data["edges"], data["weights"], attack_labels.clone(),
                      data["train"], attached, available)
    generation_seconds = elapsed(begin, device)
    x, edges, weights, labels = generator.get_poisoned(data["x"], data["edges"], data["weights"],
                                                    attack_labels.clone(), attached)
    x, edges, weights = keep_positive_edges((x, edges, weights))
    if not bool((labels[attached] == args.target).all()):
        raise ValueError("生成器未将全部选中原节点改为指定target。")
    train = torch.cat([data["train"], attached]).sort().values
    # 保留作者标签长度：原节点有标签槽位，新增触发节点没有标签槽位。
    observed = torch.full_like(labels, -1)
    observed[train] = labels[train]
    observed[data["val"]] = info["truth"][data["val"]]
    training = {"x": x, "edges": edges, "weights": weights, "labels": observed,
                "train": train, "val": data["val"], "classes": data["classes"]}
    reference_edges = torch.cat([edges, info["heldout"]], dim=1)
    reference_weights = torch.cat([weights, x.new_ones(info["heldout"].size(1))])
    # 先对全部攻击测试索引生成，再以非目标类节点作为ASR分母。
    changed = generator.inject_trigger(info["attack_test"], x, reference_edges, reference_weights)
    changed_x, changed_edges, changed_weights = keep_positive_edges(changed)
    affected = info["attack_test"][info["truth"][info["attack_test"]] != args.target]
    if affected.numel() == 0:
        raise ValueError("没有可用于ASR计算的非目标类测试节点。")
    evaluation = {"truth": info["truth"], "clean_nodes": info["clean_test"], "attack_nodes": affected,
                  "attached": attached, "target": args.target,
                  "reference_x": x, "reference_edges": reference_edges, "reference_weights": reference_weights,
                  "changed_x": changed_x, "changed_edges": changed_edges, "changed_weights": changed_weights}
    for name, value in tqdm([("training.pt", training), ("evaluation.pt", evaluation)],
                            desc="保存固定训练和评估样本", unit="份"):
        torch.save(to_cpu(value), folder / name)
    state = {}
    for name, value in generator.trojan.state_dict().items():
        state[name] = value.detach().cpu()
    torch.save({"constructor": actual_parameters, "state_dict": state}, folder / "generator.pt")
    hashes = {}
    for name in ["training.pt", "evaluation.pt", "generator.pt"]:
        hashes[name] = file_hash(folder / name)
    metadata = {"settings": settings, "actual_generator_parameters": actual_parameters,
                "generator_seconds": generation_seconds, "source": info["source"],
                "sha256": hashes, "bundle_id": hashes["training.pt"] + ":" + hashes["evaluation.pt"],
                "counts": {"original_nodes": data["x"].size(0), "training_nodes": train.numel(),
                           "original_training_nodes": original_train_count,
                           "selected_before_target_filter": requested_count,
                           "filtered_target_nodes": requested_count - count,
                           "attached": count, "validation": data["val"].numel(),
                           "label_changes": int((info["truth"][attached] != args.target).sum()),
                           "added_training_trigger_nodes": x.size(0) - data["x"].size(0),
                           "attached_fraction_of_training_nodes": count / train.numel(),
                           "clean_test": info["clean_test"].numel(), "ASR_denominator": affected.numel()},
                "generator_source": "upstream/DShield/NodeClassificationTasks/attack/" + args.attack.lower() + ".py"}
    metadata["generator_label_scope"] = "仅原训练节点的标签；其他原节点为-1；生成器自行赋予附着目标标签"
    target_counts = [0] * data["classes"]
    target_counts[args.target] = count
    allocation["poisoned_target_counts"] = target_counts
    allocation["target_class"] = args.target
    metadata["class_allocation"] = allocation
    metadata["budget"] = {"rate_percent": args.rate, "vs_number": args.vs_number,
        "budget_basis": "original_train_nodes", "original_train_nodes": original_train_count,
        "requested_trigger_groups": requested_count, "filtered_target_count": requested_count - count,
        "resolved_trigger_groups": count, "total_original_nodes": data["x"].size(0),
        "target_filter": "after_selection_no_refill",
        "rounding": "floor(N_train*rate/100)", "added_trigger_nodes": count * args.trigger_size}
    save_json(folder / "class_allocation.json", allocation)
    hashes["class_allocation.json"] = file_hash(folder / "class_allocation.json")
    save_json(metadata_path, metadata)
    tqdm.write("固定样本已保存：" + str(folder))
    return folder
