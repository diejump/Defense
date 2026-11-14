"""在原始类别中选附着位置；所有选中位置仍使用同一个目标标签。"""
import numpy as np
import torch


def allocate_uneven_counts(count, capacities, seed):
    capacities = np.asarray(capacities, dtype=np.int64)
    active = np.flatnonzero(capacities > 0)
    if active.size < 2 or count < 2:
        raise ValueError("不均匀多类别选样至少需要两个可用类别和两个附着位置。")
    rng = np.random.RandomState(seed)
    weights = np.zeros(capacities.size, dtype=np.float64)
    weights[active] = rng.dirichlet(np.ones(active.size))
    counts = np.zeros_like(capacities)
    if count >= active.size:
        counts[active] = 1
    remaining = count - int(counts.sum())
    for _ in range(remaining):
        available = counts < capacities
        probability = weights * available
        if probability.sum() <= 0:
            raise ValueError("类别剩余容量不足，无法分配完整预算。")
        probability /= probability.sum()
        index = int(rng.choice(capacities.size, p=probability))
        counts[index] += 1
    # 极小预算或偶然等分时，在容量允许的条件下打破等分。
    if count >= active.size and np.unique(counts[active]).size == 1:
        recipients = active[counts[active] < capacities[active]]
        donors = active[counts[active] > 0]
        adjusted = False
        for recipient in recipients:
            donor_options = donors[donors != recipient]
            if donor_options.size:
                counts[donor_options[0]] -= 1
                counts[recipient] += 1
                adjusted = True
                break
        if not adjusted:
            raise ValueError("当前类别容量与整数预算无法产生不均匀配额，请调整预算。")
    if counts.sum() != count or np.any(counts > capacities):
        raise ValueError("类别数量分配不满足总预算或容量限制。")
    return counts, weights


def select_attached(pool, original_labels, count, classes, seed, mode, target):
    # 保留一次原Torch选样调用，使改变模式不额外改变生成器的Torch随机消耗。
    order = torch.randperm(pool.numel(), device=pool.device)
    capacities = torch.bincount(original_labels[pool], minlength=classes).cpu().numpy()
    weights = None
    if mode == "random":
        attached = pool[order[:count]]
    elif mode == "uneven":
        counts, weights = allocate_uneven_counts(count, capacities, seed)
        rng = np.random.RandomState(seed)
        pool_values = pool.cpu().numpy()
        source_labels = original_labels[pool].cpu().numpy()
        parts = []
        for label in range(classes):
            candidates = pool_values[source_labels == label]
            shuffled = rng.permutation(candidates)
            parts.extend(shuffled[:int(counts[label])].tolist())
        attached = torch.tensor(rng.permutation(parts), dtype=torch.long, device=pool.device)
    else:
        raise ValueError("未知原始类别选样模式：" + str(mode))
    if attached.numel() != count or attached.unique().numel() != count:
        raise ValueError("选样数量错误或存在重复位置。")
    proposed = attached
    before_counts = torch.bincount(original_labels[proposed], minlength=classes).cpu().tolist()
    # 对齐作者入口：先按预算选样，再过滤原目标类；不补选。
    non_target = original_labels[proposed] != target
    filtered = proposed[~non_target]
    attached = proposed[non_target]
    if attached.numel() == 0:
        raise ValueError("过滤原目标类别后没有可用附着节点，请调整预算或seed。")
    # 被过滤的位置仍是普通未标注候选，不能从未标注池中丢掉。
    ordered_pool = pool[order]
    available = ordered_pool[~torch.isin(ordered_pool, attached)]
    selected_counts = torch.bincount(original_labels[attached], minlength=classes).cpu().tolist()
    details = {"mode": mode, "seed": seed, "eligible_source_counts": capacities.tolist(),
               "poisoned_source_counts": selected_counts,
               "source_class_weights": weights.tolist() if weights is not None else None,
               "target_source_class_included": False, "target_class": target,
               "requested_count_before_filter": count, "actual_count_after_filter": attached.numel(),
               "selected_source_counts_before_filter": before_counts,
               "selected_nodes_before_filter": proposed.cpu().tolist(),
               "filtered_target_nodes": filtered.cpu().tolist(), "filtered_target_count": filtered.numel(),
               "target_filter": "先选后滤，不补选，且在训练生成器前执行",
               "label_information": "仅生成阶段按原始标签分层选位置；防御不读取该分层名单",
               "algorithm": "全候选均匀无放回随机选样" if mode == "random" else
                   "seed生成Dirichlet(1)权重，预算允许时每个可用类别先分配1个，其余按权重及剩余容量逐个分配；若恰好完全等分则调整1个配额，再在类内无放回选样"}
    return attached, available, details
