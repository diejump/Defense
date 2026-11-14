"""DShield作者main的原始防御与逐节点二跳评估完整段。"""
import copy
import logging
import time
import torch
from torch_geometric.utils import k_hop_subgraph
from models.construct import model_construct
from defense.dshield import dshield, model_test
from attack.sba import SBA
from attack.target_node_attack import TargetNodeAttack
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation

logger = logging.getLogger(__name__)


def run(inputs, args, device):
    if args.defense_method != "DShield":
        raise ValueError("此适配器只运行作者DShield模式。")
    data = inputs["data"]
    feat = inputs["poison_x"]
    train_edge_index = inputs["poison_edge_index"]
    train_edge_weight = inputs["poison_edge_weights"]
    labels = inputs["poison_labels"]
    train_idx = inputs["idx_train"]
    attach_idx = inputs["idx_attach"]
    val_idx = inputs["idx_val"]
    clean_test_idx = inputs["idx_clean_test"]
    atk_idx = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    backdoor_gen_model = inputs["trigger"]
    train_node_idx = torch.cat([train_idx, attach_idx]).to(device)
    feat_dim = data.x.shape[1]
    num_class = int(data.y.max().item()) + 1
    loop_hits = []
    evaluation_frames = []
    reference_seconds = 0.0
    _build_begin = start_clock(device)
    begin_time = time.time()

    if attach_idx is not None:
        logger.info("Percent of left attach nodes: {:.3f}".format(
            len(set(train_node_idx.tolist()) & set(attach_idx.tolist())) / len(attach_idx))
        )

    # Evaluation
    overall_asr, overall_ca = 0, 0
    if args.defense_method not in ['DShield']:
        benign_model = model_construct(
            dataset=args.dataset, model_name=args.model, feat_dim=feat_dim,
            num_class=num_class, hidden=args.benign_hidden, dropout=args.benign_dropout,
            lr=args.benign_lr, weight_decay=args.benign_weight_decay, device=device
        ).to(device)
        benign_model.fit(feat, train_edge_index, train_edge_weight,
                         labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=True)
        output = benign_model(feat, train_edge_index, train_edge_weight)
        if attach_idx is not None:
            train_attach_rate = (output.argmax(dim=1)[attach_idx] == labels[attach_idx]).float().mean()
            loop_hits.append({"node": int(ori_node_idx.item()), "target_hit": float(train_attach_rate)})
            logger.info("Target class rate on Vs: {:.4f}".format(train_attach_rate))
    elif args.defense_method in ['DShield']:
        _reference_begin = start_clock(device)
        poisoned_model = model_construct(
            dataset=args.dataset, model_name=args.model, feat_dim=feat_dim,
            num_class=num_class, hidden=args.benign_hidden, dropout=args.benign_dropout,
            lr=args.benign_lr, weight_decay=args.benign_weight_decay, device=device
        ).to(device)
        poisoned_model.fit(feat, train_edge_index, train_edge_weight,
                           labels, train_node_idx, val_idx, train_iters=args.benign_epochs, verbose=False)
        reference_seconds = elapsed(_reference_begin, device)

        if args.defense_method == 'DShield':
            benign_model = dshield(
                poisoned_model, feat, train_edge_index, train_edge_weight, labels, train_node_idx, attach_idx, val_idx,
                args.benign_hidden, lr=args.benign_lr, weight_decay=args.benign_weight_decay, kappa1=args.dshield_kappa1,
                kappa2=args.dshield_kappa2, kappa3=args.dshield_kappa3, thresh=args.dshield_thresh, edge_drop_ratio=args.dshield_edge_drop_ratio,
                feature_drop_ratio=args.dshield_feature_drop_ratio, tau=args.dshield_tau, pretrain_epochs=args.dshield_pretrain_epochs,
                classify_epochs=args.dshield_classify_epochs, finetune_epochs=args.dshield_finetune_epochs, device=device,
                balance_factor=args.dshield_balance_factor, classify_rounds=args.dshield_classify_rounds, dataset=args.dataset, arch=args.model,
                attack_name=args.attack_method
            )
        else:
            benign_model = copy.deepcopy(poisoned_model)
    else:
        raise Exception('Error!')

    end_time = time.time()
    model_build_seconds = elapsed(_build_begin, device)
    logger.info('Defense Time = {:.6f}s'.format(end_time - begin_time))

    evaluation_frames.append(capture_evaluation(benign_model, device))
    _prediction_begin = start_clock(device)
    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.cat([train_edge_weight, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])

    if args.defense_method not in ['DShield']:
        clean_acc = benign_model.test(feat, induct_edge_index, induct_edge_weights, data.y, clean_test_idx)
    elif args.defense_method in ['DShield']:
        clean_acc = model_test(benign_model, feat, induct_edge_index, induct_edge_weights, data.y, clean_test_idx)
    else:
        raise Exception('Error!')
    logger.info("Accuracy on clean test nodes: {:.4f}".format(clean_acc))

    asr = 0
    benign_model.eval()
    target_classes = args.target_class
    if not isinstance(target_classes, int):
        target_classes = [int(i) for i in target_classes.split('-')]
    else:
        target_classes = [target_classes]

    asr = 0
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(), induct_edge_weights.clone()
    flip_idx_atk = atk_idx[(data.y[atk_idx] != labels[atk_idx]).nonzero().flatten()]
    for i, idx in enumerate(atk_idx):
        idx = int(idx)
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
            node_idx=[idx], num_hops=2, edge_index=overall_induct_edge_index, relabel_nodes=True
        )  # sub_mapping means the index of [idx] in sub_nodeset
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)

        induct_input, induct_edge_index, induct_edge_weights = feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
        backdoor_gen_model.set_node_ids(sub_induct_nodeset)
        cur_idx = 0
        if isinstance(backdoor_gen_model, list):
            num_attacks = len(backdoor_gen_model)
            cur_idx = idx % num_attacks
            cur_backdoor_gen_model = backdoor_gen_model[cur_idx]
            if isinstance(cur_backdoor_gen_model, SBA):
                induct_input, induct_edge_index, induct_edge_weights = cur_backdoor_gen_model.inject_trigger_rand(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index
                )
            elif isinstance(backdoor_gen_model, TargetNodeAttack):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights, None
                )
            else:
                induct_input, induct_edge_index, induct_edge_weights = cur_backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
                )
            induct_input, induct_edge_index, induct_edge_weights = induct_input.clone().detach(), \
                induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
        elif backdoor_gen_model is not None:
            # inject trigger on attack test nodes (idx_atk)
            if isinstance(backdoor_gen_model, SBA):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger_rand(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index
                )
            elif isinstance(backdoor_gen_model, TargetNodeAttack):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights, None
                )
            else:
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
                )
            induct_input, induct_edge_index, induct_edge_weights = induct_input.clone().detach(), \
                induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

        # attack evaluation
        with torch.no_grad():
            output = benign_model(induct_input, induct_edge_index, induct_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx] == target_classes[cur_idx]).float().mean()
        loop_hits.append({"node": int(ori_node_idx.item()), "target_hit": float(train_attach_rate)})
        asr += train_attach_rate
        induct_input, induct_edge_index, induct_edge_weights = induct_input.cpu(), induct_edge_index.cpu(), induct_edge_weights.cpu()
        output = output.cpu()

    overall_asr = asr / (atk_idx.shape[0])
    overall_ca = clean_acc
    benign_model = benign_model.cpu()
    logger.info("ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.model, args.seed))
    logger.info("Accuracy: {:.4f}".format(overall_ca))

    prediction_seconds = elapsed(_prediction_begin, device)
    return {"models": [benign_model],
            "evaluation_frames": evaluation_frames,
            "source_metrics": {"CA": float(overall_ca), "ASR": float(overall_asr),
                               "CA_scope": "作者正常测试：完整heldout边恢复后model_test",
                               "ASR_scope": "作者全部atk_idx：逐节点二跳子图注入；包含原目标类，不使用未被统计的flip_idx_atk"},
            "defended_edges": train_edge_index, "defended_weights": train_edge_weight,
            "author_training_nodes": train_node_idx.cpu().tolist(),
            "author_attached_nodes": attach_idx.cpu().tolist(),
            "auxiliary_models": {"reference": poisoned_model}, "entropy_threshold": None,
            "source_args": vars(args).copy(), "internal_seeds": [int(args.seed)],
            "model_build_seconds": float(model_build_seconds),
            "reference_seconds": float(reference_seconds), "prediction_seconds": float(prediction_seconds),
            "loop_hits": loop_hits,
            "timing_scope": {"reference": "作者poisoned_model构造与fit，已计入model_build_seconds",
                             "model_build": "作者完整Defend段：准备模型和dshield原函数",
                             "prediction": "作者正常测试、全部atk逐节点二跳抽取、注入和推理"}}


def evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):
    if args.defense_method != "DShield":
        raise ValueError("此回调只评估作者DShield模式。")
    data = inputs["data"]
    feat = inputs["poison_x"]
    train_edge_index = inputs["poison_edge_index"]
    train_edge_weight = inputs["poison_edge_weights"]
    labels = inputs["poison_labels"]
    clean_test_idx = inputs["idx_clean_test"]
    atk_idx = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    backdoor_gen_model = inputs["trigger"]
    benign_model = test_model
    loop_hits = []
    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.cat([train_edge_weight, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])

    if args.defense_method not in ['DShield']:
        clean_acc = benign_model.test(feat, induct_edge_index, induct_edge_weights, data.y, clean_test_idx)
    elif args.defense_method in ['DShield']:
        clean_acc = model_test(benign_model, feat, induct_edge_index, induct_edge_weights, data.y, clean_test_idx)
    else:
        raise Exception('Error!')

    asr = 0
    benign_model.eval()
    target_classes = args.target_class
    if not isinstance(target_classes, int):
        target_classes = [int(i) for i in target_classes.split('-')]
    else:
        target_classes = [target_classes]

    asr = 0
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(), induct_edge_weights.clone()
    flip_idx_atk = atk_idx[(data.y[atk_idx] != labels[atk_idx]).nonzero().flatten()]
    for i, idx in enumerate(atk_idx):
        idx = int(idx)
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
            node_idx=[idx], num_hops=2, edge_index=overall_induct_edge_index, relabel_nodes=True
        )  # sub_mapping means the index of [idx] in sub_nodeset
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)

        induct_input, induct_edge_index, induct_edge_weights = feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
        backdoor_gen_model.set_node_ids(sub_induct_nodeset)
        cur_idx = 0
        if isinstance(backdoor_gen_model, list):
            num_attacks = len(backdoor_gen_model)
            cur_idx = idx % num_attacks
            cur_backdoor_gen_model = backdoor_gen_model[cur_idx]
            if isinstance(cur_backdoor_gen_model, SBA):
                induct_input, induct_edge_index, induct_edge_weights = cur_backdoor_gen_model.inject_trigger_rand(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index
                )
            elif isinstance(backdoor_gen_model, TargetNodeAttack):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights, None
                )
            else:
                induct_input, induct_edge_index, induct_edge_weights = cur_backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
                )
            induct_input, induct_edge_index, induct_edge_weights = induct_input.clone().detach(), \
                induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
        elif backdoor_gen_model is not None:
            # inject trigger on attack test nodes (idx_atk)
            if isinstance(backdoor_gen_model, SBA):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger_rand(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index
                )
            elif isinstance(backdoor_gen_model, TargetNodeAttack):
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights, None
                )
            else:
                induct_input, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                    relabeled_node_idx, feat[sub_induct_nodeset], sub_induct_edge_index, sub_induct_edge_weights
                )
            induct_input, induct_edge_index, induct_edge_weights = induct_input.clone().detach(), \
                induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

        # attack evaluation
        with torch.no_grad():
            output = benign_model(induct_input, induct_edge_index, induct_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx] == target_classes[cur_idx]).float().mean()
        loop_hits.append({"node": int(ori_node_idx.item()), "target_hit": float(train_attach_rate)})
        asr += train_attach_rate
        induct_input, induct_edge_index, induct_edge_weights = induct_input.cpu(), induct_edge_index.cpu(), induct_edge_weights.cpu()
        output = output.cpu()

    overall_asr = asr / (atk_idx.shape[0])
    overall_ca = clean_acc
    benign_model = benign_model.cpu()

    return {"CA": overall_ca, "ASR": overall_asr,
            "CA_scope": "作者正常测试：已保存训练边恢复heldout边后model_test",
            "ASR_scope": "作者全部atk_idx：逐节点二跳子图注入，包含原目标类；保留原逐节点累计",
            "loop_hits": loop_hits}
