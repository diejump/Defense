"""直接调用LoSplit的Prune函数、原模型工厂及fit；不重写剪边公式。"""
import torch
from models.construct import model_construct
from upstream.LoSplit.help_funcs import prune_unrelated_edge
from src.author_support import start_clock, elapsed, capture_evaluation, logits_of


def run(inputs, args, device):
    begin = start_clock(device)
    x = inputs["poison_x"]
    edges, weights = prune_unrelated_edge(args, inputs["poison_edge_index"],
        inputs["poison_edge_weights"], x, device, large_graph=False)
    attached = inputs["idx_attach"]
    if args.attack == "UGBA":
        attached = attached[inputs["data"].y[attached] != args.target_class]
    train_nodes = torch.cat([inputs["idx_train"], attached]).to(device)
    model = model_construct(args, args.test_model, inputs["data"], device).to(device)
    model.fit(x, edges, weights, inputs["poison_labels"], train_nodes,
              inputs["idx_val"], train_iters=args.epochs, verbose=False)
    model.eval()
    build_seconds = elapsed(begin, device)
    frame = capture_evaluation(model, device)
    filtered = inputs.copy()
    filtered["poison_edge_index"], filtered["poison_edge_weights"] = edges, weights
    begin = start_clock(device)
    metrics = evaluate(filtered, args, device, model)
    return {"models": [model], "evaluation_frames": [frame], "source_metrics": metrics,
            "defended_edges": edges, "defended_weights": weights, "auxiliary_models": {},
            "source_args": vars(args).copy(), "internal_seeds": [],
            "author_training_nodes": train_nodes.cpu().tolist(),
            "author_attached_nodes": attached.cpu().tolist(), "model_build_seconds": build_seconds,
            "reference_seconds": 0.0, "prediction_seconds": elapsed(begin, device)}


def evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):
    x = inputs["poison_x"]
    heldout = inputs["mask_edge_index"]
    edges = torch.cat([inputs["poison_edge_index"], heldout], dim=1)
    weights = torch.cat([inputs["poison_edge_weights"], x.new_ones(heldout.size(1))])
    # LoSplit原入口先评估正常图，再注入，并在触发图上再次调用Prune。
    ca = test_model.test(x, edges, weights, inputs["data"].y, inputs["idx_clean_test"])
    x, edges, weights = inputs["trigger"].inject_trigger(inputs["idx_atk"], x, edges, weights, device)
    x, edges, weights = x.clone().detach(), edges.clone().detach(), weights.clone().detach()
    edges, weights = prune_unrelated_edge(args, edges, weights, x, device)
    output = logits_of(test_model(x, edges, weights))
    indices = inputs["idx_atk"]
    indices = indices[inputs["data"].y[indices] != args.target_class]
    asr = (output.argmax(dim=1)[indices] == args.target_class).float().mean()
    return {"CA": float(ca), "ASR": float(asr),
            "CA_scope": "LoSplit原Prune入口：净化训练边加原heldout边",
            "ASR_scope": "全部攻击测试节点整体注入后再次原Prune；只统计非目标类"}
