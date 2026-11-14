"""无防御和ABP的评估；作者比较方法使用各自原始评估代码。"""
import torch
from tqdm.auto import tqdm
from .abp import filter_edges
from .common import start_clock, elapsed
from .author_support import logits_of


@torch.no_grad()
def predict(model, x, edges, weights, training_nodes, method):
    if method == "abp":
        edges, weights = filter_edges(edges, weights, x.size(0), training_nodes)
    model.eval()
    output = logits_of(model(x, edges, weights))
    if not torch.isfinite(output).all():
        raise RuntimeError("模型预测含非有限值。")
    return output


def evaluate(inputs, args, device, model, auxiliary=None, entropy_threshold=None):
    method = args.method
    edges = torch.cat([inputs["poison_edge_index"], inputs["mask_edge_index"]], dim=1)
    weights = torch.cat([inputs["poison_edge_weights"], inputs["poison_x"].new_ones(inputs["mask_edge_index"].size(1))])
    graphs = [(inputs["poison_x"], edges, weights)]
    if inputs["trigger"] is not None:
        graphs.append(inputs["trigger"].inject_trigger(inputs["idx_atk"], *graphs[0]))
    outputs = []
    begin = start_clock(device)
    for x, edge, weight in tqdm(graphs, desc="正常图与触发图评估", unit="张", leave=False):
        outputs.append(predict(model, x, edge, weight, inputs["training"]["train"], method))
    truth = inputs["data"].y
    normal = inputs["idx_clean_test"]
    ca_hits = int(outputs[0][normal].argmax(-1).eq(truth[normal]).sum())
    metrics = {"CA": ca_hits / normal.numel(), "ASR": None, "CA_hits": ca_hits,
               "CA_denominator": normal.numel(), "ASR_hits": None, "ASR_denominator": 0,
               "CA_scope": "正常完整测试图上的正常测试节点",
               "ASR_scope": "完整图整体注入；原标签非目标类攻击测试节点",
               "prediction_seconds": elapsed(begin, device)}
    if len(outputs) == 2:
        target = inputs["evaluation"]["target"]
        affected = inputs["idx_atk"][truth[inputs["idx_atk"]] != target]
        hits = int(outputs[1][affected].argmax(-1).eq(target).sum())
        metrics.update(ASR=hits / affected.numel(), ASR_hits=hits, ASR_denominator=affected.numel())
    return metrics
