"""训练完成后才读取测试标签，分别计算CA和ASR。"""
from pathlib import Path
import torch
from tqdm.auto import tqdm
from .abp import filter_edges
from .common import start_clock, elapsed


@torch.no_grad()
def predict(model, graph, training_nodes, method):
    x, edges, weights = graph
    if method == "abp":
        edges, weights = filter_edges(edges, weights, x.size(0), training_nodes)
    model.eval()
    output = model(x, edges, weights)
    if not torch.isfinite(output).all():
        raise RuntimeError("模型预测含NaN或无穷值，请检查训练损失与保存权重。")
    return output


def evaluate(model, data, bundle, method, details, device):
    evaluation = torch.load(Path(bundle) / "evaluation.pt", map_location=device)
    graphs = [(evaluation["reference_x"], evaluation["reference_edges"], evaluation["reference_weights"])]
    if evaluation["target"] is not None:
        graphs.append((evaluation["changed_x"], evaluation["changed_edges"], evaluation["changed_weights"]))
    outputs = []
    seconds = []
    for graph in tqdm(graphs, desc="正常图与触发图评估", unit="张", leave=False):
        begin = start_clock(device)
        output = predict(model, graph, data["train"], method)
        seconds.append(elapsed(begin, device))
        outputs.append(output)
    normal = evaluation["clean_nodes"]
    affected = evaluation["attack_nodes"]
    ca_hits = int(outputs[0][normal].argmax(-1).eq(evaluation["truth"][normal]).sum())
    result = {"CA": ca_hits / normal.numel(), "ASR": None,
              "CA_hits": ca_hits, "CA_denominator": normal.numel(),
              "ASR_hits": None, "ASR_denominator": affected.numel(),
              "CA_on_triggered_graph": None,
              "reference_inference_seconds": seconds[0], "triggered_inference_seconds": None}
    if evaluation["target"] is not None:
        asr_hits = int(outputs[1][affected].argmax(-1).eq(evaluation["target"]).sum())
        result["ASR"] = asr_hits / affected.numel()
        result["ASR_hits"] = asr_hits
        result["CA_on_triggered_graph"] = int(outputs[1][normal].argmax(-1).eq(evaluation["truth"][normal]).sum()) / normal.numel()
        result["triggered_inference_seconds"] = seconds[1]
    if method == "abp" and evaluation["target"] is not None:
        old_count = graphs[0][0].size(0)
        result["old_prediction_changes"] = int(outputs[0].argmax(-1).ne(outputs[1][:old_count].argmax(-1)).sum())
        result["untriggered_target_predictions"] = int(outputs[0][affected].argmax(-1).eq(evaluation["target"]).sum())
    if "selected_nodes" in details:
        selected = set(details["selected_nodes"])
        attached = set(evaluation["attached"].cpu().tolist())
        true_positive = len(selected & attached)
        result["selected_count"] = len(selected)
        result["selection_TP"] = true_positive
        result["selection_FP"] = len(selected - attached)
        result["selection_recall"] = true_positive / len(attached)
        result["selection_precision"] = true_positive / len(selected) if selected else None
    return result
