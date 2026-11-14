"""从固定作者入口提取完整防御与评估段；不改写作者数学。"""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src" / "author_methods"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_sha1(path):
    content = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def add_line(mapping, number, line):
    if number not in mapping:
        mapping[number] = []
    mapping[number].append(line)


def call_is(node, receiver, name):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return isinstance(node.func.value, ast.Name) and node.func.value.id == receiver and node.func.attr == name


def apply_lines(lines, start, end, before, after, replacements):
    result = []
    for number in range(start, end + 1):
        result.extend(before.get(number, []))
        if number in replacements:
            if replacements[number] is not None:
                result.append(replacements[number])
        else:
            result.append(lines[number - 1])
        result.extend(after.get(number, []))
    return "\n".join(result)


def write_adapter(name, source, text, segments, changes):
    ast.parse(text)
    path = DESTINATION / (name + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"adapter": path.relative_to(ROOT).as_posix(),
            "adapter_sha256": sha256(path), "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(source), "source_git_blob_sha1": blob_sha1(source),
            "segments": segments, "changes": changes}


def mad_evaluator(original, tree, lines):
    loop = next(node for node in ast.walk(tree)
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "seed")
    fit_index = next(index for index, node in enumerate(loop.body)
                     if isinstance(node, ast.Expr) and call_is(node.value, "test_model", "fit"))
    first = loop.body[fit_index + 1].lineno
    last = loop.body[-1].end_lineno
    replacements = {}
    removed = []
    for node in ast.walk(loop):
        if not hasattr(node, "lineno") or node.lineno < first:
            continue
        kind = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
                kind = "pure_print"
            if call_is(node.value, "result_asr", "append") or call_is(node.value, "result_acc", "append"):
                kind = "outer_five_seed_accumulation"
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"overall_asr", "overall_ca"}:
                kind = "outer_five_seed_accumulation"
        if isinstance(node, ast.If):
            updates = [part for part in ast.walk(node) if isinstance(part, ast.AugAssign)]
            if updates and all(isinstance(part.target, ast.Name) and
                               part.target.id in {"overall_asr", "overall_ca"} for part in updates):
                kind = "outer_five_seed_accumulation"
        if kind is not None:
            for number in range(node.lineno, node.end_lineno + 1):
                replacements[number] = None
            removed.append({"kind": kind, "first_line": node.lineno, "last_line": node.end_lineno})
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "test_model":
                call = ast.get_source_segment(original, node.value)
                replacements[node.lineno] = " " * node.col_offset + "output = logits_of(" + call + ")"
    body = textwrap.dedent(apply_lines(lines, first, last, {}, {}, replacements))
    header = '''

def evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):
    if args.defense_mode != "MAD":
        raise ValueError("此回调只评估作者完整MAD模式。")
    data = inputs["data"]
    poison_x = inputs["poison_x"]
    poison_edge_index = inputs["poison_edge_index"]
    poison_edge_weights = inputs["poison_edge_weights"]
    idx_attach = inputs["idx_attach"]
    idx_clean_test = inputs["idx_clean_test"]
    idx_atk = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    model = inputs["trigger"]
    gcn_encoder = auxiliary["encoder"]
    gnn_model = auxiliary["confidence"]
    prediction_entropy_threshold = entropy_threshold
'''
    ending = '''
    return {"CA": clean_acc, "ASR": flip_asr,
            "CA_scope": "作者clean_acc：已保存净化训练边加未额外净化heldout边；当前一个原生模型",
            "ASR_scope": "作者flip_asr：原标签非目标类别的idx_atk；原测试联合掩码；当前一个原生模型"}
'''
    return header + textwrap.indent(body, "    ") + "\n" + ending, {
        "kind": "native_single_model_evaluation", "first_line": first, "last_line": last,
        "removed": removed}


def build_mad(name, relative):
    source = ROOT / "upstream" / "MAD" / relative
    original = source.read_text(encoding="utf-8-sig")
    lines = original.splitlines()
    tree = ast.parse(original)
    defense = None
    for node in tree.body:
        if isinstance(node, ast.If) and "defense_mode" in ast.unparse(node.test):
            defense = node
            break
    if defense is None:
        raise RuntimeError("作者入口没有找到完整防御条件段。")
    helper_texts = []
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            helper_texts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
            segments.append({"kind": "native_helper", "name": node.name,
                             "first_line": node.lineno, "last_line": node.end_lineno,
                             "original_scope": "module", "adapter_scope": "module"})
    before, after, replacements = {}, {}, {}
    changes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.lineno >= defense.lineno:
            for number in range(node.lineno, node.end_lineno + 1):
                replacements[number] = None
            changes.append({"kind": "preserve_original_helper_scope", "name": node.name,
                            "first_line": node.lineno, "last_line": node.end_lineno,
                            "description": "原顶层辅助函数整体保留在模块层，函数体及调用不变。"})
    add_line(after, defense.end_lineno, "reference_seconds = elapsed(_reference_begin, device)")
    add_line(after, defense.end_lineno, "model_build_seconds = reference_seconds")
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or node.lineno < defense.lineno:
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "models" for target in node.targets):
                replacements[node.lineno] = "models = [requested_model]"
                changes.append({"line": node.lineno, "kind": "requested_final_model",
                                "original": lines[node.lineno - 1], "replacement": replacements[node.lineno]})
            if isinstance(node.value, ast.Call) and any(
                    isinstance(target, ast.Name) and target.id == "test_model" for target in node.targets):
                constructors = [part for part in ast.walk(node.value)
                                if isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                                and part.func.id == "model_construct"]
                if constructors:
                    add_line(before, node.lineno, " " * node.col_offset + "_final_build_begin = start_clock(device)")
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "test_model":
                    target = node.targets[0]
                    if isinstance(target, ast.Tuple) or isinstance(target, ast.Name) and target.id == "output":
                        call = ast.get_source_segment(original, node.value)
                        replacements[node.lineno] = " " * node.col_offset + "output = logits_of(" + call + ")"
                        changes.append({"line": node.lineno, "kind": "native_tuple_tensor_interface",
                                        "original": lines[node.lineno - 1], "replacement": replacements[node.lineno]})
        if isinstance(node, ast.Expr) and call_is(node.value, "test_model", "fit"):
            indent = " " * node.col_offset
            add_line(after, node.end_lineno, indent + "model_build_seconds += elapsed(_final_build_begin, device)")
            add_line(after, node.end_lineno, indent + "evaluation_frames.append(capture_evaluation(test_model, device))")
            add_line(after, node.end_lineno, indent + "_prediction_begin = start_clock(device)")
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "seed":
            indent = " " * (node.col_offset + 4)
            add_line(after, node.body[-1].end_lineno, indent + "native_models.append(test_model)")
            add_line(after, node.body[-1].end_lineno, indent + "prediction_seconds += elapsed(_prediction_begin, device)")
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "seeds" for t in node.targets):
            add_line(after, node.end_lineno, " " * node.col_offset + "internal_seeds = seeds.tolist()")
    body = apply_lines(lines, defense.lineno, len(lines), before, after, replacements)
    segments.append({"kind": "complete_post_poisoned_flow", "first_line": defense.lineno,
                     "last_line": len(lines)})
    preamble = '''"""完整保留作者MAD入口的防御与五种子评估；由专用脚本生成。"""
import numpy as np
import torch
import torch.nn.functional as F
from models.construct import model_construct
from Detect_encoder import cluster_encoder_detect, prune_anomalous_edges_by_class
from anomaly_detect import prune_high_fluctuation_nodes, prune_edge, anomaly_embedding_edge, gnn_prune_low_entropy_nodes, prune_anomaly_nodes_edges
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation

'''
    bindings = '''
def run(inputs, args, device):
    requested_model = args.test_model
    if args.defense_mode != "MAD":
        raise ValueError("此适配器只运行作者完整MAD模式。")
    data = inputs["data"]
    poison_x = inputs["poison_x"]
    poison_edge_index = inputs["poison_edge_index"]
    poison_edge_weights = inputs["poison_edge_weights"]
    poison_labels = inputs["poison_labels"]
    idx_train = inputs["idx_train"]
    idx_attach = inputs["idx_attach"]
    idx_val = inputs["idx_val"]
    idx_clean_test = inputs["idx_clean_test"]
    idx_atk = inputs["idx_atk"]
    mask_edge_index = inputs["mask_edge_index"]
    model = inputs["trigger"]
    current_time = inputs["current_time"]
    idx_clean, idx_trigger = get_idx_clean_and_trigger(poison_x, data.x)
    native_models = []
    evaluation_frames = []
    internal_seeds = []
    prediction_seconds = 0.0
    _reference_begin = start_clock(device)
'''
    ending = '''
    # 原循环完成后仅将保留对象移至CPU，便于原生检查点保存。
    for native_model in native_models:
        native_model.cpu()
    return {"models": native_models,
            "evaluation_frames": evaluation_frames,
            "source_metrics": {"CA": float(total_overall_ca), "ASR": float(total_overall_asr),
                               "CA_scope": "作者clean_acc：净化训练边加未额外净化heldout边；五个内部种子均值",
                               "ASR_scope": "作者flip_asr：全部idx_atk中原标签非目标类别；触发图联合掩码后；五种子均值"},
            "defended_edges": poison_edge_index, "defended_weights": poison_edge_weights,
            "author_training_nodes": bkd_tn_nodes.cpu().tolist(),
            "author_attached_nodes": idx_attach.cpu().tolist(),
            "auxiliary_models": {"encoder": gcn_encoder, "confidence": gnn_model},
            "entropy_threshold": prediction_entropy_threshold,
            "source_args": vars(args).copy(), "internal_seeds": internal_seeds,
            "model_build_seconds": float(model_build_seconds),
            "reference_seconds": float(reference_seconds), "prediction_seconds": float(prediction_seconds),
            "timing_scope": {"reference": "作者两个辅助模型完整训练与训练图联合处理，计入model_build_seconds",
                             "model_build": "完整辅助阶段及原生五次最终模型构造和fit的累计",
                             "prediction": "五种子原生训练诊断前向、正常测试、触发注入与联合处理、测试和循环内CPU传输"}}
'''
    callback, callback_segment = mad_evaluator(original, tree, lines)
    segments.append(callback_segment)
    text = preamble + "\n\n".join(helper_texts) + "\n" + bindings + textwrap.indent(body, "    ") + "\n" + ending + callback
    changes.extend([{"kind": "fixed_bundle_input", "description": "仅注入已冻结原图、污染图、索引与作者生成器接口；不运行入口前半段生成。"},
                    {"kind": "read_only_budget_metadata", "description": "只读返回作者实际bkd_tn_nodes和idx_attach列表，不改变监督集合。"},
                    {"kind": "evaluation_frame_and_callback", "description": "每次fit后只读记录训练模式与RNG；纯评估回调复制原预测段，仅去掉外层五种子累计与打印。"},
                    {"kind": "measurement_and_capture", "description": "插入同步计时及原生模型对象收集，不改原运算和五种子循环。"}])
    return write_adapter(name, source, text, segments, changes)


def dshield_evaluator(body_lines, original_start):
    text = "\n".join(body_lines)
    tree = ast.parse(text)
    first = next(node.lineno for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "induct_edge_index"
                         for target in node.targets))
    before, after, replacements = {}, {}, {}
    removed = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or node.lineno < first:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            is_print = isinstance(function, ast.Name) and function.id == "print"
            is_logger = isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "logger"
            if is_print or is_logger:
                for number in range(node.lineno, node.end_lineno + 1):
                    replacements[number] = None
                removed.append({"kind": "pure_print", "first_line": original_start + node.lineno - 1,
                                "last_line": original_start + node.end_lineno - 1})
        if isinstance(node, ast.Assign):
            if node.col_offset == 4 and any(isinstance(target, ast.Name) and target.id == "cur_idx" for target in node.targets):
                add_line(before, node.lineno, " " * node.col_offset + "backdoor_gen_model.set_node_ids(sub_induct_nodeset)")
            if any(isinstance(target, ast.Name) and target.id == "train_attach_rate" for target in node.targets):
                add_line(after, node.end_lineno, " " * node.col_offset +
                         'loop_hits.append({"node": int(ori_node_idx.item()), "target_hit": float(train_attach_rate)})')
    body = apply_lines(body_lines, first, len(body_lines), before, after, replacements)
    header = '''

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
'''
    ending = '''
    return {"CA": overall_ca, "ASR": overall_asr,
            "CA_scope": "作者正常测试：已保存训练边恢复heldout边后model_test",
            "ASR_scope": "作者全部atk_idx：逐节点二跳子图注入，包含原目标类；保留原逐节点累计",
            "loop_hits": loop_hits}
'''
    return header + textwrap.indent(body, "    ") + "\n" + ending, {
        "kind": "native_single_model_evaluation", "first_line": original_start + first - 1,
        "last_line": original_start + len(body_lines) - 1, "removed": removed}


def build_dshield():
    source = ROOT / "upstream" / "DShield" / "NodeClassificationTasks" / "main.py"
    original = source.read_text(encoding="utf-8-sig")
    lines = original.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "# Defense":
            start = index + 2
            break
    if start is None:
        raise RuntimeError("没有找到DShield作者防御段。")
    body = textwrap.dedent("\n".join(lines[start - 1:]))
    body_lines = body.splitlines()
    tree = ast.parse(body)
    before, after, replacements = {}, {}, {}
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "poisoned_model" for t in node.targets):
                add_line(before, node.lineno, " " * node.col_offset + "_reference_begin = start_clock(device)")
            if any(isinstance(t, ast.Name) and t.id == "end_time" for t in node.targets):
                add_line(after, node.end_lineno, "model_build_seconds = elapsed(_build_begin, device)")
            if any(isinstance(t, ast.Name) and t.id == "induct_edge_index" for t in node.targets):
                if node.col_offset == 0:
                    add_line(before, node.lineno, "evaluation_frames.append(capture_evaluation(benign_model, device))")
                    add_line(before, node.lineno, "_prediction_begin = start_clock(device)")
            if node.col_offset == 4 and any(isinstance(t, ast.Name) and t.id == "cur_idx" for t in node.targets):
                add_line(before, node.lineno, " " * node.col_offset + "backdoor_gen_model.set_node_ids(sub_induct_nodeset)")
                changes.append({"line": start + node.lineno - 1, "kind": "local_node_identity_mapping"})
            if any(isinstance(t, ast.Name) and t.id == "train_attach_rate" for t in node.targets):
                add_line(after, node.end_lineno, " " * node.col_offset +
                         'loop_hits.append({"node": int(ori_node_idx.item()), "target_hit": float(train_attach_rate)})')
        if isinstance(node, ast.Expr) and call_is(node.value, "poisoned_model", "fit"):
            add_line(after, node.end_lineno, " " * node.col_offset + "reference_seconds = elapsed(_reference_begin, device)")
    body = apply_lines(body_lines, 1, len(body_lines), before, after, replacements)
    header = '''"""DShield作者main的原始防御与逐节点二跳评估完整段。"""
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
'''
    ending = '''
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
'''
    callback, callback_segment = dshield_evaluator(body_lines, start)
    return write_adapter("dshield", source, header + textwrap.indent(body, "    ") + "\n" + ending + callback,
                         [{"kind": "complete_defense_and_evaluation", "first_line": start, "last_line": len(lines)},
                          callback_segment],
                         changes + [{"kind": "fixed_bundle_input", "description": "保留原完整标签、原fit和dshield调用，不补伪标签。"},
                                    {"kind": "read_only_budget_metadata", "description": "只读返回作者实际train_node_idx和attach_idx列表，不改变监督集合。"},
                                    {"kind": "evaluation_frame_and_callback", "description": "原Defend结束时只读记录模式与RNG；回调保留正常测试、逐节点二跳注入、原节点累计及原no_grad位置。"},
                                    {"kind": "measurement_and_capture", "description": "仅插入同步计时和逐节点命中日志。"}])


def main():
    records = []
    for name, relative in [("mad_gta", "MAD/GTA_and_UGBA/run_GTA.py"),
                           ("mad_ugba", "MAD/GTA_and_UGBA/run_adaptive.py"),
                           ("mad_dpgba", "MAD/DPGBA/DPGBA/run/run_adaptive.py")]:
        records.append(build_mad(name, relative))
    records.append(build_dshield())
    report = {"date": "2026-11-11", "MAD_commit": "f47cef85ecc70e1946a3da79c3e5fb37723594fb",
              "algorithm_policy": "复制原始完整post-poisoned段与原函数调用；不重写数学。",
              "full_labels_policy": "用户已确认MAD全原节点标签条件；此适配不以伪标签替代。",
              "records": records}
    target = ROOT / "paper" / "mad_dshield_segments_20261111.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已生成四个作者完整调用适配器，仅静态解析，无训练。")


if __name__ == "__main__":
    main()
