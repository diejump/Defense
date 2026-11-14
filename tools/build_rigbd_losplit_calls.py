"""从核对后的作者入口抽取调用片段；不重新实现评分或恢复损失。"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src" / "author_methods"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class InterfaceOnly(ast.NodeTransformer):
    """核对时只归一化已声明的外层读取与配置参数化。"""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "logits_of":
            return node.args[0]
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        target = node.targets[0]
        if isinstance(target, ast.Tuple) and isinstance(target.elts[0], ast.Name):
            if target.elts[0].id == "output":
                node.targets = [ast.Name(id="output", ctx=ast.Store())]
        if isinstance(target, ast.Name) and target.id in {"K", "drop_ratio", "models"}:
            node.value = ast.Name(id="AUTHOR_PARAMETER_" + target.id, ctx=ast.Load())
        return node


def statement_key(node):
    return ast.dump(InterfaceOnly().visit(deepcopy(node)), include_attributes=False)


def check_statement_preservation(source, segments, generated, function_name="run", skip_diagnostic_prints=False):
    originals = []
    for first, last in segments:
        for node in ast.parse(lines(source, first, last)).body:
            if not isinstance(node, ast.FunctionDef):
                if skip_diagnostic_prints and diagnostic_print(node):
                    continue
                originals.append(node)
    run = None
    for node in ast.parse(generated).body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            run = node
    position = 0
    for original in originals:
        expected = statement_key(original)
        found = False
        while position < len(run.body):
            actual = statement_key(run.body[position])
            position += 1
            if actual == expected:
                found = True
                break
        if not found:
            raise ValueError("生成调用中缺少原作者语句：" + ast.unparse(original)[:160])
    return {"original_top_level_statements": len(originals),
            "ordered_author_statements_preserved": True,
            "normalization": "仅外层logits_of、未使用的Tuple附加值读取、K/drop与模型列表参数化。"}


def diagnostic_print(node):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    if not isinstance(node.value.func, ast.Name) or node.value.func.id != "print":
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id in {"correct", "indexs"}:
            return True
    return False


def lines(source, first, last):
    return "".join(source.splitlines(keepends=True)[first - 1:last])


def find_assignment(tree, name, call_attribute=None):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for item in target.elts:
                    if isinstance(item, ast.Name):
                        names.append(item.id)
        if name not in names:
            continue
        if call_attribute is not None:
            if not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Attribute):
                continue
            if node.value.func.attr != call_attribute:
                continue
        return node
    raise ValueError("未找到作者入口中的赋值位置：" + name)


def outer_logits(source):
    """仅适配外层forward返回的Tensor或Tuple，不触碰作者模型内部。"""
    tree = ast.parse(source)
    source_lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in source_lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id not in {"test_model", "test_model1"}:
            continue
        target = node.targets[0]
        valid = False
        if isinstance(target, ast.Name) and target.id in {"output", "output_clean"}:
            valid = True
        if isinstance(target, ast.Tuple) and isinstance(target.elts[0], ast.Name):
            if target.elts[0].id == "output":
                valid = True
                start = offsets[target.lineno - 1] + target.col_offset
                end = offsets[target.end_lineno - 1] + target.end_col_offset
                edits.append((start, end, "output"))
        if valid:
            start = offsets[call.lineno - 1] + call.col_offset
            end = offsets[call.end_lineno - 1] + call.end_col_offset
            edits.append((start, end, "logits_of(" + source[start:end] + ")"))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def bind_inputs():
    return '''data = inputs["data"]
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
'''


def imports(include_prune=True):
    text = '''import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from models.construct import model_construct
from src.author_support import start_clock, elapsed, logits_of, capture_evaluation
'''
    if include_prune:
        text += 'from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated\n'
    return text + "\n"


def finish(final_model, selected, ca_name, ca_scope):
    return '''return {
    "models": [''' + final_model + '''],
    "source_metrics": {"CA": float(''' + ca_name + '''), "ASR": float(flip_asr),
        "CA_scope": "''' + ca_scope + '''",
        "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"},
    "reference_source_metrics": reference_metrics,
    "selected_nodes": (''' + selected + ''').detach().cpu().tolist(),
    "author_training_nodes": bkd_tn_nodes.cpu().tolist(),
    "author_attached_nodes": idx_attach.cpu().tolist(),
    "evaluation_frames": [evaluation_frame],
    "defended_edges": poison_edge_index,
    "defended_weights": poison_edge_weights,
    "auxiliary_models": {},
    "source_args": vars(args).copy(),
    "internal_seeds": [],
    "model_build_seconds": model_build_seconds,
    "reference_seconds": reference_seconds,
    "prediction_seconds": prediction_seconds,
}
'''


def build_evaluate(source, final_fit, source_end, final_model, ca_name, ca_scope):
    first = final_fit.end_lineno + 1
    last = first
    omitted_prints = []
    for node in ast.parse(source).body:
        if node.lineno < first or node.end_lineno > source_end:
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {ca_name, "flip_asr"}:
                    last = max(last, node.end_lineno)
    prediction_parts = []
    for node in ast.parse(source).body:
        if node.lineno >= first and node.end_lineno <= last:
            if diagnostic_print(node):
                omitted_prints.append([node.lineno, node.end_lineno])
            else:
                prediction_parts.append(lines(source, node.lineno, node.end_lineno))
    # 纯评估只复制实际执行的原预测语句，不携带原注释中的训练备选。
    prediction = "".join(prediction_parts)
    body = bind_inputs()
    if final_model != "test_model":
        body += final_model + " = test_model\n"
    body += prediction
    body += '\nreturn {"CA": ' + ca_name + ', "ASR": flip_asr,\n'
    body += '    "CA_scope": "' + ca_scope + '",\n'
    body += '    "ASR_scope": "原作者对全部idx_atk整体注入后，仅以原标签非目标类节点计算ASR。"}\n'
    code = '\n\ndef evaluate(inputs, args, device, test_model, auxiliary=None, entropy_threshold=None):\n'
    code += textwrap.indent(outer_logits(body), "    ")
    return code, {"segments": [[first, last]], "omitted_diagnostic_prints": omitted_prints,
                  "scalar_policy": "保留原CA/ASR标量类型；不新增eval或no_grad，不重设随机种子。"}


def build_rigbd():
    relative = "upstream/RIGBD/defense.py"
    source = (ROOT / relative).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"sample_noise_all", "find_index"}:
            functions.append(node)
    reference_start = next(node.lineno for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "mask")
    formal_marker = source.index("###### formal test")
    formal_start = source[:formal_marker].count("\n") + 1
    final_fit = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "fit":
                for keyword in node.value.keywords:
                    if keyword.arg == "finetune" and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            final_fit = node
    if final_fit is None:
        raise ValueError("未找到原RIGBD的最终恢复调用。")
    reference = lines(source, reference_start, formal_start - 1)
    build = lines(source, formal_start, final_fit.end_lineno)
    for function in reversed(functions):
        original = lines(source, function.lineno, function.end_lineno)
        build = build.replace(original, "")
    build = build.replace("drop_ratio = 0.5", "drop_ratio = args.drop_ratio")
    build = build.replace("K=20", "K=args.K")
    prediction = lines(source, final_fit.end_lineno + 1, len(source.splitlines()))
    body = "global drop_ratio\n" + bind_inputs()
    body += "reference_begin = start_clock(device)\n" + reference
    body += '\nreference_seconds = elapsed(reference_begin, device)\nreference_metrics = {"CA": float(ca), "ASR": float(flip_asr)}\n'
    body += "model_build_begin = start_clock(device)\n" + build
    body += "\nmodel_build_seconds = elapsed(model_build_begin, device)\n"
    body += "evaluation_frame = capture_evaluation(test_model, device)\nprediction_begin = start_clock(device)\n" + prediction
    body += "\nprediction_seconds = elapsed(prediction_begin, device)\n"
    body += finish("test_model", "bkd_tn_nodes[index_of_less_robust][:result_index]", "ca", "原作者在已注入测试触发的整图上评估idx_clean_test。")
    header = '"""原RIGBD入口调用；仅参数化K/drop，保持原采样、目标及恢复流程。"""\n'
    header += imports(False) + "drop_ratio = 0.5\n\n"
    for function in functions:
        header += lines(source, function.lineno, function.end_lineno) + "\n\n"
    generated = header + "def run(inputs, args, device):\n" + textwrap.indent(outer_logits(body), "    ")
    evaluation, evaluation_details = build_evaluate(source, final_fit, len(source.splitlines()), "test_model", "ca",
        "原作者在已注入测试触发的整图上评估idx_clean_test。")
    generated += evaluation
    return generated, {"source": relative, "source_sha256": digest(source),
        "segments": [[reference_start, formal_start - 1], [formal_start, final_fit.end_lineno],
                     [final_fit.end_lineno + 1, len(source.splitlines())]],
        "lifted_functions": [[node.name, node.lineno, node.end_lineno] for node in functions],
        "adaptations": ["输入变量绑定", "模块级提升原函数", "K/drop常量参数化", "外层Tensor/Tuple读取", "阶段计时与结果封装",
                        "不复制片段内未使用的help_funcs导入；保留原models.construct工厂导入"],
        "internal_seed_note": "抽取开始点位于原入口初始化和生成器准备之后；片段内没有重新播种。",
        "evaluation": evaluation_details,
        "evaluation_frame_note": "最终fit之后、原最终预测之前只读捕获model.training及CPU/CUDA随机状态。"}


def build_losplit(attack):
    relative = "upstream/LoSplit/" + attack + "_LoSplit.py"
    if attack == "DPGBA":
        relative = "upstream/LoSplit/DPGBA/DPGBA_LoSplit.py"
    source = (ROOT / relative).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    poisoned = find_assignment(tree, "poison_x", "get_poisoned")
    split = find_assignment(tree, "LoSplit")
    target = find_assignment(tree, "target_label")
    saved = None
    final = find_assignment(tree, "test_model1")
    final_fit = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                if isinstance(call.func.value, ast.Name) and call.func.value.id == "torch" and call.func.attr == "save":
                    if call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "data_to_save":
                        saved = node
                if isinstance(call.func.value, ast.Name) and call.func.value.id == "test_model1" and call.func.attr == "fit":
                    final_fit = node
    if saved is None or final_fit is None:
        raise ValueError("未找到LoSplit保存材料或最终训练调用：" + attack)
    end = None
    for node in tree.body:
        if node.lineno <= final_fit.lineno or not isinstance(node, ast.Expr):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            if "FPR:" in ast.get_source_segment(source, node):
                end = node.end_lineno
                break
    if end is None:
        raise ValueError("未找到LoSplit第一轮结果结束位置：" + attack)
    reference = lines(source, poisoned.end_lineno + 1, split.lineno - 1)
    if attack == "DPGBA":
        reference = reference.replace("models = ['GCN']", "models = [requested_model]")
    build_split = lines(source, split.lineno, target.end_lineno)
    save_material = lines(source, target.end_lineno + 1, final.lineno - 1)
    build_final = lines(source, final.lineno, final_fit.end_lineno)
    prediction = lines(source, final_fit.end_lineno + 1, end)
    prelude = bind_inputs()
    if attack == "DPGBA":
        prelude += "requested_model = args.test_model\n"
    # 与原入口筛选语句相同；不排序附着名单，不清理已有污染图。
    prelude += 'idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]\n'
    prelude += 'known_nodes = torch.cat([idx_train,idx_attach]).to(device)\n'
    body = prelude + "reference_begin = start_clock(device)\n" + reference
    body += '\nreference_seconds = elapsed(reference_begin, device)\nreference_metrics = {"CA": float(clean_acc), "ASR": float(flip_asr)}\n'
    body += "model_build_begin = start_clock(device)\n" + build_split
    body += "\nmodel_build_seconds = elapsed(model_build_begin, device)\n" + save_material
    body += "model_build_begin = start_clock(device)\n" + build_final
    body += "\nmodel_build_seconds += elapsed(model_build_begin, device)\n"
    body += "evaluation_frame = capture_evaluation(test_model1, device)\nprediction_begin = start_clock(device)\n" + prediction
    body += "\nprediction_seconds = elapsed(prediction_begin, device)\n"
    body += finish("test_model1", "idx_poison_found", "clean_acc", "原作者在恢复测试关联边、尚未注入测试触发的图上评估idx_clean_test。")
    header = '"""原LoSplit ' + attack + '入口片段；保留作者划分与finetune2，不自写损失。"""\n'
    if attack == "DPGBA":
        header += '# 本模块必须在upstream/LoSplit/DPGBA独立命名空间内导入，不能使用LoSplit根分支模型。\n'
    module_imports = imports()
    if attack == "DPGBA":
        # 两分支函数AST相同，直接引用根目录原函数，避免无关的faiss导入。
        module_imports = module_imports.replace("from help_funcs import", "from upstream.LoSplit.help_funcs import")
    generated = header + module_imports + "def run(inputs, args, device):\n" + textwrap.indent(outer_logits(body), "    ")
    evaluation, evaluation_details = build_evaluate(source, final_fit, end, "test_model1", "clean_acc",
        "原作者在恢复测试关联边、尚未注入测试触发的图上评估idx_clean_test。")
    generated += evaluation
    adaptations = ["输入变量绑定", "原已有标签目标过滤", "外层Tensor/Tuple读取", "阶段计时与结果封装"]
    if attack == "DPGBA":
        adaptations.append("原models=['GCN']仅参数化为[requested_model]；保留原for循环，原模型内部问题不修复")
        adaptations.append("仅将Prune辅助导入指向LoSplit根目录原函数；两个分支对应函数AST相同，避免未使用的faiss依赖")
    return generated, {"source": relative, "source_sha256": digest(source),
        "segments": [[poisoned.end_lineno + 1, split.lineno - 1], [split.lineno, target.end_lineno],
                     [target.end_lineno + 1, final.lineno - 1], [final.lineno, final_fit.end_lineno],
                     [final_fit.end_lineno + 1, end]],
        "adaptations": adaptations, "model_namespace": "upstream/LoSplit/DPGBA" if attack == "DPGBA" else "upstream/LoSplit",
        "internal_seed_note": "原抽取片段内无重新播种；参考训练与全部原模型构造顺序保留。",
        "evaluation": evaluation_details,
        "evaluation_frame_note": "最终fit之后、原最终预测之前只读捕获model.training及CPU/CUDA随机状态。"}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"date": "2026-11-11", "approach": "原作者入口片段提取，未重新实现算法", "files": {}}
    generated, details = build_rigbd()
    files = [("rigbd.py", generated, details)]
    for attack in ["GTA", "UGBA", "DPGBA"]:
        generated, details = build_losplit(attack)
        files.append(("losplit_" + attack.lower() + ".py", generated, details))
    for name, generated, details in files:
        ast.parse(generated)
        source_path = ROOT / details["source"]
        source = source_path.read_text(encoding="utf-8-sig")
        details["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        details["segment_details"] = []
        for first, last in details["segments"]:
            details["segment_details"].append({"first_line": first, "last_line": last,
                "text_sha256": digest(lines(source, first, last))})
        details["statement_check"] = check_statement_preservation(source, details["segments"], generated)
        details["evaluation"]["statement_check"] = check_statement_preservation(
            source, details["evaluation"]["segments"], generated,
            function_name="evaluate", skip_diagnostic_prints=True)
        details["evaluation"]["segment_details"] = []
        for first, last in details["evaluation"]["segments"]:
            details["evaluation"]["segment_details"].append({"first_line": first, "last_line": last,
                "text_sha256": digest(lines(source, first, last))})
        if name == "rigbd.py":
            details["lifted_function_checks"] = {}
            original_functions = {}
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef):
                    original_functions[node.name] = node
            for node in ast.parse(generated).body:
                if isinstance(node, ast.FunctionDef) and node.name in {"sample_noise_all", "find_index"}:
                    preserved = ast.dump(node, include_attributes=False) == ast.dump(original_functions[node.name], include_attributes=False)
                    if not preserved:
                        raise ValueError("原函数体发生变化：" + node.name)
                    details["lifted_function_checks"][node.name] = {"ast_identical": True,
                        "text_sha256": digest(ast.get_source_segment(source, original_functions[node.name]))}
        path = DESTINATION / name
        path.write_text(generated, encoding="utf-8", newline="\n")
        details["generated_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][str(path.relative_to(ROOT)).replace("\\", "/")] = details
    output = ROOT / "paper" / "rigbd_losplit_segments_20261111.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("四个作者入口调用模块及片段清单已生成。")


if __name__ == "__main__":
    main()
