"""仅用既有Cora样本检查DPGBA分支原模型接口；不产生方法效果指标。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_case(args):
    import torch
    from src.common import setup, save_json
    from src.author_paths import activate
    from src.author_inputs import load_author_inputs
    from src.author_parameters import get_parameters
    from src.author_support import pack_model, unpack_model, logits_of
    from src.config import parser
    device = setup(10, "cpu")
    activate(args.method, "DPGBA")
    from models.construct import model_construct
    inputs = load_author_inputs(args.bundle, None, device)
    if inputs["metadata"]["settings"]["dataset"] != "Cora":
        raise ValueError("此检查仅允许Cora。")
    config = parser("原模型接口检查").parse_args([])
    config.attack, config.dataset, config.model, config.device = "DPGBA", "Cora", args.model, "cpu"
    parameters, source = get_parameters(args.method, config, inputs, args.output.parent)
    model = model_construct(parameters, parameters.test_model, inputs["data"], device).to(device)
    train = inputs["known_nodes"]
    report = {"dataset": "Cora", "method": args.method, "model": args.model, "branch": "DPGBA",
        "purpose": "已有Cora样本只检查原模型调用；未生成或评价DPGBA样本；finetune2名单仅为已知接口输入",
        "training_epochs_for_interface": 1, "checks": {}}
    try:
        model.fit(inputs["poison_x"], inputs["poison_edge_index"], inputs["poison_edge_weights"],
                  inputs["poison_labels"], train, inputs["idx_val"], train_iters=1, verbose=False)
        report["checks"]["native_fit"] = "passed"
        state = pack_model(model, vars(parameters))
        restored = unpack_model(state, device)
        model.eval()
        restored.eval()
        with torch.no_grad():
            before = logits_of(model(inputs["poison_x"], inputs["poison_edge_index"], inputs["poison_edge_weights"]))
            after = logits_of(restored(inputs["poison_x"], inputs["poison_edge_index"], inputs["poison_edge_weights"]))
        if not torch.equal(before, after):
            raise ValueError("原模型保存重载前向不一致。")
        report["checks"]["native_roundtrip"] = "passed"
        if args.method == "losplit":
            model.fit(inputs["poison_x"], inputs["poison_edge_index"], inputs["poison_edge_weights"],
                inputs["poison_labels"], train, inputs["idx_val"], train_iters=1, verbose=False,
                finetune2=True, attach=inputs["idx_attach"], clean=inputs["idx_train"],
                target_label=0, gamma=parameters.gamma)
            report["checks"]["native_finetune2"] = "passed"
        report["status"] = "passed"
    except Exception as error:
        report.update(status="source_error", error=type(error).__name__ + ": " + str(error))
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Cora上的原DPGBA模型分支接口检查")
    p.add_argument("--method", choices=["losplit", "mad"], default=None)
    p.add_argument("--model", choices=["GCN", "GAT", "GraphSAGE"], default="GCN")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/author_source_20261111/dpgba_model_contracts")
    args = p.parse_args()
    if args.method is not None:
        check_case(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for method in ["losplit", "mad"]:
        for model in ["GCN", "GAT", "GraphSAGE"]:
            destination = args.output / (method + "_" + model + ".json")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--method", method, "--model", model,
                "--bundle", str(args.bundle.resolve()), "--output", str(destination)], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
