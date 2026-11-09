"""核对样本、骨架关系、权重和重载指标，保存审查记录。"""
import argparse
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.config import ROOT
from src.common import file_hash, save_json, code_hashes
from src.workflow import evaluate_checkpoint


def check_bundle(folder):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    for name, digest in metadata["sha256"].items():
        assert file_hash(folder / name) == digest, "样本散列不一致：" + name
    data = torch.load(folder / "training.pt", map_location="cpu")
    evaluation = torch.load(folder / "evaluation.pt", map_location="cpu")
    train = set(data["train"].tolist())
    val = set(data["val"].tolist())
    attached = set(evaluation["attached"].tolist())
    test = set(evaluation["clean_nodes"].tolist()) | set(evaluation["attack_nodes"].tolist())
    assert not train.intersection(val), "训练和验证重叠"
    assert not attached.intersection(test), "附着位置与测试重叠"
    assert attached.issubset(train), "附着位置未加入训练"
    assert data["labels"].numel() == evaluation["truth"].numel(), "原节点标签长度不一致"
    known = torch.zeros_like(data["labels"], dtype=torch.bool)
    known[data["train"]] = True
    known[data["val"]] = True
    assert torch.all(data["labels"][~known] == -1), "未知标签没有隐藏"
    assert torch.equal(data["labels"][data["val"]], evaluation["truth"][data["val"]])
    ordinary = torch.tensor(sorted(train - attached), dtype=torch.long)
    assert torch.equal(data["labels"][ordinary], evaluation["truth"][ordinary])
    assert torch.all(data["labels"][evaluation["attached"]] == evaluation["target"])
    assert torch.all(data["weights"] > 0)
    assert torch.all(evaluation["changed_weights"] > 0)
    assert torch.equal(evaluation["reference_x"], evaluation["changed_x"][:evaluation["reference_x"].size(0)])
    return {"bundle_id": metadata["bundle_id"], "counts": metadata["counts"], "status": "passed"}


def main():
    p = argparse.ArgumentParser(description="核对已完成的实验")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="结果目录")
    p.add_argument("--device", default="cuda:0", help="重载评估设备")
    p.add_argument("--expected-runs", type=int, default=None, help="要求完成的记录数")
    args = p.parse_args()
    current_hashes = code_hashes()
    bundles = {}
    results = []
    paths = sorted((args.output_root / "runs").rglob("result.json"))
    if args.expected_runs is not None:
        assert len(paths) == args.expected_runs, "完成记录数量不一致"
    for path in tqdm(paths, desc="核对记录与模型重载", unit="份"):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "complete", str(path)
        folder = Path(record["bundle"])
        if str(folder) not in bundles:
            bundles[str(folder)] = check_bundle(folder)
        assert bundles[str(folder)]["bundle_id"] == record["bundle_id"]
        checkpoint = path.parent / "model.pt"
        assert file_hash(checkpoint) == record["checkpoint_sha256"]
        saved = torch.load(checkpoint, map_location="cpu")
        for name, value in saved["state_dict"].items():
            assert torch.isfinite(value).all(), "非有限权重：" + name
        expected_layer = {"GCN": "GCNConv", "GAT": "GATConv", "GraphSAGE": "SAGEConv"}[record["model"]]
        final = record["details"]["final_model"]
        assert final["layers"][0]["type"] == expected_layer
        if record["method"] in ["rigbd", "losplit"]:
            expected_probe = "GCNConv" if record["method"] == "losplit" else expected_layer
            assert record["details"]["scoring_model"]["layers"][0]["type"] == expected_probe
        metrics = evaluate_checkpoint(folder, checkpoint, path.parent / "reload.json", args.device)
        for name in ["CA_hits", "CA_denominator", "ASR_hits", "ASR_denominator"]:
            assert metrics[name] == record["metrics"][name], "重载指标不同：" + str(path) + " " + name
        if record["method"] == "abp":
            assert metrics["old_prediction_changes"] == 0
            assert metrics["untriggered_target_predictions"] == metrics["ASR_hits"]
        differences = []
        for name, digest in record["code_sha256"].items():
            if current_hashes.get(name) != digest:
                differences.append(name)
        results.append({"result": str(path), "status": "passed", "CA": metrics["CA"], "ASR": metrics["ASR"],
                        "model": record["model"], "method": record["method"], "dataset": record["dataset"],
                        "code_files_updated_after_this_run": differences,
                        "verification": "以最终代码重载权重；整数分子、分母与原记录一致"})
    save_json(args.output_root / "verification.json", {"status": "passed", "runs": len(results),
              "bundles": bundles, "results": results,
              "code_update_note": "模型运行后补充数值检查、环境字段及生成标签范围检查；保存每次原代码散列，未改写原结果。"})
    print("全部已保存模型重载一致，检查记录数：" + str(len(results)))


if __name__ == "__main__":
    main()
