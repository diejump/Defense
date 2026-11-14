"""对当前作者流程结果核对模型、样本和原生重载指标。"""
import argparse
import json
import math
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.config import ROOT
from src.common import file_hash, save_json
from src.workflow import evaluate_checkpoint


def main():
    p = argparse.ArgumentParser(description="核对当前作者流程的保存与重载")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="需核对的结果目录")
    p.add_argument("--device", default=None, help="默认用各记录的原设备，精确复核时保持相同")
    args = p.parse_args()
    checked, failures, historical = [], [], []
    paths = sorted((args.output_root / "runs").rglob("result.json"))
    for path in tqdm(paths, desc="核对当前作者流程", unit="份"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("protocol") != "author_original_20261111":
            historical.append(str(path))
            continue
        if record["status"] != "complete":
            failures.append({"record": str(path), "error": record.get("error", record["status"])})
            continue
        checkpoint = path.parent / "model.pt"
        try:
            if file_hash(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("模型文件散列变化。")
            bundle = Path(record["bundle"])
            info = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            if info["bundle_id"] != record["bundle_id"]:
                raise ValueError("样本包ID变化。")
            for name, digest in info.get("sha256", {}).items():
                if file_hash(bundle / name) != digest:
                    raise ValueError("样本文件散列变化：" + name)
            saved = torch.load(checkpoint, map_location="cpu")
            expected = 5 if record["method"] == "mad" else 1
            if len(saved["models"]) != expected or len(saved["evaluation_frames"]) != expected:
                raise ValueError("原内部模型数量不一致。")
            for model in saved["models"]:
                for name, tensor in model["state_dict"].items():
                    if not torch.isfinite(tensor).all():
                        raise ValueError("模型参数出现非有限值：" + name)
            device = args.device or record["arguments"]["device"]
            metrics = evaluate_checkpoint(bundle, checkpoint, path.parent / "reload.json", device)
            for name in ["CA", "ASR"]:
                before, after = record["metrics"][name], metrics[name]
                if before is None and after is None:
                    continue
                if not math.isclose(before, after, rel_tol=0, abs_tol=1e-7):
                    raise ValueError("重载指标不同：" + name + " " + str(before) + " / " + str(after))
            checked.append({"record": str(path), "status": "passed", "CA": metrics["CA"], "ASR": metrics["ASR"]})
        except Exception as error:
            failures.append({"record": str(path), "error": str(error)})
    save_json(args.output_root / "verification.json", {"status": "passed" if not failures else "incomplete",
        "verified": checked, "failed_or_unfinished": failures, "historical_skipped": historical})
    print("重载核对通过：" + str(len(checked)) + "，失败或未完成：" + str(len(failures)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
