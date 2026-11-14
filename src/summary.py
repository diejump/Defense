"""从真实完成的记录输出CSV、Markdown及多种子统计。"""
import csv
import json
import statistics
import hashlib
from pathlib import Path
from .common import save_json


def protocol_signature(record):
    # 只去掉种子与调度信息，数学参数、源码和环境不同的运行分开统计。
    values = record["arguments"].copy()
    ignored = ["seed", "seeds", "datasets", "models", "attacks", "methods", "output_root", "data_root",
               "bundle", "resume", "checkpoint", "evaluate"]
    for name in ignored:
        values.pop(name, None)
    payload = {"parameters": values, "code": record.get("code_sha256"), "environment": record.get("environment")}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def summarize(folder):
    folder = Path(folder)
    rows = []
    failures = []
    for path in sorted((folder / "runs").rglob("result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "complete":
            failures.append({"file": str(path), "status": record["status"], "error": record.get("error")})
            continue
        metrics = record["metrics"]
        rows.append({"dataset": record["dataset"], "attack": record["attack"], "model": record["model"],
                     "seed": record["seed"], "method": record["method"], "CA": metrics["CA"],
                     "ASR": metrics["ASR"], "model_build_seconds": metrics["model_build_seconds"],
                     "protocol": record.get("protocol", "historical"),
                     "CA_scope": metrics.get("CA_scope", "历史评估口径"),
                     "ASR_scope": metrics.get("ASR_scope", "历史评估口径"),
                     "label_condition": record.get("details", {}).get("label_condition", "历史标签条件"),
                     "internal_model_count": metrics.get("internal_model_count", 1),
                     "protocol_id": protocol_signature(record), "result": str(path)})
    if not rows:
        raise ValueError("此目录没有完成的实验记录。")
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 实验结果", "", "评估图、标签条件、内部模型数量见同目录CSV及各result.json；同名指标按作者原定义保存。",
             "MAD使用全原节点标签训练辅助模型，含五次最终模型训练；DShield的原ASR包含原目标类节点。", "",
             "| 数据集 | 攻击 | 骨架 | 种子 | 方法 | CA | ASR | 建立模型秒数 | 内部模型数 |",
             "|---|---|---|---:|---|---:|---:|---:|---:|"]
    groups = {}
    for row in rows:
        ca = format(row["CA"] * 100, ".2f") + "%"
        asr = "—" if row["ASR"] is None else format(row["ASR"] * 100, ".2f") + "%"
        lines.append("| " + " | ".join([str(row["dataset"]), str(row["attack"]), row["model"],
            str(row["seed"]), row["method"], ca, asr, format(row["model_build_seconds"], ".3f"), str(row["internal_model_count"])]) + " |")
        key = str(row["dataset"]) + "/" + str(row["attack"]) + "/" + row["model"] + "/" + row["method"]
        key += "/" + row["protocol_id"]
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    statistics_result = {}
    for key, values in groups.items():
        result = {"seeds": [], "n": len(values)}
        for row in values:
            result["seeds"].append(row["seed"])
        for field in ["CA", "ASR", "model_build_seconds"]:
            numbers = []
            for row in values:
                if row[field] is not None:
                    numbers.append(row[field])
            if numbers:
                result[field + "_mean"] = statistics.mean(numbers)
                result[field + "_sample_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        statistics_result[key] = result
    lines.extend(["", "CA、ASR按0—1保存。单种子不计算标准差，秒数为本次完整方法训练时间。",
                  "统计按数学参数、源码和环境分组；CSV中的protocol_id不同则不合并均值。",
                  "失败或未完成记录数：" + str(len(failures))])
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(folder / "summary_statistics.json", statistics_result)
    save_json(folder / "failed_runs.json", failures)
    return rows
