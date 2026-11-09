"""从真实完成的记录输出CSV、Markdown及多种子统计。"""
import csv
import json
import statistics
from pathlib import Path
from .common import save_json


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
                     "result": str(path)})
    if not rows:
        raise ValueError("此目录没有完成的实验记录。")
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 实验结果", "", "| 数据集 | 攻击 | 骨架 | 种子 | 方法 | CA | ASR | 建立模型秒数 |",
             "|---|---|---|---:|---|---:|---:|---:|"]
    groups = {}
    for row in rows:
        ca = format(row["CA"] * 100, ".2f") + "%"
        asr = "—" if row["ASR"] is None else format(row["ASR"] * 100, ".2f") + "%"
        lines.append("| " + " | ".join([str(row["dataset"]), str(row["attack"]), row["model"],
            str(row["seed"]), row["method"], ca, asr, format(row["model_build_seconds"], ".3f")]) + " |")
        key = str(row["dataset"]) + "/" + str(row["attack"]) + "/" + row["model"] + "/" + row["method"]
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
                  "失败或未完成记录数：" + str(len(failures))])
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(folder / "summary_statistics.json", statistics_result)
    save_json(folder / "failed_runs.json", failures)
    return rows
