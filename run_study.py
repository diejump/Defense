"""依次执行数据集、生成方法、种子、骨架与防御的预定组合。"""
from copy import copy
from tqdm.auto import tqdm
from src.config import parser, DATASETS, MODELS, ATTACKS
from src.attack import generate
from src.common import save_json, arguments
from src.workflow import run_method
from src.summary import summarize


def run_backbone(folder, args, progress):
    rows = []
    for method in args.methods:
        output = args.output_root / "runs" / args.dataset / args.attack / ("seed" + str(args.seed)) / args.model / method
        try:
            record = run_method(folder, output, method, args)
            rows.append({"dataset": args.dataset, "attack": args.attack, "seed": args.seed,
                         "model": args.model, "method": method, "status": "complete", "metrics": record["metrics"]})
        except Exception as error:
            rows.append({"dataset": args.dataset, "attack": args.attack, "seed": args.seed,
                         "model": args.model, "method": method, "status": "failed", "error": str(error)})
            tqdm.write(method + "失败，详细原因已保留，继续其他组合：" + str(error))
        progress.update(1)
    return rows


def run_seed(args, progress):
    folder = generate(args)
    rows = []
    for backbone in args.models:
        current = copy(args)
        current.model = backbone
        rows.extend(run_backbone(folder, current, progress))
        save_json(args.output_root / "latest_batch_completed_runs.json", rows)
    return rows


def run_attack(args, progress):
    rows = []
    for seed in args.seeds:
        current = copy(args)
        current.seed = seed
        rows.extend(run_seed(current, progress))
    return rows


def run_dataset(args, progress):
    rows = []
    for attack in args.attacks:
        current = copy(args)
        current.attack = attack
        rows.extend(run_attack(current, progress))
    return rows


def main():
    p = parser("批量比较，默认执行Cora、Citeseer上的三骨架UGBA实验")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=["Cora", "Citeseer"], help="数据集列表")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS, help="骨架列表")
    p.add_argument("--attacks", nargs="+", choices=ATTACKS, default=["UGBA"], help="生成算法列表")
    p.add_argument("--seeds", nargs="+", type=int, default=[10], help="生成与训练种子列表")
    args = p.parse_args()
    save_json(args.output_root / "study_parameters.json", arguments(args))
    count = len(args.datasets) * len(args.attacks) * len(args.seeds) * len(args.models) * len(args.methods)
    progress = tqdm(total=count, desc="全部预定方法比较", unit="次")
    rows = []
    for dataset in args.datasets:
        current = copy(args)
        current.dataset = dataset
        rows.extend(run_dataset(current, progress))
    progress.close()
    all_rows = summarize(args.output_root)
    failures = [row for row in rows if row["status"] != "complete"]
    status = "incomplete" if failures else "complete"
    save_json(args.output_root / "completion.json", {"status": status, "batch_runs": len(rows), "runs": len(all_rows), "failures": failures})
    print("本批处理结束，失败数：" + str(len(failures)) + "，原始结果与汇总：" + str(args.output_root))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
