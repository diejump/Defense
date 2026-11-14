"""读取固定样本，分别训练、保存和评估所选方法。"""
from src.config import parser
from src.attack import bundle_directory
from src.workflow import run_method, evaluate_checkpoint
from tqdm.auto import tqdm


def main():
    p = parser("运行ABP、RIGBD、LoSplit、DShield、MAD与原Prune")
    p.add_argument("--checkpoint", type=str, default=None, help="仅评估已经保存的model.pt")
    args = p.parse_args()
    folder = bundle_directory(args)
    if args.checkpoint is not None:
        output = args.output_root / "reevaluation.json"
        result = evaluate_checkpoint(folder, args.checkpoint, output, args.device)
        print("重载评估完成：" + str(result))
        return
    failures = []
    for method in args.methods:
        output = args.output_root / "runs" / args.dataset / args.attack / ("seed" + str(args.seed)) / args.model / method
        try:
            run_method(folder, output, method, args)
        except Exception as error:
            failures.append(method)
            tqdm.write(method + "运行失败，保留错误记录并继续其余方法：" + str(error))
    if failures:
        print("未完成的方法：" + "、".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
