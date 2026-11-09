"""独立生成固定样本；加--evaluate时训练指定骨架并计算无防御指标。"""
from src.config import parser
from src.attack import generate
from src.workflow import run_method


def main():
    p = parser("生成GTA、UGBA或DPGBA样本")
    p.add_argument("--evaluate", action="store_true", help="另训练无防御模型，计算该骨架的CA和ASR")
    args = p.parse_args()
    folder = generate(args)
    if args.evaluate:
        output = args.output_root / "attack_checks" / args.dataset / args.attack / ("seed" + str(args.seed)) / args.model
        run_method(folder, output, "none", args)


if __name__ == "__main__":
    main()
