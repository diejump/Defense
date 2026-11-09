"""汇总一个输出目录，不重新运行训练。"""
import argparse
from pathlib import Path
from src.config import ROOT
from src.summary import summarize


def main():
    p = argparse.ArgumentParser(description="汇总实际完成的实验")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="实验结果目录")
    args = p.parse_args()
    rows = summarize(args.output_root)
    print("汇总完成，记录数：" + str(len(rows)))


if __name__ == "__main__":
    main()
