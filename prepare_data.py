"""准备公开数据；Planetoid直接读取原始文件地址，避免网页跳转。"""
import argparse
from pathlib import Path
import shutil
from urllib.request import urlopen
from tqdm.auto import tqdm
from src.config import ROOT, DATASETS, parser
from src.common import setup, save_json
from src.data import load_clean


def main():
    p = argparse.ArgumentParser(description="下载并核对公开图数据")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=["Cora", "Citeseer"], help="需要准备的数据集")
    p.add_argument("--data-root", type=Path, default=ROOT / "data", help="数据缓存目录")
    options = p.parse_args()
    reports = {}
    for name in options.datasets:
        if name in ["Cora", "Citeseer", "Pubmed"]:
            raw = options.data_root / name / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            suffixes = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
            for suffix in tqdm(suffixes, desc=name + "原始文件", unit="个"):
                filename = "ind." + name.lower() + "." + suffix
                target = raw / filename
                if target.exists() and target.stat().st_size > 0:
                    continue
                url = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/" + filename
                # 先写临时文件，下载完成后再交给PyG读取。
                temporary = target.with_suffix(target.suffix + ".download")
                with urlopen(url, timeout=45) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                temporary.replace(target)
        args = parser("数据核对").parse_args([])
        args.dataset = name
        args.data_root = options.data_root
        data, info = load_clean(args, setup(10, "cpu"))
        reports[name] = info["source"]
        print(name + "准备完成，节点数：" + str(data["x"].size(0)))
    save_json(options.data_root / "sources.json", reports)


if __name__ == "__main__":
    main()
