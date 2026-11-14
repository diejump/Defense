"""所有入口共用的参数；既可以修改默认值，也可以使用命令行。"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Cora", "Citeseer", "Pubmed", "Physics", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE"]
ATTACKS = ["GTA", "UGBA", "DPGBA"]
METHODS = ["none", "abp", "rigbd", "losplit", "dshield", "mad", "prune"]
DEFAULT_METHODS = ["none", "abp", "rigbd", "losplit", "dshield", "mad", "prune"]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=DATASETS, default="Cora", help="数据集")
    p.add_argument("--model", choices=MODELS, default="GCN", help="最终分类骨架")
    p.add_argument("--attack", choices=ATTACKS, default="UGBA", help="生成方法")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS, help="当前作者实现比较方法")
    p.add_argument("--seed", type=int, default=10, help="生成与模型训练种子；没有隐含偏移")
    p.add_argument("--split-seed", type=int, default=10, help="固定节点划分种子")
    p.add_argument("--device", default="cuda:0", help="设备，例如cuda:0或cpu")
    p.add_argument("--data-root", type=Path, default=ROOT / "data", help="公开数据缓存")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="所有结果目录")
    p.add_argument("--bundle", type=Path, default=None, help="指定已保存的样本目录")
    p.add_argument("--resume", action="store_true", help="参数一致时继续已经保存的实验")
    p.add_argument("--hidden", type=int, default=64, help="隐藏宽度；GAT为每头宽度，固定8头")
    p.add_argument("--epochs", type=int, default=400, help="最终分类训练轮数")
    p.add_argument("--lr", type=float, default=0.01, help="分类学习率")
    p.add_argument("--weight-decay", type=float, default=0.0005, help="分类权重衰减")
    p.add_argument("--dropout", type=float, default=0.5, help="分类随机丢弃比例")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--rate", type=float, default=None, help="按注入前原训练节点数计算的百分比；3表示3%，与vs_number二选一；选后过滤目标类，不补选")
    budget.add_argument("--vs_number", "--vs-number", dest="vs_number", type=int, default=None, help="显式指定训练触发器选样预算，与rate二选一；选后过滤目标类，不补选")
    p.add_argument("--class-sampling", "--class_sampling", dest="class_sampling", choices=["random", "uneven"], default="random",
                   help="原始类别选样：random保留原流程，uneven按seed不均匀分配；两者均统一改为target")
    p.add_argument("--target", type=int, default=0, help="生成目标类别；作者方法从固定样本元数据读取同一目标类别")
    p.add_argument("--trigger-size", type=int, default=3, help="每组新增节点数")
    p.add_argument("--attack-epochs", type=int, default=400, help="生成器训练轮数")
    p.add_argument("--attack-hidden", type=int, default=32, help="生成器和辅助GCN隐藏宽度")
    p.add_argument("--attack-inner", type=int, default=None, help="UGBA默认5、DPGBA默认1")
    p.add_argument("--attack-lr", type=float, default=0.01, help="生成器学习率")
    p.add_argument("--attack-weight-decay", type=float, default=0.0005, help="生成器权重衰减")
    p.add_argument("--threshold", type=float, default=0.5, help="生成连接的阈值")
    p.add_argument("--target-weight", type=float, default=None, help="UGBA默认5、DPGBA默认1")
    p.add_argument("--homo-weight", type=float, default=100.0, help="UGBA相似性损失权重")
    p.add_argument("--homo-threshold", type=float, default=0.8, help="UGBA相似性阈值")
    p.add_argument("--gta-loss-factor", type=float, default=1.0, help="GTA损失系数")
    p.add_argument("--rec-epochs", type=int, default=100, help="DPGBA重建训练轮数")
    p.add_argument("--detector-updates", type=int, default=5, help="DPGBA判别器更新次数")
    p.add_argument("--outer-size", type=int, default=4096, help="DPGBA作者入口分支阈值，默认4096；原函数在抽样分支固定抽4096个")
    p.add_argument("--ood-weight", type=float, default=1.0, help="DPGBA分布项权重")
    p.add_argument("--class-weight", type=float, default=1.0, help="DPGBA目标类项权重")
    p.add_argument("--rigbd-repeats", "--rigbd-k", dest="rigbd_repeats", type=int, default=None, help="覆盖作者RIGBD的K；默认读取原始配置20")
    p.add_argument("--rigbd-drop-ratio", type=float, default=None, help="覆盖作者RIGBD丢边率；原始配置0.5")
    p.add_argument("--prune-threshold", type=float, default=None, help="覆盖LoSplit原Prune阈值；默认按原入口")
    p.add_argument("--author-overrides", type=Path, default=None, help="作者参数覆盖JSON，键为方法名，值为原参数名和值")
    return p
