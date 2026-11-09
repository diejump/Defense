"""所有入口共用的参数；既可以修改默认值，也可以使用命令行。"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Cora", "Citeseer", "Pubmed", "Physics", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE"]
ATTACKS = ["GTA", "UGBA", "DPGBA"]
METHODS = ["none", "abp", "rigbd", "losplit", "dshield"]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=DATASETS, default="Cora", help="数据集")
    p.add_argument("--model", choices=MODELS, default="GCN", help="最终分类骨架")
    p.add_argument("--attack", choices=ATTACKS, default="UGBA", help="生成方法")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS, help="需要执行的方法")
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
    p.add_argument("--poison-fraction", type=float, default=0.01, help="训练附着数量占原节点总数的比例")
    p.add_argument("--poison-count", type=int, default=None, help="显式附着数量；设置后优先于比例")
    p.add_argument("--target", type=int, default=0, help="生成目标类别，防御不使用该参数")
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
    p.add_argument("--outer-size", type=int, default=1024, help="DPGBA外层样本数上限")
    p.add_argument("--ood-weight", type=float, default=1.0, help="DPGBA分布项权重")
    p.add_argument("--class-weight", type=float, default=1.0, help="DPGBA目标类项权重")
    p.add_argument("--rigbd-probe-epochs", type=int, default=1000, help="RIGBD同骨架评分模型训练轮数")
    p.add_argument("--rigbd-repeats", type=int, default=20, help="RIGBD随机丢边次数")
    p.add_argument("--rigbd-loss", choices=["source", "paper"], default="source", help="源码式概率恢复目标或论文式对数目标")
    p.add_argument("--losplit-epochs", type=int, default=None, help="早期GCN轮数；默认采用作者数据集配置")
    p.add_argument("--losplit-lr", type=float, default=None, help="早期GCN学习率；默认采用作者配置")
    p.add_argument("--losplit-gamma", type=float, default=None, help="恢复系数；默认采用作者配置")
    p.add_argument("--losplit-loss", choices=["source", "paper"], default="source", help="源码等价恢复或论文公式恢复")
    p.add_argument("--dshield-pretrain", type=int, default=400, help="DShield自监督轮数")
    p.add_argument("--dshield-finetune", type=int, default=400, help="DShield关系验证轮数")
    p.add_argument("--dshield-rounds", type=int, default=1, help="DShield防御轮数")
    p.add_argument("--kappa1", type=float, default=5.0, help="DShield第一项权重")
    p.add_argument("--kappa2", type=float, default=5.0, help="DShield第二项权重")
    p.add_argument("--kappa3", type=float, default=0.1, help="DShield恢复权重")
    p.add_argument("--dshield-threshold", type=float, default=2.5, help="DShield筛选阈值")
    p.add_argument("--dshield-edge-drop", type=float, default=0.2, help="DShield丢边比例")
    p.add_argument("--dshield-feature-drop", type=float, default=0.2, help="DShield特征丢弃比例")
    p.add_argument("--dshield-temperature", type=float, default=0.9, help="DShield温度")
    p.add_argument("--dshield-balance", type=float, default=0.5, help="DShield验证选择权重")
    return p
