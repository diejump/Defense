"""显式数量与百分比预算；生成前校验，读取已有数据不强制重新指定。"""
import math


def validate_budget(args):
    rate = args.rate
    number = args.vs_number
    if (rate is None) == (number is None):
        raise ValueError("生成样本前必须且只能指定一个数量参数：--rate 或 --vs_number。")
    if rate is not None:
        if not math.isfinite(rate) or not 0 < rate <= 100:
            raise ValueError("rate需为0到100之间的有限百分数，且大于0；例如3表示3%。")
    if number is not None:
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError("vs_number必须为正整数。")


def resolve_budget(args, original_train_nodes, available_nodes):
    validate_budget(args)
    if args.rate is not None:
        count = math.floor(original_train_nodes * args.rate / 100)
    else:
        count = args.vs_number
    if count < 1:
        raise ValueError("当前rate向下取整后数量为0，请增加rate或使用--vs_number。")
    if count >= available_nodes:
        raise ValueError("数量超出可用候选范围；需至少保留一个未选候选节点。")
    return count
