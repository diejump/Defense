"""无防御与ABP共用RIGBD的原骨架；此文件只在独立工作进程内导入。"""
from types import SimpleNamespace
from models.construct import model_construct
from .author_support import native_signature


def make_model(inputs, args, device):
    values = SimpleNamespace(dataset=args.dataset, hidden=args.hidden, dropout=args.dropout,
                             train_lr=args.lr, weight_decay=args.weight_decay)
    name = "GraphSage" if args.model == "GraphSAGE" else args.model
    return model_construct(values, name, inputs["data"], device).to(device)


def describe(model):
    return native_signature(model)
