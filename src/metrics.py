"""作者模型使用的准确率函数。"""
import torch


def accuracy(output, labels):
    predictions = output.argmax(dim=1)
    return predictions.eq(labels).double().mean()
