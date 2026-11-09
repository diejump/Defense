"""普通分类训练：每轮训练一次，再用验证集选择模型。"""
from copy import deepcopy
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def fit_classifier(model, data, epochs, lr, weight_decay, title):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_accuracy = -1.0
    best_state = deepcopy(model.state_dict())
    history = []
    for epoch in tqdm(range(epochs), desc=title, unit="轮", leave=False):
        model.train()
        optimizer.zero_grad()
        output = model(data["x"], data["edges"], data["weights"])
        # 对原始分数与已经log_softmax的输出均适用。
        loss = F.cross_entropy(output[data["train"]], data["labels"][data["train"]])
        if not torch.isfinite(loss):
            raise RuntimeError(title + "出现非有限损失。")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(data["x"], data["edges"], data["weights"])
            accuracy = output[data["val"]].argmax(-1).eq(data["labels"][data["val"]]).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())
        history.append({"epoch": epoch + 1, "loss": loss.item(), "val_accuracy": accuracy})
    model.load_state_dict(best_state)
    model.eval()
    return history
