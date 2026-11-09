from tqdm.auto import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from upstream.DShield.NodeClassificationTasks import utils
from copy import deepcopy
from upstream.DShield.NodeClassificationTasks.models.metric import accuracy

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


class MLP(nn.Module):

    def __init__(self, n_feat, n_hid, n_class,
                 dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):

        super(MLP, self).__init__()

        assert device is not None, '请明确指定计算设备。'
        self.device = device
        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.n_class = n_class
        self.dropout = dropout
        self.lr = lr

        self.weight_decay = weight_decay
        self.body = nn.Sequential(nn.Linear(n_feat, n_hid),
                                  nn.ReLU(),
                                  nn.Linear(n_hid, n_hid))
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x):
        return self.body(x)

    def fit(self, features, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, **kwargs):
        features = features.to(self.device)
        labels = labels.to(self.device)
        if idx_val is None:
            self._train_without_val(features, labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(features, labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, features, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in tqdm(range(train_iters), desc='辅助感知机模型训练', unit='轮', dynamic_ncols=True, leave=False, mininterval=0.5):
            optimizer.zero_grad()
            output = self.forward(features)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('轮数 {}，训练损失：{}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(features)
        return output

    def _train_with_val(self, features, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('开始训练多层感知机模型')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in tqdm(range(train_iters), desc='辅助感知机模型训练', unit='轮', dynamic_ncols=True, leave=False, mininterval=0.5):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(features)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            if verbose and i % 10 == 0:
                print('轮数 {}，训练损失：{}'.format(i, loss_train.item()))
                logger.info('验证准确率：{:.4f}'.format(acc_val))

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

        if verbose:
            print('根据验证集表现恢复最佳模型参数')
        self.load_state_dict(weights)

    @torch.no_grad()
    def test(self, features, labels, idx_test):
        self.eval()
        output = self.forward(features)
        _ = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
