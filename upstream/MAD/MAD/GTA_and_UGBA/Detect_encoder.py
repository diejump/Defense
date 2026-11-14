from random import random
import torch 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from models.construct import model_construct

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix


class GCN_Encoder(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2, device=None, use_ln=False,
                 layer_norm_first=False):

        super(GCN_Encoder, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.use_ln = use_ln
        self.layer_norm_first = layer_norm_first
        self.body = GCN_body(nfeat, nhid, dropout, layer, device=None, use_ln=use_ln, layer_norm_first=layer_norm_first)
        self.fc = nn.Linear(nhid, nclass)

        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, edge_weight=None):
        x = self.body(x, edge_index, edge_weight)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_h(self, x, edge_index, edge_weight):
        self.eval()
        x = self.body(x, edge_index, edge_weight)
        return x

    def fit(self, features, edge_index, edge_weight, labels, train_iters=200, verbose=False):
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self._train_without_val(self.labels, train_iters, verbose)

    def _train_without_val(self, labels, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[:len(labels)], labels)
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def test(self, features, edge_index, edge_weight):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            probabilities = F.softmax(output, dim=1)
            # 获取每个节点的预测标签
            predicted_labels = probabilities.argmax(dim=1)
        return predicted_labels

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()  # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids


class GCN_body(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, layer=2, device=None, layer_norm_first=False, use_ln=False):
        super(GCN_body, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer - 1):
            self.convs.append(GCNConv(nhid, nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(torch.nn.LayerNorm(nhid))
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if (self.layer_norm_first):
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        return x

def prune_anomalous_edges_by_class(args, edge_index, edge_weights, all_anomalies, device, large_graph=False):
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    all_anomalies = torch.tensor(all_anomalies, dtype=torch.long, device=device)

    keep_edges_mask = ~(torch.isin(edge_index[0], all_anomalies) | torch.isin(edge_index[1], all_anomalies))

    updated_edge_index = edge_index[:, keep_edges_mask]
    updated_edge_weights = edge_weights[keep_edges_mask]

    return updated_edge_index, updated_edge_weights


def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def compute_distances_to_cluster_centers(embeddings, cluster_labels, cluster_centers):
    embeddings = embeddings.cpu()
    # cluster_labels = cluster_labels.cpu()
    embeddings = np.asarray(embeddings)
    cluster_labels = np.asarray(cluster_labels)
    cluster_centers = np.asarray(cluster_centers)

    distances = np.linalg.norm(embeddings - cluster_centers[cluster_labels], axis=1)

    return distances

import os
import time
from sklearn_extra import cluster
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# from kmeans_pytorch import kmeans, kmeans_predict

def find_mismatched_nodes(y_pred_x, y_pred):
    mismatched_indices = np.where(y_pred_x != y_pred)[0]
    return mismatched_indices

def self_loop_only(edge_index, num_nodes):
    self_loops = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    return torch.stack([self_loops, self_loops], dim=0)

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def pagerank_based_cluster_centers(x, y, train_edge_index):
    data = Data(x=x, edge_index=train_edge_index)


    graph = to_networkx(data, to_undirected=True)

    degree_centrality = nx.degree_centrality(graph)
    scores = np.array(list(degree_centrality.values()))

    cluster_labels = np.unique(y.cpu().numpy())
    cluster_centers = []

    for cluster in cluster_labels:
        cluster_indices = np.where(y.cpu().numpy() == cluster)[0]
        cluster_scores = scores[cluster_indices]
        max_index = cluster_indices[np.argmax(cluster_scores)]
        cluster_centers.append(x[max_index].cpu().numpy())

    return np.array(cluster_centers)

def cluster_encoder_detect(args, x, y, train_edge_index, device,idx_attach, idx_trigger, current_time):
    x = x.to(device)
    y = y.to(device)
    # train_edge_index = self_loop_only(train_edge_index, x.size(0)).to(device)
    train_edge_index = train_edge_index.to(device)
    gcn_encoder = GCN_Encoder(nfeat=x.shape[1],
                        nhid=args.hidden,
                        nclass=int(y.max() + 1),
                        dropout=args.dropout,
                        lr=args.train_lr,
                        weight_decay=args.weight_decay,
                        layer=2,
                        device=device).to(device)
    gcn_encoder.train()

    t_total = time.time()
    gcn_encoder.fit(x, train_edge_index, None, y, train_iters=args.epochs, verbose=True)
    print("Training encoder Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    nclass = np.unique(y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(x, train_edge_index, None).clone().detach()
    y_pred = gcn_encoder.test(x, train_edge_index, None)

    cluster_centers = pagerank_based_cluster_centers(encoder_x, y, train_edge_index)
    print("idx_attach:",idx_attach)
    y_pred = y_pred.cpu()

    distances = compute_distances_to_cluster_centers(encoder_x, y_pred, cluster_centers)

    return encoder_x, y_pred, gcn_encoder

def save_results_to_csv(args, mismatched_indices, y_pred_x, y_pred, current_time):
    output_dir = f"results/{args.dataset}/index"
    os.makedirs(output_dir, exist_ok=True)

    data = {
        "Node Index": range(len(y_pred_x)),
        "Original Cluster Label": y_pred_x,
        "Encoded Cluster Label": y_pred,
        "Mismatch": [1 if idx in mismatched_indices else 0 for idx in range(len(y_pred_x))]
    }
    df = pd.DataFrame(data)

    file_path = os.path.join(output_dir, f"{current_time}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved cluster results to {file_path}")
