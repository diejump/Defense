import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx

from models.construct import model_construct

from models.GCN import GCN
from models.GAT import GAT
from models.SAGE import GraphSage

import torch.optim as optim
import os

from collections import Counter


def calculate_std(features):
    std = torch.std(features, dim=1)
    return std


def prune_high_fluctuation_nodes(args, edge_index, edge_weights, x, device, large_graph=False):
    edge_index = edge_index[:, edge_weights > 0.0].to(device)
    edge_weights = edge_weights[edge_weights > 0.0].to(device)
    x = x.to(device)

    std = calculate_std(x)

    threshold = np.percentile(std.cpu().numpy(), 99)
    threshold = args.fluctuation_thrd

    high_std_nodes = torch.where(std >= threshold)[0]

    keep_edges_mask = ~(torch.isin(edge_index[0], high_std_nodes) | torch.isin(edge_index[1], high_std_nodes))

    updated_edge_index = edge_index[:, keep_edges_mask]
    updated_edge_weights = edge_weights[keep_edges_mask]

    return updated_edge_index, updated_edge_weights


# Calculate the entropy of each node feature vector
def calculate_entropy(features):
    features_min = features.min(dim=1, keepdim=True)[0]
    features_max = features.max(dim=1, keepdim=True)[0]
    features_normalized = (features - features_min) / (features_max - features_min + 1e-10)
    row_sums = features_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probabilities = features_normalized / row_sums

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy


# Prune irrelevant edges based on the Euclidean distance of node features
def prune_edge(args, edge_index, edge_weights, x, device, large_graph=True):
    edge_index = edge_index[:, edge_weights > 0.0].to(device)
    edge_weights = edge_weights[edge_weights > 0.0].to(device)
    x = x.to(device)

    if large_graph:
        edge_dists = torch.tensor([], dtype=torch.float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N / num_split)
        for i in range(num_split):
            if i == num_split - 1:
                edge_dist1 = torch.norm(
                    x[edge_index[0][N_split * i:]] - x[edge_index[1][N_split * i:]], dim=1
                ).cpu()
            else:
                edge_dist1 = torch.norm(
                    x[edge_index[0][N_split * i:N_split * (i + 1)]] - x[edge_index[1][N_split * i:N_split * (i + 1)]],
                    dim=1
                ).cpu()
            edge_dists = torch.cat([edge_dists, edge_dist1])
    else:
        edge_dists = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)

    keep_edges_mask = edge_dists <= args.dist_thrd
    updated_edge_index = edge_index[:, keep_edges_mask]
    updated_edge_weights = edge_weights[keep_edges_mask]

    return updated_edge_index, updated_edge_weights


def anomaly_embedding_edge(args, edge_index, edge_weights, x, device, large_graph=True):
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    x = x.to(device)
    if large_graph:
        edge_dists = torch.tensor([], dtype=torch.float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N / num_split)
        for i in range(num_split):
            if i == num_split - 1:
                edge_dist1 = torch.norm(
                    x[edge_index[0][N_split * i:]] - x[edge_index[1][N_split * i:]], dim=1
                ).cpu()
            else:
                edge_dist1 = torch.norm(
                    x[edge_index[0][N_split * i:N_split * (i + 1)]] - x[edge_index[1][N_split * i:N_split * (i + 1)]],
                    dim=1
                ).cpu()
            edge_dists = torch.cat([edge_dists, edge_dist1])
    else:
        edge_dists = torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1)

    high_dist_edges_mask = edge_dists <= args.dist_thrd

    return high_dist_edges_mask


def gnn_prune_low_entropy_nodes(args, poison_edge_index, poison_edge_weights, poison_x, poison_labels, device,
                                large_graph=True):
    poison_x = poison_x.to(device)
    poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0].to(device)
    poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0].to(device)
    poison_labels = poison_labels
    # if args.test_model=='GCN':
    # gnn_model = GCN(nfeat=poison_x.shape[1],
    #                         nhid=args.hidden,
    #                         nclass=poison_labels.max().item() + 1,
    #                         dropout=0.0, device=device).to(device)
    # elif args.test_model=='GAT':
    # gnn_model = GAT(nfeat=poison_x.shape[1],
    #                         nhid=args.hidden,
    #                         nclass=poison_labels.max().item() + 1,
    #                         heads=8,
    #                         dropout=0.0, device=device).to(device)
    # elif args.test_model=='GraphSage':
    gnn_model = GraphSage(nfeat=poison_x.shape[1],
                          nhid=args.hidden,
                          nclass=poison_labels.max().item() + 1,
                          dropout=0.0, device=device).to(device)
    # gnn_model = gnn_model.to(device)
    gnn_model.train()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = gnn_model(poison_x, poison_edge_index, poison_edge_weights)
        # loss = F.nll_loss(output[idx], ori_x[idx])
        loss = F.nll_loss(output[:len(poison_labels)], poison_labels)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('Epoch {}, training loss: {}'.format(epoch, loss.item()))

    gnn_model.eval()

    with torch.no_grad():
        output = gnn_model(poison_x, poison_edge_index, poison_edge_weights)
        probabilities = F.softmax(output, dim=1)
        # predicted_labels = probabilities.argmax(dim=1)

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    threshold = np.percentile(entropy.detach().cpu().numpy(), args.defense_thrd)
    # mask = entropy < threshold
    low_entropy_target_class_mask = (entropy < threshold)  # & target_class_mask

    keep_edges_mask = ~(low_entropy_target_class_mask[poison_edge_index[0]] | low_entropy_target_class_mask[
        poison_edge_index[1]])

    # Filter the edge_index by the edges we want to keep
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    # Filter the edge weights similarly
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]

    return filtered_poison_edge_index, filtered_poison_edge_weights, gnn_model, threshold  # ,most_common_class#,poison_labels


def prune_anomaly_nodes_edges(args, poison_edge_index, poison_edge_weights, poison_x, poison_labels, encoder_x, y_pred,
                              device, large_graph=True):
    poison_x = poison_x.to(device)
    poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0].to(device)
    poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0].to(device)
    poison_labels = poison_labels
    # if args.test_model == 'GCN':
    # gnn_model = GCN(nfeat=poison_x.shape[1],
    #                 nhid=args.hidden,
    #                 nclass=poison_labels.max().item() + 1,
    #                 dropout=0.0, device=device).to(device)
    # elif args.test_model == 'GAT':
    # gnn_model = GAT(nfeat=poison_x.shape[1],
    #                 nhid=args.hidden,
    #                 nclass=poison_labels.max().item() + 1,
    #                 heads=8,
    #                 dropout=0.0, device=device).to(device)
    # elif args.test_model == 'GraphSage':
    gnn_model = GraphSage(nfeat=poison_x.shape[1],
                          nhid=args.hidden,
                          nclass=poison_labels.max().item() + 1,
                          dropout=0.0, device=device).to(device)
    # gnn_model = gnn_model.to(device)
    gnn_model.train()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = gnn_model(poison_x, poison_edge_index, poison_edge_weights)
        # loss = F.nll_loss(output[idx], ori_x[idx])
        loss = F.nll_loss(output[:len(poison_labels)], poison_labels)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('Epoch {}, training loss: {}'.format(epoch, loss.item()))

    gnn_model.eval()

    with torch.no_grad():
        output = gnn_model(poison_x, poison_edge_index, poison_edge_weights)
        probabilities = F.softmax(output, dim=1)
        # predicted_labels = probabilities.argmax(dim=1)
    feature_std = calculate_std(poison_x)
    high_feature_std_mask = (feature_std > args.fluctuation_thrd)
    high_dist_edges_mask = anomaly_embedding_edge(args, poison_edge_index, poison_edge_weights, encoder_x, device)
    prediction_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    prediction_entropy_threshold = np.percentile(prediction_entropy.detach().cpu().numpy(), args.defense_thrd)
    # target_class_mask = (predicted_labels == most_common_class)
    low_prediction_entropy_mask = (prediction_entropy < prediction_entropy_threshold)  # & target_class_mask
    keep_edges_mask = ~(
            low_prediction_entropy_mask[poison_edge_index[0]] | high_feature_std_mask[poison_edge_index[0]] |
            low_prediction_entropy_mask[poison_edge_index[1]] | high_feature_std_mask[poison_edge_index[1]])

    keep_edges_mask = keep_edges_mask.to(device)
    high_dist_edges_mask = high_dist_edges_mask.to(device)
    final_edges_mask = keep_edges_mask | high_dist_edges_mask

    filtered_poison_edge_index = poison_edge_index[:, final_edges_mask]
    filtered_poison_edge_weights = poison_edge_weights[final_edges_mask]

    return filtered_poison_edge_index, filtered_poison_edge_weights, gnn_model, prediction_entropy_threshold
