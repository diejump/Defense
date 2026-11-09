"""从作者defense.py逐字提取的丢边采样函数。"""
import torch
from torch.distributions import Bernoulli
drop_ratio = 0.5

def sample_noise_all(edge_index, edge_weight, device):
    # Ensure inputs are on the correct device
    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    else:
        edge_weight = edge_weight.to(device)

    # Generate mask for edge dropping
    drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()

    # Apply mask to edges
    noisy_edge_index = edge_index[:, drop_mask]
    noisy_edge_weight = edge_weight[drop_mask]

    # Get node degrees
    node_degrees = torch.zeros(edge_index.max() + 1, device=device)
    node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))
    # print('degree', node_degrees)

    # Restore edges for isolated nodes
    isolated_nodes = node_degrees == 0
    # print('isolated_nodes', isolated_nodes)
    if isolated_nodes.any():
        potential_restore_edges = isolated_nodes[edge_index[0]]
        # print('potential_restore_edges', potential_restore_edges)
        restore_edges = edge_index[:, potential_restore_edges]
        noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
        restored_weights = torch.ones(restore_edges.size(1), device=device)
        noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

    return noisy_edge_index, noisy_edge_weight
