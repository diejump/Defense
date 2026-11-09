"""ABP：接收节点移除后，发送方仍能连接训练节点时才允许传递。"""
from collections import deque
import networkx as nx
import torch


def build_tree(edges, node_count):
    # 点双连通分解把图拆成若干部分；每部分用一个额外节点表示。
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    pairs = edges.t().detach().cpu().tolist()
    for sender, receiver in pairs:
        if sender != receiver:
            graph.add_edge(sender, receiver)
    tree = nx.Graph()
    tree.add_nodes_from(range(node_count))
    edge_part = {}
    for index, part in enumerate(nx.biconnected_component_edges(graph)):
        part_id = node_count + index
        for sender, receiver in part:
            tree.add_edge(part_id, sender)
            tree.add_edge(part_id, receiver)
            edge_part[(sender, receiver)] = part_id
            edge_part[(receiver, sender)] = part_id
    return tree, edge_part, pairs


def find_training_connections(tree, train_nodes):
    # 逐步移除没有训练节点的末端，剩余部分连接多个训练节点。
    training = set(train_nodes.detach().cpu().tolist())
    active = set(tree.nodes)
    degree = dict(tree.degree())
    queue = deque()
    for node in tree.nodes:
        if degree[node] <= 1 and node not in training:
            queue.append(node)
    while queue:
        node = queue.popleft()
        if node not in active:
            continue
        active.remove(node)
        for neighbor in tree.neighbors(node):
            if neighbor in active:
                degree[neighbor] -= 1
                if degree[neighbor] <= 1 and neighbor not in training:
                    queue.append(neighbor)
    # 从所有训练节点同时向外走，记住每部分朝向训练节点的连接。
    parent = {}
    queue = deque(training)
    for node in training:
        parent[node] = None
    while queue:
        node = queue.popleft()
        for neighbor in tree.neighbors(node):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return active, parent


def filter_edges(edges, weights, node_count, train_nodes):
    if train_nodes.numel() == 0:
        raise ValueError("ABP需要至少一个训练节点。")
    tree, edge_part, pairs = build_tree(edges, node_count)
    active, parent = find_training_connections(tree, train_nodes)
    retained = []
    for sender, receiver in pairs:
        keep = True
        if sender != receiver:
            part = edge_part[(sender, receiver)]
            if part not in parent:
                keep = False
            elif part not in active and receiver == parent[part]:
                # 关闭向训练信息来源一侧传入的方向，保留向外传递。
                keep = False
        retained.append(keep)
    mask = torch.tensor(retained, dtype=torch.bool, device=edges.device)
    return edges[:, mask], weights[mask]
