"""枚举五节点以内的全部小图和训练位置，独立核对ABP定义。"""
from pathlib import Path
import networkx as nx
import torch
from tqdm.auto import tqdm
from src.abp import filter_edges
from src.common import save_json, file_hash


def main():
    checked = 0
    for count in range(1, 6):
        possible = []
        for sender in range(count):
            for receiver in range(sender + 1, count):
                possible.append((sender, receiver))
        for graph_bits in tqdm(range(1 << len(possible)), desc=str(count) + "节点图枚举", unit="图"):
            graph = nx.Graph()
            graph.add_nodes_from(range(count))
            pairs = []
            for index, edge in enumerate(possible):
                if graph_bits & (1 << index):
                    graph.add_edge(*edge)
                    pairs.append(edge)
                    pairs.append((edge[1], edge[0]))
            edges = torch.empty((2, 0), dtype=torch.long)
            if pairs:
                edges = torch.tensor(pairs).t().contiguous()
            components = {}
            for receiver in range(count):
                reduced = graph.copy()
                reduced.remove_node(receiver)
                components[receiver] = {}
                for component in nx.connected_components(reduced):
                    for node in component:
                        components[receiver][node] = component
            for train_bits in range(1, 1 << count):
                training = []
                for node in range(count):
                    if train_bits & (1 << node):
                        training.append(node)
                actual, _ = filter_edges(edges, torch.ones(edges.size(1)), count, torch.tensor(training))
                actual = set(map(tuple, actual.t().tolist()))
                expected = set()
                for sender, receiver in pairs:
                    if components[receiver][sender].intersection(training):
                        expected.add((sender, receiver))
                assert actual == expected, str((count, graph_bits, train_bits))
                checked += 1
    root = Path(__file__).resolve().parents[1]
    save_json(root / "outputs/abp_definition_check.json", {"status": "passed", "combinations": checked,
        "maximum_nodes": 5, "implementation_sha256": file_hash(root / "src/abp.py")})
    print("ABP定义一致性枚举通过：" + str(checked) + "组。")


if __name__ == "__main__":
    main()
