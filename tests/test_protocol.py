"""检查骨架关系、ABP定义及恢复目标，不进行数据集训练。"""
import unittest
from copy import copy
import torch
import networkx as nx
from unittest.mock import patch
from src.config import parser
from src.models import make_model, describe
from src.abp import filter_edges
from src.losplit import parameters, recover
from src.training import fit_classifier
from src.common import setup
from src.attack import labels_for_generator, make_generator


def small_data():
    torch.manual_seed(10)
    return {"x": torch.randn(7, 4),
            "edges": torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2],
                                   [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 2, 4]]),
            "weights": torch.ones(12), "labels": torch.tensor([0, 1, 0, 1, 0, 1, 0]),
            "train": torch.tensor([0, 1, 2, 3]), "val": torch.tensor([4, 5]), "classes": 2}


class ProtocolTests(unittest.TestCase):
    def test_three_author_backbones_are_available(self):
        data = small_data()
        args = parser("检查").parse_args([])
        args.hidden = 32
        for backbone in ["GCN", "GAT", "GraphSAGE"]:
            args.model = backbone
            for family in ["rigbd", "losplit", "dshield"]:
                model = make_model(data, args, "cpu", family)
                output = model(data["x"], data["edges"], data["weights"])
                self.assertEqual(tuple(output.shape), (7, 2))
                self.assertTrue(torch.isfinite(output).all())

    def test_rigbd_probe_keeps_selected_backbone(self):
        data = small_data()
        args = parser("检查").parse_args([])
        args.hidden = 32
        expected = {"GCN": "GCNConv", "GAT": "GATConv", "GraphSAGE": "SAGEConv"}
        for backbone in expected:
            args.model = backbone
            probe = describe(make_model(data, args, "cpu", "rigbd", scoring=True))
            final = describe(make_model(data, args, "cpu", "rigbd"))
            self.assertEqual(probe["layers"][0]["type"], expected[backbone])
            self.assertEqual(final["layers"][0]["type"], expected[backbone])
            if backbone == "GCN":
                self.assertFalse(probe["layers"][0]["add_self_loops"])
                self.assertTrue(final["layers"][0]["add_self_loops"])
            if backbone == "GAT":
                self.assertTrue(probe["layers"][0]["add_self_loops"])

    def test_abp_matches_remove_receiver_definition(self):
        data = small_data()
        actual, _, = filter_edges(data["edges"], data["weights"], 7, data["train"][:2])
        actual = set(map(tuple, actual.t().tolist()))
        graph = nx.Graph()
        graph.add_nodes_from(range(7))
        graph.add_edges_from(data["edges"].t().tolist())
        expected = set()
        for sender, receiver in data["edges"].t().tolist():
            reduced = graph.copy()
            reduced.remove_node(receiver)
            component = nx.node_connected_component(reduced, sender)
            if component.intersection({0, 1}):
                expected.add((sender, receiver))
        self.assertEqual(actual, expected)

    def test_source_losplit_keeps_observed_labels_unchanged(self):
        data = small_data()
        args = parser("检查").parse_args([])
        args.hidden = 32
        args.epochs = 2
        before = data["labels"].clone()
        model = make_model(data, args, "cpu", "losplit")
        history = recover(model, data, torch.tensor([0, 1]), torch.tensor([2, 3]), 0, 0.2, args)
        self.assertEqual(len(history), 2)
        self.assertTrue(torch.equal(before, data["labels"]))

    def test_losplit_dataset_parameters(self):
        args = parser("检查").parse_args([])
        self.assertEqual(parameters(args)[:3], (15, 0.01, 0.2))
        args.dataset = "Citeseer"
        self.assertEqual(parameters(args)[:3], (10, 0.01, 0.45))

    def test_classifier_uses_only_training_and_validation_labels(self):
        data = small_data()
        data["labels"][6] = -1
        args = parser("检查").parse_args([])
        model = make_model(data, args, "cpu")
        history = fit_classifier(model, data, 2, 0.01, 0.0, "普通训练核查")
        self.assertEqual(len(history), 2)

    def test_ugba_training_does_not_use_unknown_node_truth(self):
        args = parser("检查").parse_args([])
        args.attack_epochs = 2
        args.attack_inner = 1
        setup(8, "cpu")
        x = torch.rand(64, 8)
        x = x / x.sum(1, keepdim=True)
        ring = torch.arange(64)
        edges = torch.stack([ring, (ring + 1) % 64])
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        truth = ring % 2
        data = {"labels": truth, "train": torch.arange(16), "classes": 2}
        observed = labels_for_generator(data, 0)
        states = []
        for labels in [truth, observed]:
            setup(12, "cpu")
            generator, _ = make_generator(args, torch.device("cpu"), 45)
            generator.fit(x, edges, torch.ones(edges.size(1)), labels.clone(), data["train"],
                          torch.tensor([16, 17, 18]), torch.arange(19, 64))
            states.append(generator.trojan.state_dict())
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    def test_dshield_keeps_original_node_label_length(self):
        from src.dshield import train
        data = small_data()
        data["labels"] = data["labels"][:6]
        data["labels"][5] = -1
        data["val"] = torch.tensor([4])
        args = parser("检查").parse_args([])
        args.hidden = 32
        for backbone in ["GCN", "GAT", "GraphSAGE"]:
            args.model = backbone
            with patch("src.dshield.fit_classifier"):
                with patch("src.dshield.dshield") as call:
                    probe = make_model(data, args, "cpu", "dshield")
                    call.return_value = (probe, torch.tensor([], dtype=torch.long))
                    train(data, args, torch.device("cpu"))
                    passed_labels = call.call_args.args[4]
                    self.assertEqual(passed_labels.numel(), 6)
                    self.assertTrue((passed_labels >= 0).all())
                    expected = "GraphSage" if backbone == "GraphSAGE" else backbone
                    self.assertEqual(call.call_args.kwargs["arch"], expected)

    def test_losplit_early_model_is_gcn_for_all_final_backbones(self):
        from src.losplit import train
        data = small_data()
        args = parser("检查").parse_args([])
        args.hidden = 32
        args.epochs = 1
        for backbone in ["GCN", "GAT", "GraphSAGE"]:
            args.model = backbone
            with patch("src.losplit.LoSplit.fit") as fit:
                fit.return_value = ([torch.tensor([2, 3])], [torch.tensor([0, 1])], [1.0], [0])
                model, details = train(data, args, torch.device("cpu"))
            self.assertEqual(details["scoring_model"]["layers"][0]["type"], "GCNConv")
            self.assertIn("LoSplit.models", details["final_model"]["class"])


if __name__ == "__main__":
    unittest.main()
