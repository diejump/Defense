"""核对GTA的作者标签条件及旧生成配置隔离，不运行模型训练。"""
import unittest
from types import SimpleNamespace
import torch
from src.attack import attack_labels_for_generator, generation_parameters
from src.config import parser


class GTALabelInputTests(unittest.TestCase):
    def test_gta_receives_all_original_labels_without_mutating_truth(self):
        truth = torch.tensor([0, 1, 2, 0, 2])
        data = {"labels": torch.tensor([0, -1, 2, -1, -1]),
                "train": torch.tensor([0, 2]), "classes": 3}
        labels = attack_labels_for_generator(data, truth, SimpleNamespace(attack="GTA", target=0))
        self.assertTrue(torch.equal(labels, truth))
        self.assertEqual((labels == 0).nonzero().flatten().tolist(), [0, 3])
        labels[3] = 1
        self.assertEqual(truth[3].item(), 0)

    def test_other_attacks_keep_existing_label_scope(self):
        truth = torch.tensor([0, 1, 2, 0, 2])
        data = {"labels": torch.tensor([0, -1, 2, -1, -1]),
                "train": torch.tensor([0, 2]), "classes": 3}
        for attack in ["UGBA", "DPGBA"]:
            labels = attack_labels_for_generator(data, truth, SimpleNamespace(attack=attack, target=0))
            self.assertEqual(labels.tolist(), [0, -1, 2, -1, -1])

    def test_gta_settings_distinguish_old_restricted_label_bundles(self):
        args = parser("标签条件检查").parse_args(["--attack", "GTA", "--rate", "3"])
        settings = generation_parameters(args)
        self.assertEqual(settings["generator_label_scope"], "dshield_original_full_node_labels")
        previous_settings = settings.copy()
        previous_settings.pop("generator_label_scope")
        self.assertNotEqual(previous_settings, settings)


if __name__ == "__main__":
    unittest.main()
