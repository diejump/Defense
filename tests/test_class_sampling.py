"""按类别选位置的纯参数与索引检查，不运行模型实验。"""
import unittest
import numpy as np
import torch
from src.class_sampling import allocate_uneven_counts, select_attached


class ClassSamplingTests(unittest.TestCase):
    def test_reproducible_exact_budget(self):
        first, weights = allocate_uneven_counts(81, [100] * 7, 10)
        second, _ = allocate_uneven_counts(81, [100] * 7, 10)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), 81)
        self.assertGreater(len(set(first.tolist())), 1)
        self.assertTrue((first > 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_different_seed(self):
        first, _ = allocate_uneven_counts(81, [100] * 7, 10)
        second, _ = allocate_uneven_counts(81, [100] * 7, 20)
        self.assertFalse(np.array_equal(first, second))

    def test_small_equal_budget_is_not_silently_uniform(self):
        counts, _ = allocate_uneven_counts(7, [100] * 7, 10)
        self.assertEqual(int(counts.sum()), 7)
        self.assertGreater(np.unique(counts).size, 1)

    def test_capacity_and_absent_class(self):
        counts, weights = allocate_uneven_counts(10, [0, 1, 2, 100], 10)
        self.assertEqual(int(counts.sum()), 10)
        self.assertTrue((counts <= np.array([0, 1, 2, 100])).all())
        self.assertEqual(int(counts[0]), 0)

    def test_unique_nonoverlapping_selection(self):
        pool = torch.arange(100)
        labels = pool % 7
        torch.manual_seed(10)
        first, remaining, details = select_attached(pool, labels, 30, 7, 10, 'uneven', 0)
        torch.manual_seed(10)
        second, _, _ = select_attached(pool, labels, 30, 7, 10, 'uneven', 0)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.unique().numel(), first.numel())
        self.assertEqual(first.numel() + details['filtered_target_count'], 30)
        self.assertEqual(remaining.numel(), 100 - first.numel())
        self.assertTrue(bool((labels[first] != 0).all()))
        self.assertFalse(bool(torch.isin(first, remaining).any()))
        self.assertEqual(details['poisoned_source_counts'], torch.bincount(labels[first], minlength=7).tolist())

    def test_random_selects_then_filters_without_refill(self):
        pool = torch.arange(100)
        labels = pool % 7
        torch.manual_seed(10)
        order = torch.randperm(100)
        torch.manual_seed(10)
        selected, available, details = select_attached(pool, labels, 30, 7, 10, 'random', 0)
        proposed = pool[order[:30]]
        expected = proposed[labels[proposed] != 0]
        self.assertTrue(torch.equal(selected, expected))
        self.assertLess(selected.numel(), 30)
        self.assertTrue(torch.equal(available, pool[order][~torch.isin(pool[order], expected)]))
        self.assertEqual(details['selected_nodes_before_filter'], proposed.tolist())
        filtered = proposed[labels[proposed] == 0]
        self.assertTrue(bool(torch.isin(filtered, available).all()))

    def test_all_selected_target_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            select_attached(torch.arange(10), torch.zeros(10, dtype=torch.long), 3, 2, 10, 'random', 0)

    def test_default_mode_is_random(self):
        from src.config import parser
        self.assertEqual(parser('检查默认模式').parse_args([]).class_sampling, 'random')


if __name__ == '__main__':
    unittest.main()
