"""预算参数检查，不训练模型或生成样本。"""
import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch
from types import SimpleNamespace
from src.poison_budget import validate_budget, resolve_budget
from src.config import parser


class PoisonBudgetTests(unittest.TestCase):
    def test_parser_has_no_implicit_budget(self):
        args = parser('预算测试').parse_args([])
        self.assertIsNone(args.rate)
        self.assertIsNone(args.vs_number)

    def test_parser_rejects_two_explicit_budgets(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser('预算测试').parse_args(['--rate', '3', '--vs_number', '81'])

    def test_missing_budget_stops_before_loading_data(self):
        from src.attack import generate
        args = parser('预算测试').parse_args([])
        with patch('src.attack.load_clean') as loader:
            with self.assertRaises(ValueError):
                generate(args)
            loader.assert_not_called()

    def test_requires_exactly_one_budget(self):
        for rate, number in [(None, None), (3.0, 81)]:
            with self.assertRaises(ValueError):
                validate_budget(SimpleNamespace(rate=rate, vs_number=number))

    def test_cora_three_percent(self):
        count = resolve_budget(SimpleNamespace(rate=3.0, vs_number=None), 541, 1355)
        self.assertEqual(count, 16)

    def test_explicit_count(self):
        count = resolve_budget(SimpleNamespace(rate=None, vs_number=81), 2708, 1355)
        self.assertEqual(count, 81)

    def test_invalid_rates(self):
        for value in [0.0, -1.0, 101.0, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                validate_budget(SimpleNamespace(rate=value, vs_number=None))

    def test_invalid_counts(self):
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                validate_budget(SimpleNamespace(rate=None, vs_number=value))

    def test_zero_after_floor_and_insufficient_pool(self):
        for args in [SimpleNamespace(rate=0.0001, vs_number=None), SimpleNamespace(rate=None, vs_number=1355)]:
            with self.assertRaises(ValueError):
                resolve_budget(args, 541, 1355)


if __name__ == '__main__':
    unittest.main()
