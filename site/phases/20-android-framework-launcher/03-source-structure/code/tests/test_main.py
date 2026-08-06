# 测试：ItemInfo 网格坐标校验逻辑
import unittest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import GridConfig, ItemInfo, load_and_validate


GRID = GridConfig(columns=5, rows=5)


class TestItemInfoValidate(unittest.TestCase):
    def test_valid_item(self):
        item = ItemInfo("App", 0, 0)
        ok, reason = item.validate(GRID)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_valid_widget(self):
        item = ItemInfo("Widget", 1, 2, span_x=2, span_y=2)
        ok, _ = item.validate(GRID)
        self.assertTrue(ok)

    def test_x_out_of_bounds(self):
        item = ItemInfo("Bad", 4, 0, span_x=2)
        ok, reason = item.validate(GRID)
        self.assertFalse(ok)
        self.assertIn("X 方向越界", reason)

    def test_y_out_of_bounds(self):
        item = ItemInfo("Bad", 0, 5)
        ok, reason = item.validate(GRID)
        self.assertFalse(ok)
        self.assertIn("Y 方向越界", reason)

    def test_negative_cell_x(self):
        item = ItemInfo("Bad", -1, 0)
        ok, reason = item.validate(GRID)
        self.assertFalse(ok)
        self.assertIn("cellX", reason)

    def test_span_less_than_one(self):
        item = ItemInfo("Bad", 0, 0, span_x=0)
        ok, reason = item.validate(GRID)
        self.assertFalse(ok)

    def test_item_exactly_at_boundary(self):
        # cellX=4, spanX=1 → 4+1=5 == columns, 合法
        item = ItemInfo("Edge", 4, 4)
        ok, _ = item.validate(GRID)
        self.assertTrue(ok)


class TestLoadAndValidate(unittest.TestCase):
    def test_separates_valid_invalid(self):
        items = [
            ItemInfo("Good", 0, 0),
            ItemInfo("Bad", 4, 0, span_x=2),
        ]
        valid, invalid = load_and_validate(items, GRID)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(valid[0].label, "Good")

    def test_all_valid(self):
        items = [ItemInfo(f"App{i}", i, 0) for i in range(5)]
        valid, invalid = load_and_validate(items, GRID)
        self.assertEqual(len(valid), 5)
        self.assertEqual(len(invalid), 0)

    def test_all_invalid(self):
        items = [ItemInfo("Bad", 10, 10) for _ in range(3)]
        valid, invalid = load_and_validate(items, GRID)
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(invalid), 3)
