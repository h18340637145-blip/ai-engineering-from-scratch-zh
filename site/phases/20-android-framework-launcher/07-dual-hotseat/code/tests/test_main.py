import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import HotseatConfig, rank_to_cell, cell_to_rank


DUAL = HotseatConfig(rows=2, cols=5)
SINGLE = HotseatConfig(rows=1, cols=5)


class TestRankToCell(unittest.TestCase):
    def test_single_row_all_row_zero(self):
        for rank in range(5):
            row, col = rank_to_cell(rank, SINGLE)
            self.assertEqual(row, 0)
            self.assertEqual(col, rank)

    def test_dual_row_boundary(self):
        row, col = rank_to_cell(5, DUAL)
        self.assertEqual(row, 1)
        self.assertEqual(col, 0)

    def test_dual_last_item(self):
        row, col = rank_to_cell(9, DUAL)
        self.assertEqual(row, 1)
        self.assertEqual(col, 4)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            rank_to_cell(10, DUAL)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            rank_to_cell(-1, DUAL)


class TestRoundTrip(unittest.TestCase):
    def test_rank_cell_rank_roundtrip(self):
        for rank in range(DUAL.total):
            row, col = rank_to_cell(rank, DUAL)
            back = cell_to_rank(row, col, DUAL)
            self.assertEqual(back, rank)
