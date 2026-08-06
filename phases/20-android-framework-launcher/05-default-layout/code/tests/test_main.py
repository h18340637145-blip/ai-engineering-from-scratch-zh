import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import LayoutItem, LayoutConfig, validate_item, CONTAINER_DESKTOP, CONTAINER_HOTSEAT

CFG = LayoutConfig(columns=5, rows=5, hotseat_count=5)

class TestValidateItem(unittest.TestCase):
    def _desk(self, x, y, sx=1, sy=1):
        return LayoutItem("pkg", "cls", CONTAINER_DESKTOP, 0, x, y, sx, sy)

    def test_valid_desktop(self):
        ok, _ = validate_item(self._desk(0, 0), CFG)
        self.assertTrue(ok)

    def test_x_overflow(self):
        ok, msg = validate_item(self._desk(4, 0, 2), CFG)
        self.assertFalse(ok)
        self.assertIn("X", msg)

    def test_y_overflow(self):
        ok, msg = validate_item(self._desk(0, 5), CFG)
        self.assertFalse(ok)
        self.assertIn("Y", msg)

    def test_negative_coords(self):
        ok, _ = validate_item(self._desk(-1, 0), CFG)
        self.assertFalse(ok)

    def test_valid_hotseat(self):
        item = LayoutItem("pkg", "cls", CONTAINER_HOTSEAT, 2, 0, 0)
        ok, _ = validate_item(item, CFG)
        self.assertTrue(ok)

    def test_hotseat_overflow(self):
        item = LayoutItem("pkg", "cls", CONTAINER_HOTSEAT, 5, 0, 0)
        ok, msg = validate_item(item, CFG)
        self.assertFalse(ok)
        self.assertIn("Hotseat", msg)

    def test_unknown_container(self):
        item = LayoutItem("pkg", "cls", 999, 0, 0, 0)
        ok, msg = validate_item(item, CFG)
        self.assertFalse(ok)
        self.assertIn("未知", msg)

    def test_boundary_x_exactly(self):
        ok, _ = validate_item(self._desk(4, 0, 1), CFG)
        self.assertTrue(ok)
