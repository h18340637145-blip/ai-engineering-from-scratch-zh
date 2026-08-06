import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import RenderRequest, TouchDispatcher, scroll_owner
except ImportError:
    RenderRequest = TouchDispatcher = scroll_owner = None


class TestViewRenderingAndTouch(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(RenderRequest, "尚未实现绘制与触摸分发模型")

    def test_geometry_change_requests_layout_and_draw(self):
        self._require()
        self.assertEqual(RenderRequest(True, True).plan(), ["requestLayout", "invalidate"])

    def test_pixel_only_change_requests_invalidate(self):
        self._require()
        self.assertEqual(RenderRequest(False, True).plan(), ["invalidate"])

    def test_child_receives_down_when_it_consumes_event(self):
        self._require()
        self.assertEqual(TouchDispatcher().dispatch("DOWN", True, False), ["child:DOWN"])

    def test_parent_intercept_sends_cancel_to_child(self):
        self._require()
        self.assertEqual(
            TouchDispatcher().dispatch("MOVE", True, True),
            ["child:CANCEL", "parent:MOVE"],
        )

    def test_unconsumed_down_ends_child_sequence(self):
        self._require()
        self.assertEqual(TouchDispatcher().dispatch("DOWN", False, False), ["unhandled:DOWN"])

    def test_scroll_owner_uses_direction_and_child_capacity(self):
        self._require()
        self.assertEqual(scroll_owner(20, 5, True), "parent-horizontal")
        self.assertEqual(scroll_owner(1, 20, True), "child-vertical")
        self.assertEqual(scroll_owner(1, 20, False), "parent-vertical")

