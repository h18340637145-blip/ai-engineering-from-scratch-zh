import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import Overlay, OverlayRegistry
except ImportError:
    Overlay = OverlayRegistry = None


class TestResourceOverlays(unittest.TestCase):
    def _registry(self):
        self.assertIsNotNone(OverlayRegistry, "尚未实现资源 Overlay 注册表")
        return OverlayRegistry()

    def test_highest_priority_enabled_overlay_wins(self):
        registry = self._registry()
        registry.add(Overlay("com.example.low", "android", "navigation", 1, True, "RRO"))
        registry.add(Overlay("com.example.high", "android", "navigation", 10, True, "RRO"))
        self.assertEqual(registry.resolve("navigation", user=0).package, "com.example.high")

    def test_disabled_overlay_does_not_win(self):
        registry = self._registry()
        registry.add(Overlay("com.example.disabled", "android", "navigation", 20, False, "RRO"))
        registry.add(Overlay("com.example.enabled", "android", "navigation", 1, True, "RRO"))
        self.assertEqual(registry.resolve("navigation", user=0).package, "com.example.enabled")

    def test_exclusive_category_enables_only_requested_overlay(self):
        registry = self._registry()
        registry.add(Overlay("com.example.buttons", "android", "navigation", 1, True, "RRO"))
        registry.add(Overlay("com.example.gesture", "android", "navigation", 2, False, "RRO"))
        registry.enable_exclusive("navigation", "com.example.gesture")
        self.assertEqual(registry.resolve("navigation", user=0).package, "com.example.gesture")

    def test_sro_is_not_runtime_toggleable(self):
        registry = self._registry()
        static = Overlay("com.example.static", "android", "theme", 1, True, "SRO")
        registry.add(static)
        self.assertFalse(registry.can_toggle("com.example.static"))

    def test_rro_is_runtime_toggleable(self):
        registry = self._registry()
        registry.add(Overlay("com.example.runtime", "android", "theme", 1, True, "RRO"))
        self.assertTrue(registry.can_toggle("com.example.runtime"))

    def test_missing_category_returns_none(self):
        registry = self._registry()
        self.assertIsNone(registry.resolve("missing", user=0))

