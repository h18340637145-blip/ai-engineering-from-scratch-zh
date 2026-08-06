import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ImageCache, ListPerformanceAdvisor
except ImportError:
    ImageCache = ListPerformanceAdvisor = None


class TestListRenderingAndImageCache(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(ImageCache, "尚未实现列表与图片缓存模型")

    def test_nested_fixed_list_reuses_pool(self):
        self._require()
        recommendations = ListPerformanceAdvisor.configure(nested=True, fixed_size=True)
        self.assertIn("shared RecycledViewPool", recommendations)
        self.assertIn("setHasFixedSize(true)", recommendations)

    def test_list_updates_prefer_diff_over_full_refresh(self):
        self._require()
        self.assertEqual(ListPerformanceAdvisor.update_strategy(4), "notifyItem range")

    def test_cache_fetches_network_on_first_miss(self):
        self._require()
        cache = ImageCache()
        self.assertEqual(cache.fetch("avatar"), "network")

    def test_cache_fetches_memory_after_first_load(self):
        self._require()
        cache = ImageCache()
        cache.fetch("avatar")
        self.assertEqual(cache.fetch("avatar"), "memory")

    def test_disk_is_used_before_network_when_available(self):
        self._require()
        cache = ImageCache(disk={"avatar"})
        self.assertEqual(cache.fetch("avatar"), "disk")

    def test_soft_reference_is_not_primary_cache_strategy(self):
        self._require()
        self.assertFalse(ListPerformanceAdvisor.uses_soft_reference_as_primary_cache())

