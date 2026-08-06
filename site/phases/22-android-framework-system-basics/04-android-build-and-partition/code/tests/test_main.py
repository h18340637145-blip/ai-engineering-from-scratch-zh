import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ModuleSpec, installation_path
except ImportError:
    ModuleSpec = installation_path = None


class TestAndroidBuildAndPartition(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(ModuleSpec, "尚未实现 Android 模块与分区校验")

    def test_platform_privileged_system_ext_app_is_valid(self):
        self._require()
        self.assertTrue(ModuleSpec("Demo", "system_ext", True, "platform").validate().ok)

    def test_privileged_vendor_app_is_rejected_by_teaching_policy(self):
        self._require()
        self.assertFalse(ModuleSpec("Demo", "vendor", True, "platform").validate().ok)

    def test_unknown_partition_is_rejected(self):
        self._require()
        self.assertFalse(ModuleSpec("Demo", "unknown", False, "PRESIGNED").validate().ok)

    def test_system_privileged_app_path_uses_priv_app(self):
        self._require()
        self.assertEqual(installation_path("system", True), "/system/priv-app")

    def test_product_non_privileged_app_path_uses_app(self):
        self._require()
        self.assertEqual(installation_path("product", False), "/product/app")

    def test_presigned_certificate_is_allowed_for_prebuilt(self):
        self._require()
        self.assertTrue(ModuleSpec("Prebuilt", "product", False, "PRESIGNED").validate().ok)

