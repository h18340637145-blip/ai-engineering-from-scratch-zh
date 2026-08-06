import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import BinderTransaction, BuildPipeline, choose_class_loader
except ImportError:
    BinderTransaction = BuildPipeline = choose_class_loader = None


class TestBinderClassloadingBuildAndInstall(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(BinderTransaction, "尚未实现 Binder 与构建流水线模型")

    def test_small_binder_transaction_is_safe(self):
        self._require()
        self.assertTrue(BinderTransaction(64 * 1024).is_safe())

    def test_large_binder_transaction_is_rejected(self):
        self._require()
        self.assertFalse(BinderTransaction(2_000_000).is_safe())

    def test_installed_app_uses_path_class_loader(self):
        self._require()
        self.assertEqual(choose_class_loader("installed-app"), "PathClassLoader")

    def test_controlled_plugin_uses_dex_class_loader(self):
        self._require()
        self.assertEqual(choose_class_loader("trusted-plugin"), "DexClassLoader")

    def test_build_pipeline_has_aapt2_before_d8(self):
        self._require()
        steps = BuildPipeline.steps()
        self.assertLess(steps.index("AAPT2"), steps.index("D8"))

    def test_build_pipeline_signs_before_release(self):
        self._require()
        self.assertEqual(BuildPipeline.steps()[-1], "sign and zipalign")

