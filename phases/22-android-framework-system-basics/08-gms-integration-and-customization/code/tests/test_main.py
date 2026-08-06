import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import PrebuiltPackage, WizardScriptGraph, partner_receiver_valid
except ImportError:
    PrebuiltPackage = WizardScriptGraph = partner_receiver_valid = None


class TestGmsIntegrationAndCustomization(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(PrebuiltPackage, "尚未实现预置包与向导脚本模型")

    def test_product_presigned_package_is_valid(self):
        self._require()
        self.assertTrue(PrebuiltPackage("GmsCore", "product", "PRESIGNED", False).validate().ok)

    def test_unknown_partition_is_invalid(self):
        self._require()
        self.assertFalse(PrebuiltPackage("GmsCore", "unknown", "PRESIGNED", False).validate().ok)

    def test_privileged_package_is_explicit(self):
        self._require()
        self.assertTrue(PrebuiltPackage("SetupWizard", "product", "PRESIGNED", True).validate().ok)

    def test_wizard_graph_follows_named_result(self):
        self._require()
        graph = WizardScriptGraph({("welcome", 111): "setup_navigation"})
        self.assertEqual(graph.next_action("welcome", 111), "setup_navigation")

    def test_unknown_result_has_no_next_action(self):
        self._require()
        graph = WizardScriptGraph({})
        self.assertIsNone(graph.next_action("welcome", 0))

    def test_partner_receiver_requires_customization_action(self):
        self._require()
        self.assertTrue(partner_receiver_valid(["com.android.setupwizard.action.PARTNER_CUSTOMIZATION"]))
        self.assertFalse(partner_receiver_valid(["android.intent.action.BOOT_COMPLETED"]))

