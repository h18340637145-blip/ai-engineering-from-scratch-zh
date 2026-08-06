import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ProvisioningState, SetupIntentValidator
except ImportError:
    ProvisioningState = SetupIntentValidator = None


class TestSetupWizardAndProvisioning(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(ProvisioningState, "尚未实现开机向导状态模型")

    def test_unprovisioned_device_enters_setup_wizard(self):
        self._require()
        self.assertEqual(ProvisioningState(0, 0).next_home(), "SetupWizard")

    def test_completed_device_and_user_enter_launcher(self):
        self._require()
        self.assertEqual(ProvisioningState(1, 1).next_home(), "Launcher")

    def test_device_complete_but_user_incomplete_stays_in_setup(self):
        self._require()
        self.assertEqual(ProvisioningState(1, 0).next_home(), "SetupWizard")

    def test_setup_intent_requires_setup_wizard_category(self):
        self._require()
        self.assertFalse(SetupIntentValidator(["MAIN", "DEFAULT"]).is_valid())

    def test_setup_intent_with_home_is_rejected(self):
        self._require()
        self.assertFalse(SetupIntentValidator(["MAIN", "SETUP_WIZARD", "HOME"]).is_valid())

    def test_complete_setup_sets_both_status_flags(self):
        self._require()
        state = ProvisioningState(0, 0)
        completed = state.complete()
        self.assertEqual((completed.device_provisioned, completed.user_setup_complete), (1, 1))

