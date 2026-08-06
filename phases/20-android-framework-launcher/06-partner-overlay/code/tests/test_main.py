import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import PackageInfo, ReceiverInfo, discover_partners, PARTNER_ACTION


class TestPartnerDiscovery(unittest.TestCase):
    def _partner(self, is_system=True, enabled=True):
        return PackageInfo("pkg", "Pkg", is_system,
                           [ReceiverInfo(PARTNER_ACTION, enabled)])

    def test_valid_partner_detected(self):
        pkg = self._partner()
        self.assertTrue(pkg.is_valid_partner())

    def test_non_system_rejected(self):
        pkg = self._partner(is_system=False)
        self.assertFalse(pkg.is_valid_partner())

    def test_disabled_receiver_rejected(self):
        pkg = self._partner(enabled=False)
        self.assertFalse(pkg.is_valid_partner())

    def test_no_receiver_rejected(self):
        pkg = PackageInfo("pkg", "Pkg", True)
        self.assertFalse(pkg.is_valid_partner())

    def test_discover_filters_correctly(self):
        pkgs = [self._partner(), self._partner(is_system=False), PackageInfo("x", "X", True)]
        result = discover_partners(pkgs)
        self.assertEqual(len(result), 1)

    def test_discover_empty_list(self):
        self.assertEqual(discover_partners([]), [])
