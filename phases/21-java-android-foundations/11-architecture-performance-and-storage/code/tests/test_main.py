import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ArchitectureAdvisor, OptimizationPlan, cross_process_storage
except ImportError:
    ArchitectureAdvisor = OptimizationPlan = cross_process_storage = None


class TestArchitecturePerformanceAndStorage(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(ArchitectureAdvisor, "尚未实现架构与性能决策模型")

    def test_observable_ui_state_prefers_mvvm(self):
        self._require()
        self.assertEqual(ArchitectureAdvisor.recommend("observable-state"), "MVVM")

    def test_passive_view_with_presenter_prefers_mvp(self):
        self._require()
        self.assertEqual(ArchitectureAdvisor.recommend("passive-view"), "MVP")

    def test_unmeasured_problem_starts_with_measurement(self):
        self._require()
        self.assertEqual(OptimizationPlan.next_action(False), "measure")

    def test_measured_hotspot_can_be_optimized(self):
        self._require()
        self.assertEqual(OptimizationPlan.next_action(True), "optimize")

    def test_shared_preferences_is_not_cross_process_store(self):
        self._require()
        self.assertEqual(cross_process_storage("SharedPreferences"), "avoid")

    def test_content_provider_is_valid_cross_process_store(self):
        self._require()
        self.assertEqual(cross_process_storage("ContentProvider"), "valid")

