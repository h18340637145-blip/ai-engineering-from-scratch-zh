import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import BootAnimationSpec, BootTimeline
except ImportError:
    BootAnimationSpec = BootTimeline = None


class TestBootChainAndBootanimation(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(BootAnimationSpec, "尚未实现开机动画规格模型")

    def test_desc_header_parses_dimensions_and_fps(self):
        self._require()
        spec = BootAnimationSpec.parse("1080 1920 30\np 0 0 part0")
        self.assertEqual((spec.width, spec.height, spec.fps), (1080, 1920, 30))

    def test_p_part_is_interruptible(self):
        self._require()
        spec = BootAnimationSpec.parse("1080 1920 30\np 0 0 part0")
        self.assertTrue(spec.parts[0].interruptible)

    def test_c_part_must_finish_current_loop(self):
        self._require()
        spec = BootAnimationSpec.parse("1080 1920 30\nc 1 0 part0")
        self.assertFalse(spec.parts[0].interruptible)

    def test_invalid_header_is_rejected(self):
        self._require()
        with self.assertRaises(ValueError):
            BootAnimationSpec.parse("1080 1920")

    def test_timeline_reports_gap_between_stop_and_done(self):
        self._require()
        timeline = BootTimeline(["10 stop_bootanim", "17 wm_boot_animation_done"])
        self.assertEqual(timeline.find_gap(), 7)

    def test_timeline_returns_none_without_both_markers(self):
        self._require()
        self.assertIsNone(BootTimeline(["10 stop_bootanim"]).find_gap())

