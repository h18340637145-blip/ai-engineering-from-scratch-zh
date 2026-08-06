# 测试：启动链路日志解析与耗时计算
import unittest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import parse_log, extract_startup_chain, parse_timestamp_ms


SAMPLE = """\
08-06 10:00:00.100  1234  1234 D LauncherStudy: Launcher.onCreate begin
08-06 10:00:00.200  1234  1234 D LauncherStudy: Launcher.setupViews begin
08-06 10:00:00.500  1234  1234 D LauncherStudy: Launcher.finishBindingItems
"""


class TestParseTimestampMs(unittest.TestCase):
    def test_basic(self):
        ms = parse_timestamp_ms("08-06 10:00:01", "100")
        self.assertEqual(ms, 10 * 3600 * 1000 + 1 * 1000 + 100)

    def test_zero(self):
        ms = parse_timestamp_ms("08-06 00:00:00", "000")
        self.assertEqual(ms, 0)


class TestParseLog(unittest.TestCase):
    def test_parses_three_lines(self):
        lines = parse_log(SAMPLE)
        self.assertEqual(len(lines), 3)

    def test_timestamps_relative(self):
        lines = parse_log(SAMPLE)
        self.assertEqual(lines[0].timestamp_ms, 0)
        self.assertEqual(lines[1].timestamp_ms, 100)
        self.assertEqual(lines[2].timestamp_ms, 400)

    def test_ignores_invalid_lines(self):
        lines = parse_log("not a logcat line\n" + SAMPLE)
        self.assertEqual(len(lines), 3)


class TestExtractStartupChain(unittest.TestCase):
    def test_extracts_in_order(self):
        lines = parse_log(SAMPLE)
        chain = extract_startup_chain(lines)
        self.assertEqual(len(chain), 3)
        self.assertIn("onCreate", chain[0][0])
        self.assertIn("setupViews", chain[1][0])
        self.assertIn("finishBindingItems", chain[2][0])

    def test_no_duplicate_stages(self):
        double = SAMPLE + SAMPLE
        lines = parse_log(double)
        chain = extract_startup_chain(lines)
        stages = [s for s, _ in chain]
        self.assertEqual(len(stages), len(set(stages)))

    def test_empty_log_returns_empty(self):
        self.assertEqual(extract_startup_chain([]), [])
