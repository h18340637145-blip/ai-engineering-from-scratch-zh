# 测试：logcat 解析与过滤逻辑
import unittest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import LogEntry, LogcatFilter, parse_logcat, filter_by_tag, count_by_level


SAMPLE_LINE = "08-06 10:00:00.100  1234  1234 D LauncherStudy: Launcher.onCreate begin"
WARN_LINE   = "08-06 10:00:00.200  1234  1235 W LoaderCursor  : item overlap"
ERROR_LINE  = "08-06 10:00:00.300  1234  1234 E LauncherStudy: null itemInfo"
BAD_LINE    = "this is not a valid logcat line"


class TestLogEntryParse(unittest.TestCase):
    def test_parse_valid_debug_line(self):
        e = LogEntry.parse(SAMPLE_LINE)
        self.assertIsNotNone(e)
        self.assertEqual(e.level, "D")
        self.assertEqual(e.tag, "LauncherStudy")
        self.assertIn("onCreate", e.message)

    def test_parse_warn_line(self):
        e = LogEntry.parse(WARN_LINE)
        self.assertIsNotNone(e)
        self.assertEqual(e.level, "W")

    def test_parse_invalid_line_returns_none(self):
        self.assertIsNone(LogEntry.parse(BAD_LINE))

    def test_parse_empty_line_returns_none(self):
        self.assertIsNone(LogEntry.parse(""))


class TestLogcatFilter(unittest.TestCase):
    def setUp(self):
        self.debug_entry = LogEntry.parse(SAMPLE_LINE)
        self.error_entry = LogEntry.parse(ERROR_LINE)

    def test_tag_match(self):
        f = LogcatFilter("LauncherStudy", "D")
        self.assertTrue(f.match(self.debug_entry))

    def test_tag_no_match(self):
        f = LogcatFilter("OtherTag", "D")
        self.assertFalse(f.match(self.debug_entry))

    def test_level_filter_excludes_below_min(self):
        f = LogcatFilter("LauncherStudy", "E")
        self.assertFalse(f.match(self.debug_entry))  # D < E
        self.assertTrue(f.match(self.error_entry))


class TestParseLogcat(unittest.TestCase):
    TEXT = SAMPLE_LINE + "\n" + WARN_LINE + "\n" + BAD_LINE

    def test_parses_valid_entries(self):
        entries = parse_logcat(self.TEXT)
        self.assertEqual(len(entries), 2)

    def test_filter_by_tag(self):
        entries = parse_logcat(SAMPLE_LINE + "\n" + WARN_LINE)
        result = filter_by_tag(entries, "LauncherStudy")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].level, "D")

    def test_count_by_level(self):
        text = SAMPLE_LINE + "\n" + WARN_LINE + "\n" + ERROR_LINE
        entries = parse_logcat(text)
        counts = count_by_level(entries)
        self.assertEqual(counts["D"], 1)
        self.assertEqual(counts["W"], 1)
        self.assertEqual(counts["E"], 1)
        self.assertEqual(counts["V"], 0)
