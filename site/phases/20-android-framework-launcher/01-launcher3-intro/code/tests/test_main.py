# 测试：Launcher3 系统桌面识别逻辑
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import (
    IntentFilter, ActivityInfo, PackageInfo,
    resolve_home_activity,
    CATEGORY_HOME, CATEGORY_DEFAULT, ACTION_MAIN,
)


class TestIntentFilter(unittest.TestCase):
    def test_home_intent_recognised(self):
        f = IntentFilter(
            actions=[ACTION_MAIN],
            categories=[CATEGORY_HOME, CATEGORY_DEFAULT],
        )
        self.assertTrue(f.is_home_intent())

    def test_missing_category_home_fails(self):
        f = IntentFilter(actions=[ACTION_MAIN], categories=[CATEGORY_DEFAULT])
        self.assertFalse(f.is_home_intent())

    def test_missing_action_main_fails(self):
        f = IntentFilter(actions=[], categories=[CATEGORY_HOME])
        self.assertFalse(f.is_home_intent())

    def test_empty_filter_fails(self):
        self.assertFalse(IntentFilter().is_home_intent())


class TestActivityInfo(unittest.TestCase):
    def _make_activity(self, has_home: bool) -> ActivityInfo:
        cats = [CATEGORY_HOME, CATEGORY_DEFAULT] if has_home else [CATEGORY_DEFAULT]
        return ActivityInfo(
            name="com.example.Main",
            package="com.example",
            intent_filters=[IntentFilter(actions=[ACTION_MAIN], categories=cats)],
        )

    def test_activity_with_home_can_be_home(self):
        self.assertTrue(self._make_activity(True).can_be_home())

    def test_activity_without_home_cannot_be_home(self):
        self.assertFalse(self._make_activity(False).can_be_home())

    def test_activity_no_filters(self):
        a = ActivityInfo(name="X", package="Y")
        self.assertFalse(a.can_be_home())


class TestResolveHomeActivity(unittest.TestCase):
    def _launcher_pkg(self, pkg_name: str) -> PackageInfo:
        a = ActivityInfo(
            name=f"{pkg_name}.Launcher",
            package=pkg_name,
            intent_filters=[
                IntentFilter(actions=[ACTION_MAIN], categories=[CATEGORY_HOME])
            ],
        )
        return PackageInfo(package=pkg_name, label="Launcher", activities=[a])

    def test_resolve_returns_first_candidate(self):
        pkgs = [self._launcher_pkg("com.a"), self._launcher_pkg("com.b")]
        result = resolve_home_activity(pkgs)
        self.assertIsNotNone(result)
        self.assertEqual(result.package, "com.a")

    def test_resolve_no_candidates_returns_none(self):
        settings = PackageInfo(
            package="com.android.settings",
            label="设置",
            activities=[ActivityInfo(name="S", package="com.android.settings")],
        )
        self.assertIsNone(resolve_home_activity([settings]))

    def test_resolve_empty_list_returns_none(self):
        self.assertIsNone(resolve_home_activity([]))
