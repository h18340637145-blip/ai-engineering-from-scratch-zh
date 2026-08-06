# Launcher3 入门 - 模拟系统识别默认桌面的核心逻辑
# 课程文档：phases/20-android-framework-launcher/01-launcher3-intro/docs/en.md
# 参考：Android AOSP Launcher3 源码 packages/apps/Launcher3/AndroidManifest.xml

"""
模拟 Android PackageManagerService 识别 CATEGORY_HOME 的逻辑。

Android 系统在处理 Home 键时，会查询所有声明了 CATEGORY_HOME 的 Activity。
本脚本通过解析简化版 Manifest 配置字典，演示该判断过程。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional


CATEGORY_HOME = "android.intent.category.HOME"
CATEGORY_DEFAULT = "android.intent.category.DEFAULT"
ACTION_MAIN = "android.intent.action.MAIN"


@dataclass
class IntentFilter:
    """模拟 AndroidManifest 中的 <intent-filter> 节点。"""
    actions: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def is_home_intent(self) -> bool:
        """判断是否为系统主屏幕候选声明。"""
        return (
            ACTION_MAIN in self.actions
            and CATEGORY_HOME in self.categories
        )


@dataclass
class ActivityInfo:
    """模拟一个 Activity 的基本信息。"""
    name: str
    package: str
    intent_filters: List[IntentFilter] = field(default_factory=list)

    def can_be_home(self) -> bool:
        """判断该 Activity 是否可成为系统桌面。"""
        return any(f.is_home_intent() for f in self.intent_filters)


@dataclass
class PackageInfo:
    """模拟已安装应用的 Manifest 信息。"""
    package: str
    label: str
    activities: List[ActivityInfo] = field(default_factory=list)

    def get_home_activities(self) -> List[ActivityInfo]:
        """返回所有声明了 CATEGORY_HOME 的 Activity 列表。"""
        return [a for a in self.activities if a.can_be_home()]


def resolve_home_activity(packages: List[PackageInfo]) -> Optional[ActivityInfo]:
    """
    模拟系统解析默认桌面的过程。

    当有多个候选时，系统会弹出选择对话框。
    此处简化为返回第一个候选。
    """
    candidates: List[ActivityInfo] = []
    for pkg in packages:
        candidates.extend(pkg.get_home_activities())

    if not candidates:
        return None
    return candidates[0]


def build_demo_packages() -> List[PackageInfo]:
    """构建演示用的已安装应用列表。"""
    # Launcher3QuickStep - 声明了 CATEGORY_HOME
    launcher_activity = ActivityInfo(
        name="com.android.launcher3.uioverrides.QuickstepLauncher",
        package="com.android.launcher3",
        intent_filters=[
            IntentFilter(
                actions=[ACTION_MAIN],
                categories=[CATEGORY_HOME, CATEGORY_DEFAULT],
            )
        ],
    )
    launcher_pkg = PackageInfo(
        package="com.android.launcher3",
        label="Launcher3QuickStep",
        activities=[launcher_activity],
    )

    # 设置应用 - 没有声明 CATEGORY_HOME
    settings_activity = ActivityInfo(
        name="com.android.settings.Settings",
        package="com.android.settings",
        intent_filters=[
            IntentFilter(
                actions=[ACTION_MAIN],
                categories=[CATEGORY_DEFAULT],
            )
        ],
    )
    settings_pkg = PackageInfo(
        package="com.android.settings",
        label="设置",
        activities=[settings_activity],
    )

    return [settings_pkg, launcher_pkg]


def main() -> None:
    packages = build_demo_packages()

    print("=== 模拟 PackageManagerService 扫描已安装应用 ===\n")
    for pkg in packages:
        home_acts = pkg.get_home_activities()
        status = "✅ 可作为系统桌面" if home_acts else "❌ 无法成为系统桌面"
        print(f"  {pkg.label} ({pkg.package}): {status}")

    print("\n=== 模拟 Home 键按下 → 解析默认桌面 ===\n")
    winner = resolve_home_activity(packages)
    if winner:
        print(f"  系统将启动: {winner.name}")
        print(f"  所属包名:   {winner.package}")
    else:
        print("  未找到任何系统桌面候选，这在 AOSP 中不应出现。")


if __name__ == "__main__":
    main()
