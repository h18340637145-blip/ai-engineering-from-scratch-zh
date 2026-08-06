# QuickStep 调用链模拟
# 课程文档：phases/20-android-framework-launcher/09-recents-and-quickstep/docs/en.md
# 参考：AOSP quickstep/src/com/android/quickstep/TouchInteractionService.java

"""
模拟 QuickStep 手势识别的输入消费者选择逻辑：
根据当前前台应用判断使用哪个 InputConsumer。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class GestureType(Enum):
    SWIPE_UP = "上滑"
    SWIPE_LEFT = "左滑"
    SWIPE_RIGHT = "右滑"


@dataclass
class SystemState:
    foreground_package: str    # 当前前台应用包名
    is_launcher: bool          # 是否在 Launcher 界面
    nav_mode: str              # "three_button" 或 "gesture"


def resolve_input_consumer(state: SystemState, gesture: GestureType) -> str:
    """
    模拟 TouchInteractionService 选择 InputConsumer 的逻辑。
    """
    if state.nav_mode == "three_button":
        return "三键导航 - 由 SystemUI 处理，不经过 TouchInteractionService"

    if gesture == GestureType.SWIPE_UP:
        if state.is_launcher:
            return "LauncherSwipeHandler - 从 Launcher 上滑进入 Overview"
        else:
            return "OtherActivityInputConsumer - 从其他应用上滑进入 Overview"

    if gesture in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT):
        return "TaskSwitchInputConsumer - 左右快速切换最近任务"

    return "DefaultInputConsumer - 默认处理"


def main() -> None:
    print("=== QuickStep InputConsumer 路由模拟 ===\n")
    scenarios = [
        (SystemState("com.android.launcher3", True, "gesture"), GestureType.SWIPE_UP, "桌面上滑"),
        (SystemState("com.android.settings", False, "gesture"), GestureType.SWIPE_UP, "设置上滑"),
        (SystemState("com.android.chrome", False, "gesture"), GestureType.SWIPE_LEFT, "Chrome 左滑"),
        (SystemState("com.android.camera2", False, "three_button"), GestureType.SWIPE_UP, "三键导航上滑"),
    ]
    for state, gesture, desc in scenarios:
        consumer = resolve_input_consumer(state, gesture)
        print(f"  场景: {desc}")
        print(f"  前台: {state.foreground_package}")
        print(f"  手势: {gesture.value}")
        print(f"  → {consumer}\n")


if __name__ == "__main__":
    main()
