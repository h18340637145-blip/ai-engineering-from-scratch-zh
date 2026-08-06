# 技能：Quickstep InputConsumer 路由分析

## 用途
根据当前前台应用和导航模式，判断 TouchInteractionService（TIS）应将手势事件路由给哪个 InputConsumer。

## 核心接口
```python
from main import SystemState, GestureType, resolve_input_consumer

state = SystemState(
    foreground_pkg="com.android.settings",
    is_launcher=False,
    nav_mode="gesture",
)
consumer = resolve_input_consumer(state, GestureType.SWIPE_UP)
print(f"InputConsumer: {consumer}")
```

## 路由规则
| 前台 | 导航模式 | 手势 | 路由目标 |
|------|----------|------|---------|
| Launcher | gesture | 上滑 | LauncherSwipeHandler |
| 其他 App | gesture | 上滑 | OtherAppSwipeHandler |
| 任意 | gesture | 左/右滑 | TaskSwitchConsumer |
| 任意 | three_button | 任意 | 三键导航（跳过 TIS） |

## 来源
AOSP `packages/apps/Launcher3/quickstep/src/com/android/launcher3/taskbar/`
