# Activity、Window 与 Service 生命周期模拟
# 课程文档：phases/21-java-android-foundations/09-activity-window-and-service/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 Activity、Fragment、Window 与 Service 章节
# 以确定性路径模拟组件生命周期和启动模式，不依赖 Android 运行时。

from __future__ import annotations


class LifecyclePlanner:
    @staticmethod
    def foreground_path() -> list[str]:
        return ["onCreate", "onStart", "onResume"]

    @staticmethod
    def return_path() -> list[str]:
        return ["onRestart", "onStart", "onResume"]

    @staticmethod
    def needs_saved_state(state: str) -> bool:
        return state in {"stopped", "background"}


class LaunchModeResolver:
    @staticmethod
    def resolve(mode: str, top_matches: bool, task_has_existing: bool) -> str:
        if mode == "singleTop" and top_matches:
            return "reuse-onNewIntent"
        if mode == "singleTask" and task_has_existing:
            return "reuse-clear-above"
        if mode == "singleInstance":
            return "isolated-task"
        return "new-instance"


def service_mode(requirement: str) -> str:
    modes = {
        "long-running": "foreground service",
        "client-bound": "bound service",
        "short-work": "WorkManager",
    }
    if requirement not in modes:
        raise ValueError("未知服务需求")
    return modes[requirement]


def main() -> None:
    print("=== 组件生命周期 ===")
    print("进入前台：", " -> ".join(LifecyclePlanner.foreground_path()))
    print("singleTop：", LaunchModeResolver.resolve("singleTop", True, False))
    print("长期任务：", service_mode("long-running"))


if __name__ == "__main__":
    main()
