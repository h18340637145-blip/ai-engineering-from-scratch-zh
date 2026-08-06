# Android Framework 集成验收实验室
# 课程文档：phases/22-android-framework-system-basics/09-framework-integration-lab/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的启动、Overlay、设置与向导章节
# 将多个系统层的验收条件汇总为明确的缺口和下一条只读诊断命令。

from __future__ import annotations


class FrameworkReadinessReport:
    REQUIRED = ("boot", "zygote", "overlay", "setup_wizard", "permissions")

    def __init__(self, checks: dict[str, bool]) -> None:
        self.checks = checks

    @property
    def ready(self) -> bool:
        return not self.missing()

    def missing(self) -> list[str]:
        return [name for name in self.REQUIRED if not self.checks.get(name, False)]

    def next_command(self) -> str:
        missing = self.missing()
        if not missing:
            return "adb reboot && adb logcat -b events -d"
        first = missing[0]
        commands = {
            "boot": "adb logcat -b events -d | grep -E 'boot_progress|wm_boot_animation_done'",
            "zygote": "adb shell ps -A | grep zygote",
            "overlay": "adb shell cmd overlay list --user current",
            "setup_wizard": "adb shell settings get global device_provisioned",
            "permissions": "adb shell dumpsys package <package>",
        }
        return commands[first]


def main() -> None:
    print("=== Framework 集成验收 ===")
    report = FrameworkReadinessReport({"boot": True, "zygote": True, "overlay": False, "setup_wizard": True, "permissions": True})
    print("就绪：", report.ready)
    print("缺口：", report.missing())
    print("下一条命令：", report.next_command())


if __name__ == "__main__":
    main()
