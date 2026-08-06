# Android Native 日志与调用栈模块配置检查
# 课程文档：phases/22-android-framework-system-basics/02-native-log-and-callstack/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的 Native 日志与调用栈章节
# 模型仅验证 Android.bp 依赖意图，不编译或修改任何 Native 模块。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Validation:
    ok: bool
    messages: list[str]


@dataclass
class NativeModuleConfig:
    log_ndebug: int
    libs: set[str] = field(default_factory=set)
    uses_callstack: bool = False
    uses_log: bool = False

    def debug_logging_enabled(self) -> bool:
        return self.log_ndebug == 0

    def validate(self) -> Validation:
        messages: list[str] = []
        if self.uses_callstack and "libutils" not in self.libs:
            messages.append("libutils")
        if self.uses_log and "liblog" not in self.libs:
            messages.append("liblog")
        return Validation(not messages, messages)


def main() -> None:
    print("=== Native 日志模块检查 ===")
    config = NativeModuleConfig(0, {"libutils", "liblog"}, uses_callstack=True, uses_log=True)
    print("调试日志启用：", config.debug_logging_enabled())
    print("依赖有效：", config.validate().ok)


if __name__ == "__main__":
    main()
