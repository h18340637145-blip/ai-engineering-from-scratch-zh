# Android 构建模块与分区归属校验
# 课程文档：phases/22-android-framework-system-basics/04-android-build-and-partition/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的构建与分区配置章节
# 教学模型显式标出简化策略，实际产品仍需以当前分区策略为准。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Validation:
    ok: bool
    messages: list[str]


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    partition: str
    privileged: bool
    certificate: str

    def validate(self) -> Validation:
        messages: list[str] = []
        if self.partition not in {"system", "system_ext", "product", "vendor"}:
            messages.append("unknown partition")
        if self.privileged and self.partition == "vendor":
            messages.append("teaching policy rejects privileged vendor app")
        if self.certificate not in {"platform", "shared", "PRESIGNED"}:
            messages.append("unknown certificate")
        return Validation(not messages, messages)


def installation_path(partition: str, privileged: bool) -> str:
    suffix = "priv-app" if privileged else "app"
    return f"/{partition}/{suffix}"


def main() -> None:
    print("=== Android 模块与分区 ===")
    spec = ModuleSpec("ExampleApp", "system_ext", True, "platform")
    print("模块有效：", spec.validate().ok)
    print("安装路径：", installation_path(spec.partition, spec.privileged))


if __name__ == "__main__":
    main()
