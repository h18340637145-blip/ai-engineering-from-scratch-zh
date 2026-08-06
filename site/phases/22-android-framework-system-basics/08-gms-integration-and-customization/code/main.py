# GMS 预置包与 Setup Wizard Partner 定制模拟
# 课程文档：phases/22-android-framework-system-basics/08-gms-integration-and-customization/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的 GMS 集成与自定义向导适配章节
# 模型不包含 GMS 二进制或授权材料，只验证产品集成描述的完整性。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Validation:
    ok: bool
    messages: list[str]


@dataclass(frozen=True)
class PrebuiltPackage:
    name: str
    partition: str
    certificate: str
    privileged: bool

    def validate(self) -> Validation:
        messages: list[str] = []
        if self.partition not in {"system", "system_ext", "product", "vendor"}:
            messages.append("unknown partition")
        if self.certificate not in {"PRESIGNED", "platform", "shared"}:
            messages.append("unknown certificate")
        if not self.name:
            messages.append("missing module name")
        return Validation(not messages, messages)


class WizardScriptGraph:
    def __init__(self, transitions: dict[tuple[str, int], str]) -> None:
        self.transitions = transitions

    def next_action(self, action_id: str, result_code: int) -> str | None:
        return self.transitions.get((action_id, result_code))


def partner_receiver_valid(actions: list[str]) -> bool:
    return "com.android.setupwizard.action.PARTNER_CUSTOMIZATION" in actions


def main() -> None:
    print("=== GMS 预置与向导定制 ===")
    package = PrebuiltPackage("GmsCore", "product", "PRESIGNED", False)
    print("预置包有效：", package.validate().ok)
    graph = WizardScriptGraph({("welcome", 111): "setup_navigation"})
    print("welcome 后动作：", graph.next_action("welcome", 111))
    print("Partner Receiver 有效：", partner_receiver_valid(["com.android.setupwizard.action.PARTNER_CUSTOMIZATION"]))


if __name__ == "__main__":
    main()
