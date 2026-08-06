# Setup Wizard Provisioning 状态与 Intent 类别模拟
# 课程文档：phases/22-android-framework-system-basics/07-setup-wizard-and-provisioning/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的开机向导定制章节
# 模型仅表示状态转移和 Intent 校验，不写 Settings 或禁用真实组件。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProvisioningState:
    device_provisioned: int
    user_setup_complete: int

    def next_home(self) -> str:
        return "Launcher" if self.device_provisioned and self.user_setup_complete else "SetupWizard"

    def complete(self) -> "ProvisioningState":
        return ProvisioningState(1, 1)


class SetupIntentValidator:
    def __init__(self, categories: list[str]) -> None:
        self.categories = set(categories)

    def is_valid(self) -> bool:
        return {"MAIN", "SETUP_WIZARD"}.issubset(self.categories) and "HOME" not in self.categories


def main() -> None:
    print("=== Setup Wizard 状态 ===")
    state = ProvisioningState(0, 0)
    print("首次启动入口：", state.next_home())
    print("完成后入口：", state.complete().next_home())
    print("向导 Intent 有效：", SetupIntentValidator(["MAIN", "DEFAULT", "SETUP_WIZARD"]).is_valid())


if __name__ == "__main__":
    main()
