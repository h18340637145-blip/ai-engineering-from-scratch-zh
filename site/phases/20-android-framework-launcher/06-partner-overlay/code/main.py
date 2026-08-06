# Partner APK 发现逻辑模拟
# 课程文档：phases/20-android-framework-launcher/06-partner-overlay/docs/en.md
# 参考：AOSP packages/apps/Launcher3/src/com/android/launcher3/partner/PartnerProvider.java

"""
模拟 Launcher3 发现 Partner APK 的核心逻辑：
查询所有已安装系统应用，找出声明了 PARTNER_CUSTOMIZATION Action 的应用。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


PARTNER_ACTION = "com.android.launcher3.action.PARTNER_CUSTOMIZATION"


@dataclass
class ReceiverInfo:
    action: str
    enabled: bool = True


@dataclass
class PackageInfo:
    package: str
    label: str
    is_system: bool
    receivers: List[ReceiverInfo] = field(default_factory=list)

    def has_partner_action(self) -> bool:
        return any(
            r.action == PARTNER_ACTION and r.enabled
            for r in self.receivers
        )

    def is_valid_partner(self) -> bool:
        """Partner APK 必须是系统应用且注册了正确的 Action。"""
        return self.is_system and self.has_partner_action()


def discover_partners(packages: List[PackageInfo]) -> List[PackageInfo]:
    """模拟 PackageManager.queryBroadcastReceivers() 查找 Partner APK。"""
    return [p for p in packages if p.is_valid_partner()]


def main() -> None:
    packages = [
        PackageInfo("com.android.launcher3", "Launcher3QuickStep", is_system=True),
        PackageInfo("com.oem.partner", "OEM Partner", is_system=True,
                    receivers=[ReceiverInfo(PARTNER_ACTION)]),
        PackageInfo("com.user.app", "用户安装应用", is_system=False,
                    receivers=[ReceiverInfo(PARTNER_ACTION)]),   # 非系统应用，无效
        PackageInfo("com.oem.disabled", "禁用 Partner", is_system=True,
                    receivers=[ReceiverInfo(PARTNER_ACTION, enabled=False)]),
        PackageInfo("com.android.settings", "设置", is_system=True),
    ]

    print("=== Partner APK 发现扫描 ===\n")
    for pkg in packages:
        valid = pkg.is_valid_partner()
        status = "✅ Partner APK" if valid else "❌"
        print(f"  {status} {pkg.label} ({pkg.package})")

    partners = discover_partners(packages)
    print(f"\n发现 {len(partners)} 个有效 Partner APK")
    for p in partners:
        print(f"  → {p.package}")


if __name__ == "__main__":
    main()
