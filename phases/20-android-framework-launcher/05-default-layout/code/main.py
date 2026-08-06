# Launcher3 默认布局解析与校验
# 课程文档：phases/20-android-framework-launcher/05-default-layout/docs/en.md
# 参考：AOSP packages/apps/Launcher3/res/xml/default_workspace_*.xml
#       DefaultLayoutParser.java

"""
解析 Launcher3 默认桌面布局配置（字典格式，模拟 XML 解析结果），
校验每个 Item 的坐标合法性，并生成摘要报告。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


CONTAINER_DESKTOP = -100
CONTAINER_HOTSEAT = -101


@dataclass
class LayoutItem:
    """对应默认布局 XML 中的一个 <resolve> 节点。"""
    package_name: str
    class_name: str
    container: int       # -100=桌面, -101=Hotseat
    screen: int          # Workspace 页面或 Hotseat rank
    cell_x: int
    cell_y: int
    span_x: int = 1
    span_y: int = 1

    @property
    def location(self) -> str:
        loc = "桌面" if self.container == CONTAINER_DESKTOP else "Hotseat"
        return f"{loc}(screen={self.screen}, x={self.cell_x}, y={self.cell_y})"


@dataclass
class LayoutConfig:
    """模拟设备网格配置。"""
    columns: int
    rows: int
    hotseat_count: int   # Hotseat 支持的最大图标数


def validate_item(item: LayoutItem, cfg: LayoutConfig) -> Tuple[bool, Optional[str]]:
    """校验 LayoutItem 是否在配置范围内合法。"""
    if item.container == CONTAINER_DESKTOP:
        if item.cell_x < 0 or item.cell_y < 0:
            return False, "坐标不能为负数"
        if item.cell_x + item.span_x > cfg.columns:
            return False, f"X 越界 ({item.cell_x}+{item.span_x} > {cfg.columns})"
        if item.cell_y + item.span_y > cfg.rows:
            return False, f"Y 越界 ({item.cell_y}+{item.span_y} > {cfg.rows})"
    elif item.container == CONTAINER_HOTSEAT:
        if item.screen >= cfg.hotseat_count:
            return False, f"Hotseat 位置 {item.screen} 超出最大数 {cfg.hotseat_count}"
    else:
        return False, f"未知 container 值: {item.container}"
    return True, None


def parse_layout(items: List[LayoutItem], cfg: LayoutConfig) -> None:
    """打印布局解析和校验报告。"""
    desktop = [i for i in items if i.container == CONTAINER_DESKTOP]
    hotseat = [i for i in items if i.container == CONTAINER_HOTSEAT]

    print(f"=== 布局解析报告（{cfg.columns}列×{cfg.rows}行，Hotseat={cfg.hotseat_count}格）===\n")
    print(f"桌面 Item: {len(desktop)} 个   Hotseat Item: {len(hotseat)} 个\n")

    valid_count = 0
    for item in items:
        ok, err = validate_item(item, cfg)
        status = "✅" if ok else "❌"
        print(f"  {status} {item.package_name.split('.')[-1]:<20} {item.location}")
        if not ok:
            print(f"      原因: {err}")
        else:
            valid_count += 1

    print(f"\n合法: {valid_count} / {len(items)}")


# ── 演示数据 ─────────────────────────────────────────────────────────

def main() -> None:
    cfg = LayoutConfig(columns=5, rows=5, hotseat_count=5)
    items = [
        LayoutItem("com.android.settings", "com.android.settings.Settings",
                   CONTAINER_DESKTOP, screen=0, cell_x=0, cell_y=0),
        LayoutItem("com.android.calculator2", "com.android.calculator2.Calculator",
                   CONTAINER_DESKTOP, screen=0, cell_x=1, cell_y=0),
        LayoutItem("com.android.camera2", "com.android.camera2.CameraActivity",
                   CONTAINER_DESKTOP, screen=0, cell_x=4, cell_y=0, span_x=2),   # X 越界
        LayoutItem("com.android.dialer", "com.android.dialer.DialtactsActivity",
                   CONTAINER_HOTSEAT, screen=0, cell_x=0, cell_y=0),
        LayoutItem("com.android.contacts", "com.android.contacts.activities.PeopleActivity",
                   CONTAINER_HOTSEAT, screen=5, cell_x=0, cell_y=0),              # Hotseat 越界
        LayoutItem("com.android.mms", "com.android.mms.MmsActivity",
                   CONTAINER_HOTSEAT, screen=1, cell_x=0, cell_y=0),
    ]
    parse_layout(items, cfg)


if __name__ == "__main__":
    main()
