# 文件夹与小组件数据模型
# 课程文档：phases/20-android-framework-launcher/10-folders-and-widgets/docs/en.md
# 参考：AOSP packages/apps/Launcher3/src/com/android/launcher3/FolderInfo.java

"""
模拟 Launcher3 文件夹的数据库操作：创建文件夹、添加图标、重命名。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


CONTAINER_DESKTOP = -100
ITEM_TYPE_APPLICATION = 0
ITEM_TYPE_FOLDER = 2


@dataclass
class AppItem:
    id: int
    title: str
    package_name: str
    container: int  # 桌面 ID 或文件夹 ID
    cell_x: int
    cell_y: int
    item_type: int = ITEM_TYPE_APPLICATION


@dataclass
class FolderItem:
    id: int
    title: str
    container: int  # 桌面 ID
    cell_x: int
    cell_y: int
    item_type: int = ITEM_TYPE_FOLDER
    children: List[AppItem] = field(default_factory=list)

    def add_child(self, app: AppItem) -> None:
        """模拟图标拖入文件夹：更新 container 字段。"""
        app.container = self.id
        self.children.append(app)

    def rename(self, new_title: str) -> None:
        """模拟文件夹重命名：只更新 title 字段。"""
        old = self.title
        self.title = new_title
        print(f"  DB UPDATE favorites SET title='{new_title}' WHERE id={self.id} (原: '{old}')")


class LauncherDatabase:
    """简化版 launcher.db 操作模拟。"""

    def __init__(self) -> None:
        self._next_id = 1
        self.items: Dict[int, object] = {}

    def _new_id(self) -> int:
        iid = self._next_id
        self._next_id += 1
        return iid

    def create_folder(self, title: str, cell_x: int, cell_y: int) -> FolderItem:
        folder = FolderItem(id=self._new_id(), title=title,
                            container=CONTAINER_DESKTOP, cell_x=cell_x, cell_y=cell_y)
        self.items[folder.id] = folder
        print(f"  DB INSERT folder id={folder.id} title='{title}' ({cell_x},{cell_y})")
        return folder

    def create_app(self, title: str, pkg: str, x: int, y: int) -> AppItem:
        app = AppItem(id=self._new_id(), title=title, package_name=pkg,
                      container=CONTAINER_DESKTOP, cell_x=x, cell_y=y)
        self.items[app.id] = app
        return app


def main() -> None:
    db = LauncherDatabase()
    print("=== 文件夹操作模拟 ===\n")

    # 创建两个应用图标
    settings = db.create_app("设置", "com.android.settings", 0, 0)
    calculator = db.create_app("计算器", "com.android.calculator2", 1, 0)

    # 创建文件夹
    folder = db.create_folder("工具", 2, 0)

    # 将图标拖入文件夹
    print("\n  [拖入文件夹]")
    folder.add_child(settings)
    folder.add_child(calculator)
    print(f"  DB UPDATE favorites SET container={folder.id} WHERE id IN ({settings.id}, {calculator.id})")

    # 重命名文件夹
    print("\n  [重命名文件夹]")
    folder.rename("实用工具")

    # 显示最终状态
    print(f"\n=== 最终状态 ===")
    print(f"  文件夹: id={folder.id}, title='{folder.title}', 子图标数={len(folder.children)}")
    for child in folder.children:
        print(f"    - {child.title} (container={child.container})")


if __name__ == "__main__":
    main()
