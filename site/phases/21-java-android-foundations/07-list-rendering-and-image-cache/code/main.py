# RecyclerView 与图片缓存策略模拟
# 课程文档：phases/21-java-android-foundations/07-list-rendering-and-image-cache/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 RecyclerView、图片缓存与性能章节
# 模型展示列表局部更新和 memory → disk → network 的缓存读取顺序。

from __future__ import annotations


class ListPerformanceAdvisor:
    @staticmethod
    def configure(nested: bool, fixed_size: bool) -> list[str]:
        actions: list[str] = []
        if nested:
            actions.append("shared RecycledViewPool")
        if fixed_size:
            actions.append("setHasFixedSize(true)")
        return actions

    @staticmethod
    def update_strategy(changed_items: int) -> str:
        return "notifyItem range" if changed_items > 0 else "no update"

    @staticmethod
    def uses_soft_reference_as_primary_cache() -> bool:
        return False


class ImageCache:
    def __init__(self, disk: set[str] | None = None) -> None:
        self.memory: set[str] = set()
        self.disk = set(disk or set())

    def fetch(self, key: str) -> str:
        if key in self.memory:
            return "memory"
        if key in self.disk:
            self.memory.add(key)
            return "disk"
        self.disk.add(key)
        self.memory.add(key)
        return "network"


def main() -> None:
    print("=== 列表与图片缓存 ===")
    print("嵌套固定列表：", ListPerformanceAdvisor.configure(True, True))
    cache = ImageCache()
    print("首次读取 avatar：", cache.fetch("avatar"))
    print("再次读取 avatar：", cache.fetch("avatar"))


if __name__ == "__main__":
    main()
