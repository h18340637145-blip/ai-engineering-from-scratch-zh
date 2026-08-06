# Android 架构、性能与跨进程存储决策模拟
# 课程文档：phases/21-java-android-foundations/11-architecture-performance-and-storage/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 MVC/MVP/MVVM、性能与 SharedPreferences 章节
# 模型强调先测量再优化，以及 SharedPreferences 不提供多进程一致性。

from __future__ import annotations


class ArchitectureAdvisor:
    @staticmethod
    def recommend(requirement: str) -> str:
        mapping = {
            "observable-state": "MVVM",
            "passive-view": "MVP",
            "simple-controller": "MVC",
        }
        if requirement not in mapping:
            raise ValueError("未知架构需求")
        return mapping[requirement]


class OptimizationPlan:
    @staticmethod
    def next_action(has_measurement: bool) -> str:
        return "optimize" if has_measurement else "measure"


def cross_process_storage(candidate: str) -> str:
    if candidate == "SharedPreferences":
        return "avoid"
    if candidate in {"ContentProvider", "Binder/AIDL", "single-writer"}:
        return "valid"
    return "review"


def main() -> None:
    print("=== 架构与性能优化 ===")
    print("可观察 UI 状态：", ArchitectureAdvisor.recommend("observable-state"))
    print("未知热点：", OptimizationPlan.next_action(False))
    print("跨进程偏好：", cross_process_storage("SharedPreferences"))


if __name__ == "__main__":
    main()
