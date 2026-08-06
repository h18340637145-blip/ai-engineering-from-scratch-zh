# Binder、类加载、构建与安装流程模拟
# 课程文档：phases/21-java-android-foundations/12-binder-classloading-build-and-install/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 Binder、类加载、APK/AAB 构建与安装章节
# 将 Binder 事务大小和现代 Android 构建顺序表达为可测试规则。

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BinderTransaction:
    bytes_count: int
    max_bytes: int = 1_048_576

    def is_safe(self) -> bool:
        return self.bytes_count <= self.max_bytes


def choose_class_loader(source: str) -> str:
    if source == "installed-app":
        return "PathClassLoader"
    if source == "trusted-plugin":
        return "DexClassLoader"
    raise ValueError("动态代码仅应来自可信来源")


class BuildPipeline:
    @staticmethod
    def steps() -> list[str]:
        return ["merge manifest and resources", "AAPT2", "compile sources", "D8", "R8", "package", "sign and zipalign"]


def main() -> None:
    print("=== Binder 与构建流水线 ===")
    print("64 KB Binder 事务安全：", BinderTransaction(64 * 1024).is_safe())
    print("安装包类加载器：", choose_class_loader("installed-app"))
    print("构建：", " -> ".join(BuildPipeline.steps()))


if __name__ == "__main__":
    main()
