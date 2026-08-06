# Android 模块分区审查卡

## 使用时机

新增预置应用、Native 模块或配置文件，或遇到 artifact path requirement 错误时使用。

## 检查顺序

1. 参考同目录模块决定 Android.mk 或 Android.bp 写法。
2. 明确模块类型、安装分区、证书和 privileged 标记。
3. 核对 system、system_ext、product、vendor 的产品策略。
4. 以目标模块构建后检查真实安装路径。
5. 不用 allow list 掩盖原本的分区归属错误。
