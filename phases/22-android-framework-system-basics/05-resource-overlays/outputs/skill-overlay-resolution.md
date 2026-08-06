# SRO/RRO Overlay 解析卡

## 使用时机

资源未生效、导航模式切换冲突或多个产品 Overlay 叠加时使用。

## 检查顺序

1. 固定产品资源用 SRO；运行时主题或模式切换用 RRO。
2. 核对 targetPackage、可选 targetName、isStatic 与 priority。
3. 确认覆盖文件相对目标资源路径一致。
4. 用 `cmd overlay list --user current` 查看当前状态。
5. 同类别模式使用独占启用，避免多个 RRO 同时生效。
