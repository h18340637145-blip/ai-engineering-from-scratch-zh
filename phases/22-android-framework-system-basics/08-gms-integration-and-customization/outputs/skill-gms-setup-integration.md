# GMS 与 Setup Wizard 集成检查卡

## 使用时机

产品预置 GMS 或自定义 Setup Wizard 需要与合作方流程衔接时使用。

## 检查顺序

1. 使用符合授权和设备认证的正式交付包。
2. 为每个 APK 声明模块名、签名、分区和特权属性。
3. XML 通过 `prebuilt_etc` 安装，产品包列表用 `+=` 追加。
4. 按实际向导版本提供 Partner Customization 接收器和资源。
5. 在 Wizard Script 中明确每个 action、result code 与下一步。
