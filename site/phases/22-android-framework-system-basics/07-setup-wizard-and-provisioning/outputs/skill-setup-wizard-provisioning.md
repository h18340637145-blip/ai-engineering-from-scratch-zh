# Setup Wizard Provisioning 检查卡

## 使用时机

设备重复进入向导、首次启动误进 Launcher 或自定义向导与桌面冲突时使用。

## 检查顺序

1. 读取 `device_provisioned` 和 `user_setup_complete`。
2. 核对 Activity 的 MAIN、DEFAULT 与 SETUP_WIZARD 类别。
3. 不为提高匹配率盲目增加 HOME。
4. 完成向导时写入两类状态并 `finish()`。
5. 在可恢复设备完成恢复出厂、重启和跳过/返回场景测试。
