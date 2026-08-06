# Android Framework 集成验收卡

## 使用时机

完成启动、Overlay、向导或权限定制后，准备在设备上验收时使用。

## 验收顺序

1. events buffer：启动与 bootanimation 事件。
2. `ps` 与 socket：Zygote、SystemServer 与应用进程。
3. `cmd overlay list`：当前用户资源覆盖状态。
4. `settings get`：设备和用户 provisioning 状态。
5. `dumpsys package`：签名、权限和组件状态。
6. 每次只修复最早失败的一项，重启后重新采集证据。
