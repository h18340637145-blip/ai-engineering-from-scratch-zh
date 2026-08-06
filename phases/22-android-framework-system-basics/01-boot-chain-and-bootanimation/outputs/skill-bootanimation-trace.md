# 开机动画追踪卡

## 使用时机

设备启动卡在 Logo、黑屏或 bootanimation 时使用。

## 检查顺序

1. 区分 bootloader、Kernel、init 和 Android 用户空间画面来源。
2. 搜索当前源码和分区中的 `bootanimation`、`logo`、`splash`。
3. 解析 `desc.txt` 的宽高帧率和 p/c 段。
4. 用 events buffer 关联 `stop_bootanim` 与 `wm_boot_animation_done`。
5. 量产版本通过产品配置安装 ZIP，不依赖手动 push。
