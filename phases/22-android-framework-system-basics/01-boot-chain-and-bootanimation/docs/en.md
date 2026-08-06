# 系统启动与开机动画：从 Bootloader 到桌面

> “开机画面卡住”不是一个问题，而是一条跨 bootloader、init、SurfaceFlinger、Zygote 和 WindowManager 的时间线。

**Type:** Build
**Languages:** Python
**Prerequisites:** 20-android-framework-launcher
**Time:** ~90 分钟

![Android Framework 系统基础阶段封面](assets/android-framework-system-cover.png)

## 学习目标

- 列出从上电到 Launcher 的核心启动阶段
- 区分 bootloader、Kernel、init 与 Android 用户空间画面的来源
- 解释 `bootanimation` 的启动和结束条件
- 解析 `bootanimation.zip` 的 `desc.txt`
- 用 Event Log 定位动画停止与结束之间的停顿

## 概念

源资料 `docs/AndroidFramework/Android Framework 基础.md` 强调：不同启动阶段的画面来源不同，不能只凭固定路径判断。Android 用户空间动画由 `/system/bin/bootanimation` 读取 ZIP，并在 SurfaceFlinger 可用后绘制。

```mermaid
flowchart LR
    A[Boot ROM / Bootloader] --> B[Linux Kernel]
    B --> C[init PID 1]
    C --> D[SurfaceFlinger 与 Native 服务]
    D --> E[bootanimation]
    E --> F[Zygote]
    F --> G[system_server]
    G --> H[SystemUI / Launcher / Setup Wizard]
```

`desc.txt` 首行是宽、高和帧率。后续段的 `p` 可以在启动完成时中断，`c` 必须播完当前段。图像按名字典序播放，ZIP 应使用仅存储方式，避免运行时解压开销。

## 构建它

`BootAnimationSpec.parse()` 解析 `desc.txt`；`BootTimeline.find_gap()` 用 `stop_bootanim` 与 `wm_boot_animation_done` 的时间差发现可能的退出卡点。

```bash
cd phases/22-android-framework-system-basics/01-boot-chain-and-bootanimation/code
python3 main.py
python3 -m unittest discover tests -v
```

## 诊断练习

先读取 `adb logcat -b events -v threadtime` 中的 `boot_progress` 和 `wm_boot_animation_done`，再看 SurfaceFlinger、动画 ZIP 路径与当前产品分区。不要在未确认设备实现前修改启动 Logo 路径。

## 发布它

开机动画排查卡见 `outputs/skill-bootanimation-trace.md`。
