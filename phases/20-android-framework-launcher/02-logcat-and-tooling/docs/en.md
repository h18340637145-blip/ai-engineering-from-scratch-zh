# Logcat 与调试工具：在 AOSP 中看见系统在做什么

> 在动手改源码之前，先学会读日志：logcat 是理解 Launcher3 运行时行为的第一工具。

**Type:** Build
**Languages:** Python
**Prerequisites:** 01-launcher3-intro
**Time:** ~40 分钟

## 学习目标

- 使用 `adb logcat` 的核心参数过滤 Launcher3 相关日志
- 区分 Verbose / Debug / Info / Warn / Error 五个日志等级
- 配合 `grep` 同时追踪多个 Tag 或排除噪声日志
- 设计一个可验证的日志埋点方案（Tag 命名 + 关键入口）
- 用 `adb shell` 命令查询 Launcher 进程、数据路径和组件状态

## 概念

### logcat 核心用法

```bash
# 清空历史日志
adb logcat -c

# 输出当前日志后退出（不持续等待）
adb logcat -d

# 只显示指定 Tag 的 Debug 级及以上日志
adb logcat -s LauncherStudy:D

# 同时追踪多个 Tag
adb logcat -d | grep -iE "LauncherStudy|LoaderTask|LoaderCursor"

# 排除触摸移动等噪声日志
adb logcat -d | grep -vE "ACTION_MOVE"
```

### 日志等级对照

| 缩写 | 等级 | 典型用途 |
|---|---|---|
| V | Verbose | 最详细，循环内每帧日志 |
| D | Debug | 开发调试，方法进出 |
| I | Info | 关键状态变化 |
| W | Warn | 可恢复的异常情况 |
| E | Error | 错误，通常需要关注 |

### 标准日志埋点模板

```java
private static final String STUDY_TAG = "LauncherStudy";

@Override
protected void onCreate(Bundle savedInstanceState) {
    Log.d(STUDY_TAG, "Launcher.onCreate begin");
    super.onCreate(savedInstanceState);
    Log.d(STUDY_TAG, "Launcher.onCreate end");
}
```

### 常用 adb shell 诊断命令

```bash
# 查看 Launcher 进程 ID
adb shell pidof com.android.launcher3

# 查看 Launcher APK 安装路径
adb shell pm path com.android.launcher3

# 强制停止 Launcher（下次按 Home 键时重启）
adb shell am force-stop com.android.launcher3

# 清除 Launcher 数据（下次启动重建数据库）
adb shell pm clear com.android.launcher3

# 模拟按下 Home 键
adb shell input keyevent KEYCODE_HOME

# 查询当前系统默认桌面组件
adb shell cmd package resolve-activity \
  -a android.intent.action.MAIN \
  -c android.intent.category.HOME
```

## 构建它

实现一个 Python 脚本，解析 logcat 文本并提取指定 Tag 的日志行，同时统计各等级分布——模拟 `adb logcat -s TAG:D` 的过滤行为。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

输出示例：

```text
=== 过滤 Tag: LauncherStudy ===
D  LauncherStudy: Launcher.onCreate begin
D  LauncherStudy: Launcher.setupViews begin
D  LauncherStudy: LoaderTask.run begin

=== 日志等级统计 ===
D: 3 条
W: 0 条
E: 0 条
```

## 发布它

输出技能见 `outputs/skill-logcat-workflow.md`。
