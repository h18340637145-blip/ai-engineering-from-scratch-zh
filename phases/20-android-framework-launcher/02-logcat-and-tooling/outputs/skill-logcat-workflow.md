# Skill: Launcher3 Logcat 调试工作流

## 标准分析流程

1. `adb logcat -c` — 清空历史日志
2. 操作设备复现问题
3. `adb logcat -d | grep -iE "LauncherStudy|LoaderTask"` — 捕获目标日志
4. 分析调用顺序和异常信息

## 常用埋点 Tag

| Tag | 用途 |
|---|---|
| LauncherStudy | 自定义学习/调试入口 |
| LoaderTask | 数据加载线程 |
| LoaderCursor | 数据库读取校验 |
| PagedViewStudy | 桌面分页滑动追踪 |

## 速查命令

```bash
adb logcat -c
adb logcat -s LauncherStudy:D
adb logcat -d | grep -iE "LauncherStudy|LoaderTask|LoaderCursor"
adb shell pidof com.android.launcher3
adb shell pm clear com.android.launcher3
```
