# Skill: Launcher3 系统桌面基础

## 用途

快速判断一个 Android 应用是否声明了成为系统桌面的能力。

## 关键检查点

1. `AndroidManifest.xml` 是否包含 `CATEGORY_HOME` 声明
2. 是否同时声明了 `ACTION_MAIN`
3. 是否安装在系统分区（`/system_ext/priv-app`）并持有平台签名

## 快速命令

```bash
# 查询系统中所有声明了 CATEGORY_HOME 的 Activity
adb shell cmd package query-activities \
  -a android.intent.action.MAIN \
  -c android.intent.category.HOME

# 确认当前默认桌面
adb shell cmd package resolve-activity \
  -a android.intent.action.MAIN \
  -c android.intent.category.HOME
```

## 核心结论

- Launcher3 是应用层系统桌面，不是 Framework 本身
- `CATEGORY_HOME` 是成为系统桌面的唯一必要 Manifest 声明
- Launcher3QuickStep = Launcher3 + 最近任务 + 全面屏手势
