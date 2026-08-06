# Skill: Launcher3 启动链路追踪

## 快速追踪步骤

1. 清空日志：`adb logcat -c`
2. 强制重启 Launcher：`adb shell am force-stop com.android.launcher3 && adb shell input keyevent KEYCODE_HOME`
3. 捕获启动日志：`adb logcat -d | grep LauncherStudy`
4. 用本课 `main.py` 分析耗时

## 关键调用链（按顺序）

```text
Launcher.onCreate → setupViews → LauncherModel.startLoader
    → LoaderTask.run → loadWorkspace → bindWorkspace
    → loadAllApps → bindAllApplications → finishBindingItems
```

## 异常判断

| 现象 | 可能原因 |
|---|---|
| finishBindingItems 未出现 | LoaderTask 抛出异常 |
| bindWorkspace 很慢 | 数据库 Item 过多或有越界项 |
| startLoader 未出现 | LauncherModel 初始化失败 |
