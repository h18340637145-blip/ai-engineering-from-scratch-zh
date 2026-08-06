# 默认桌面布局：指定首次启动时的图标排列

> 数据库为空时 Launcher3 怎么知道要放什么图标？答案藏在一个 XML 文件里。

**Type:** Build
**Languages:** Python
**Prerequisites:** 04-startup-trace
**Time:** ~40 分钟

## 学习目标

- 定位 Launcher3 默认布局 XML 文件的路径和引用方式
- 解释默认布局只在数据库首次创建时生效的原因
- 编写合法的默认布局 XML，添加应用图标和 Hotseat 图标
- 通过清除 Launcher 数据验证布局变更
- 分析布局 XML 解析失败的常见原因（坐标越界、组件名拼写错误）

## 概念

### 默认布局 XML

默认布局文件路径：

```text
packages/apps/Launcher3/res/xml/default_workspace_*.xml
```

通过 `launcher_layout` 属性引用：

```xml
<!-- res/values/config.xml -->
<string name="default_workspace_file">@xml/default_workspace_5x5</string>
```

### 布局 XML 结构

```xml
<favorites xmlns:launcher="http://schemas.android.com/apk/res-auto/com.android.launcher3">

    <!-- 桌面第一页图标：Settings at (0,0) -->
    <resolve
        launcher:container="-100"
        launcher:screen="0"
        launcher:x="0"
        launcher:y="0">
        <favorite
            launcher:packageName="com.android.settings"
            launcher:className="com.android.settings.Settings" />
    </resolve>

    <!-- Hotseat 图标：拨号在位置 0 -->
    <resolve
        launcher:container="-101"
        launcher:screen="0"
        launcher:x="0"
        launcher:y="0">
        <favorite
            launcher:packageName="com.android.dialer"
            launcher:className="com.android.dialer.DialtactsActivity" />
    </resolve>

</favorites>
```

### 关键规则

1. **只在首次创建数据库时导入**——修改布局 XML 后必须执行 `adb shell pm clear com.android.launcher3` 才能验证
2. **坐标不能越界**——cellX + spanX 不能超过网格列数
3. **className 必须完整且存在**——拼写错误会导致该图标被跳过

### 验证流程

```bash
# 1. 修改 XML 文件后重新编译
m Launcher3QuickStep -j4

# 2. 推送 APK
adb root && adb remount
adb push out/.../Launcher3QuickStep.apk /system_ext/priv-app/...

# 3. 清除数据（让数据库重建）
adb shell pm clear com.android.launcher3

# 4. 按 Home 键启动，观察布局
adb shell input keyevent KEYCODE_HOME

# 5. 查询数据库验证
adb shell sqlite3 /data/data/com.android.launcher3/databases/launcher.db \
  "SELECT title, cellX, cellY, container FROM favorites;"
```

## 构建它

实现一个 Python 脚本，解析默认布局 XML（或其字典表示），校验每个 Item 坐标，并生成布局摘要报告。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

## 发布它

输出技能见 `outputs/skill-default-layout.md`。
