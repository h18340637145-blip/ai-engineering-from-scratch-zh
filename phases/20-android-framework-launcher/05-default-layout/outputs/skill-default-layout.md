# Skill: Launcher3 默认布局定制

## 核心流程

1. 修改 `res/xml/default_workspace_NxN.xml`
2. 编译 `m Launcher3QuickStep -j4`
3. `adb root && adb remount && adb sync system_ext`
4. `adb shell pm clear com.android.launcher3`
5. `adb shell input keyevent KEYCODE_HOME`
6. 验证：`sqlite3 /data/data/.../launcher.db "SELECT title,cellX,cellY FROM favorites;"`

## XML 节点速查

```xml
<!-- 桌面图标 -->
<resolve launcher:container="-100" launcher:screen="0" launcher:x="0" launcher:y="0">
    <favorite launcher:packageName="com.example" launcher:className="com.example.Main"/>
</resolve>

<!-- Hotseat 图标 -->
<resolve launcher:container="-101" launcher:screen="0" launcher:x="0" launcher:y="0">
    <favorite launcher:packageName="com.android.dialer" .../>
</resolve>
```
