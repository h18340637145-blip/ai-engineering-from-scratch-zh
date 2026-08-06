# Partner APK 与 Overlay：非侵入式定制布局和壁纸

**Type:** Build
**Languages:** Python
**Prerequisites:** 05-default-layout
**Time:** ~45 分钟

## 学习目标

- 解释 Partner APK 的发现机制（PackageManager Receiver 查询）
- 说明 Overlay APK 与 Partner APK 的区别
- 编写 Partner APK 的 Receiver 声明和资源文件结构
- 验证 Partner APK 是否被 Launcher3 发现
- 分析 Partner 壁纸不显示的常见原因

## 概念

Partner APK 是 Launcher3 提供的非侵入式定制机制。OEM 可以通过一个独立的系统 APK 提供自定义壁纸和布局，而不需要修改 Launcher3 源码。

### Partner APK 发现流程

Launcher3 在启动时查询所有注册了特定 Action 的 BroadcastReceiver：

```java
// 查询 Action
String PARTNER_PLACEMENT_METADATA_ACTION =
    "com.android.launcher3.action.PARTNER_CUSTOMIZATION";
```

设备上验证：
```bash
adb shell cmd package query-receivers \
  -a com.android.launcher3.action.PARTNER_CUSTOMIZATION
```

### Partner APK 必要条件

1. 是系统应用（安装在 `/system/app` 或 `/system_ext/priv-app`）
2. 注册了 `PARTNER_CUSTOMIZATION` Action 的 BroadcastReceiver
3. 资源数组名与 Launcher3 源码中的常量一致

### 壁纸资源结构

```xml
<!-- res/values/arrays.xml in Partner APK -->
<string-array name="partner_wallpapers">
    <item>@drawable/wallpaper_01</item>
    <item>@drawable/wallpaper_02</item>
</string-array>
```

### Overlay 与 Partner APK 对比

| 维度 | Partner APK | Resource Overlay |
|---|---|---|
| 修改源码 | 不需要 | 不需要 |
| 独立发布 | ✅ | ❌（需要与目标包一起） |
| 支持动态壁纸 | ❌ | ❌ |
| 布局定制 | ✅ | ✅ |

## 构建它

实现一个 Python 脚本，模拟 Launcher3 的 Partner APK 发现逻辑：给定一个已安装包列表，找出其中声明了 PARTNER_CUSTOMIZATION Action 的系统应用。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

## 发布它

输出技能见 `outputs/skill-partner-overlay.md`。
