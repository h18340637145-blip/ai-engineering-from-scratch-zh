# Android Framework 基础

**\#\# 17\. 系统启动与开机动画**



Android 从上电到桌面通常经历下面几个阶段：



```Plain Text
Boot ROM / Bootloader
    -> Linux Kernel
    -> init (PID 1)
    -> native 服务（如 SurfaceFlinger）
    -> bootanimation
    -> zygote
    -> system_server
    -> SystemUI / Launcher / Setup Wizard
```



**\#\#\# 启动画面的来源**



屏幕上看到的“开机画面”不一定来自同一套代码，应先分阶段确认：



|阶段|常见画面来源|说明|
|---|---|---|
|Bootloader|厂商 Logo、充电图、Verified Boot 提示|通常由 bootloader 或固件提供，不属于 AOSP Framework。|
|Kernel|framebuffer Logo 或 DRM splash|是否存在、源码目录和资源格式由内核及厂商实现决定。历史代码可能在 `drivers/video/logo/`；高通产品也常有厂商显示链路。|
|`init` 早期|黑屏、保留上一阶段画面或厂商静态画面|现代 AOSP 通常没有通用的 `system/core/init/logo.c` 负责绘制 Logo；必须按当前分支和设备实现确认。|
|Android 用户空间|`bootanimation` 动画|SurfaceFlinger 可用后，由 `/system/bin/bootanimation` 渲染 ZIP 动画。|



定位时不要只按网络文章中的固定路径判断。可在源码树中搜索 `bootanimation`、`logo`、`splash`，并在设备上检查实际挂载点和文件：



```Bash
adb shell getprop | grep -E 'bootanim|boot_completed'
adb shell ls -l /system/media /product/media /oem/media 2>/dev/null
adb shell find /system /product /vendor -name bootanimation.zip 2>/dev/null
```



**\#\#\# \`bootanimation\` 启动与结束流程**



1. Kernel 启动 `init`，`init` 解析 `init.rc` 及其导入的 rc 文件。

2. `init` 启动 SurfaceFlinger；其入口通常是 `frameworks/native/services/surfaceflinger/main_surfaceflinger.cpp`。

3. SurfaceFlinger 初始化显示合成服务，并在合适时机设置与 boot animation 相关的系统属性。

4. rc 中的 `bootanim` service 由属性触发，执行 `/system/bin/bootanimation`。

5. `BootAnimation` 读取 `bootanimation.zip`，解析 `desc.txt` 并逐帧绘制。

6. SystemServer/WindowManager 在系统已可交互时停止动画；`bootanimation` 观察退出条件后结束，WindowManager 记录启动完成事件。

不同 Android 版本中类名、属性写入点和 rc 条件可能变化。常见跟踪点如下：



```Plain Text
frameworks/native/services/surfaceflinger/main_surfaceflinger.cpp
frameworks/native/services/surfaceflinger/SurfaceFlinger.cpp
frameworks/base/cmds/bootanimation/BootAnimation.cpp
system/core/rootdir/init*.rc
```



`event.log` 中可重点关联以下事件，按时间戳判断卡在哪一阶段：



```Bash
adb logcat -b events -v threadtime | grep -E 'boot_progress|wm_boot_animation_done'
```



- `boot_progress...stop_bootanim`：系统请求结束 boot animation 的时机。

- `wm_boot_animation_done`：WindowManager 记录动画结束；若两者间隔很大，应继续检查 bootanimation 退出和 SurfaceFlinger 状态。

**\#\#\# 制作 \`bootanimation\.zip\`**



常见安装位置是 `/system/media/bootanimation.zip`，也可能是 `/product/media/bootanimation.zip`。优先级和实际使用位置以当前 `BootAnimation` 源码及设备分区内容为准。



目录结构：



```Plain Text
bootanimation.zip
├── desc.txt
├── part0/
│   ├── 00000.png
│   └── ...
└── part1/
    ├── 00000.png
    └── ...
```



`desc.txt` 的首行是 `宽 高 帧率`；后续每行描述一个动画段：



```Plain Text
1080 1920 30
p 1 0 part0
p 0 0 part1
c 1 0 part2
```



|字段|含义|
|---|---|
|`TYPE`|`p` 可在启动完成时中断；`c` 必须播放完当前段。|
|`COUNT`|播放次数；`0` 表示持续循环，直到启动完成或被中断。|
|`PAUSE`|本段结束后的停留帧数。|
|`PATH`|帧图片目录，例如 `part0`。|
|`#RRGGBB`|可选背景色。|
|`CLOCK`|可选时钟绘制位置，具体支持情况取决于版本和设备。|



图片一般按文件名字典序播放。为避免启动时解压开销，ZIP 必须使用“仅存储”方式打包：



```Bash
zip -r -X -Z store bootanimation.zip desc.txt part*/*
unzip -lv bootanimation.zip
```



快速验证：



```Bash
adb root
adb remount
adb push bootanimation.zip /system/media/bootanimation.zip
adb reboot
```



量产版本应通过产品配置安装，而不是手动 push：



```Makefile
PRODUCT_COPY_FILES += \
    vendor/<company>/bootanimation/bootanimation.zip:system/media/bootanimation.zip
```



若构建报 artifact path requirement 错误，说明模块/产品分区归属不一致。优先将文件放入正确分区；确有合理原因时才在对应产品 mk 中显式允许该产物路径：



```Makefile
PRODUCT_ARTIFACT_PATH_REQUIREMENT_ALLOWED_LIST += \
    system/media/bootanimation.zip
```



**\#\# 18\. Native 日志与调用栈**



**\#\#\# 打开 native 调试日志**



在采用 `LOG_NDEBUG` 的 C/C\+\+ 文件中，`0` 表示不禁用 debug 日志，`1` 表示禁用：



```C++
#define LOG_NDEBUG 0
#include <utils/CallStack.h>
#include <utils/Log.h>
```



打印当前调用栈：



```C++
android::CallStack stack;
stack.update();
stack.log("MyNativeTag");
```



调用栈是否完整取决于二进制是否保留符号、优化级别以及运行环境。模块使用 `Android.bp` 时按实际依赖加入 `shared_libs`，例如：



```ABAP
shared_libs: [
    "libutils",
    "liblog",
],
```



不要机械地添加 `libcutils`；只有源代码确实引用其 API 时才加入。修改后应只编译目标模块，并用 `adb logcat -s MyNativeTag` 验证。



**\#\# 19\. Zygote 启动流程**



Android 的所有应用进程和大多数 Java 系统进程都源自 Zygote；从 Linux 进程树看，它们都是 `init` 的后代。



```Plain Text
init
  -> zygote / zygote64
       -> system_server
       -> application process
```



**\#\#\# 启动步骤**



1. `init` 读取 `system/core/rootdir/init.rc`。

2. rc 根据 `ro.zygote` 导入相应配置，例如 `init.zygote64_32.rc`。

3. `init` 创建 `zygote` service，运行 `/system/bin/app_process64` 或 `/system/bin/app_process32`。

4. `app_process` 进入 Zygote，预加载常用类和资源，创建 Unix domain socket。

5. Zygote 根据 `--start-system-server` fork `system_server`；AMS 等服务请求时再 fork 应用进程。

典型 rc 配置（不同版本会有差异）：



```Plain Text
service zygote /system/bin/app_process64 -Xzygote /system/bin --zygote \
    --start-system-server --socket-name=zygote
    class main
    socket zygote stream 660 root system
    onrestart restart audioserver
    onrestart restart cameraserver
    onrestart restart netd
```



检查设备状态：



```Bash
adb shell ps -A | grep zygote
adb shell ls -l /dev/socket/zygote*
adb logcat -b system -v threadtime | grep -E 'Zygote|SystemServer'
```



**\#\# 20\. Android 构建与分区配置**



**\#\#\# 初始化与常用命令**



在 Android 源码根目录执行：



```Bash
source build/envsetup.sh
lunch <product>-<variant>
m -j"$(nproc)"
```



`lunch` 会选择产品和编译变体并设置环境变量。常见变体包括 `user`、`userdebug` 和 `eng`；具体产品名称以项目定义为准。构建产物通常位于 `out/target/product/<product>/`。



**\#\#\# \`Android\.mk\` 基础**



`Android.mk` 是 Make 构建描述；新模块优先遵循项目是否已迁移到 Soong 的规范。



```Makefile
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := Demo
LOCAL_SRC_FILES := Demo.apk
LOCAL_MODULE_CLASS := APPS
LOCAL_MODULE_TAGS := optional
LOCAL_CERTIFICATE := PRESIGNED
LOCAL_PRIVILEGED_MODULE := false

include $(BUILD_PREBUILT)
```



常见属性：



|属性|作用|
|---|---|
|`LOCAL_MODULE`|构建模块名。|
|`LOCAL_SRC_FILES`|相对 `LOCAL_PATH` 的输入文件。|
|`LOCAL_CERTIFICATE`|签名配置，例如 `platform`、`shared`、`PRESIGNED`。|
|`LOCAL_PRIVILEGED_MODULE`|`true` 时安装为 `priv-app`。|
|`LOCAL_SYSTEM_EXT_MODULE` / `LOCAL_PRODUCT_MODULE` / `LOCAL_VENDOR_MODULE`|指定模块所在分区，需与产品分区策略一致。|
|`LOCAL_OVERRIDES_PACKAGES`|声明替代的模块，避免这些模块同时安装。|



`/system/priv-app` 中的应用可申请部分仅限特权应用的权限，但仍需平台签名、权限白名单和 SELinux 等条件共同满足；把 APK 放入该目录并不会自动获得所有权限。



**\#\#\# \`Android\.bp\` 基础**



`Android.bp` 是 Soong 的声明式构建文件，不支持 Make 风格的任意流程控制。每个模块以类型开头，必须具备 `name`。



```ABAP
android_app {
    name: "ExampleApp",
    srcs: ["src/**/*.java"],
    manifest: "AndroidManifest.xml",
    certificate: "platform",
    privileged: true,
    system_ext_specific: true,
    platform_apis: true,
}
```



常见模块类型：



|类型|产物|
|---|---|
|`android_app`|APK|
|`cc_binary`|Native 可执行文件|
|`cc_library_shared` / `cc_library_static`|Native 动态库 / 静态库|
|`java_library`|Java 库|
|`prebuilt_etc`|预置配置文件|



编写新模块时优先参考同目录相近模块的签名、分区和依赖写法；不要仅因示例可编译就混用 `system`、`system_ext`、`product` 和 `vendor`。



**\#\# 21\. 资源 Overlay**



Overlay 用于替换目标包的资源，不应与“修改业务代码”混为一谈。



|类型|生效时机|适用场景|
|---|---|---|
|SRO（Static Resource Overlay）|编译/打包时|产品固定定制，例如默认资源、尺寸、字符串。|
|RRO（Runtime Resource Overlay）|运行时|可启停的主题、导航模式或产品配置。|



**\#\#\# SRO**



产品 mk 中声明 Overlay 目录：



```Makefile
PRODUCT_PACKAGE_OVERLAYS += \
    vendor/<company>/overlay
```



Overlay 目录必须与目标资源的相对路径一致，例如覆盖 framework 中某个 `res/values/config.xml` 的值。Android 版本和产品配置也可能自带 `DEVICE_PACKAGE_OVERLAYS`；移植时先确认其优先级，避免多个 Overlay 意外覆盖同一资源。



**\#\#\# RRO**



RRO APK 的 manifest 指向目标包：



```XML
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.overlay">
    <application android:hasCode="false" />
    <overlay
        android:targetPackage="com.example.target"
        android:targetName="optional_overlayable_name"
        android:isStatic="true"
        android:priority="1" />
</manifest>
```



- `android:targetPackage`：被覆盖的包名。

- `android:targetName`：目标包声明的 `overlayable` 分组名；目标未采用分组时不填写。

- `android:isStatic="true"`：静态 RRO，默认启用且通常不可运行时关闭。

- `android:priority`：同一资源存在多个静态 RRO 时的优先级，数值越大优先级越高。

查看并控制 RRO：



```Bash
adb shell cmd overlay list --user current
adb shell cmd overlay enable --user current <overlay-package>
adb shell cmd overlay disable --user current <overlay-package>
```



对于导航模式等同类别 Overlay，系统通常使用 `IOverlayManager.setEnabledExclusiveInCategory()`，保证该类别内只有一个 Overlay 被启用。



**\#\# 22\. 系统属性与跨进程设置**



**\#\#\# System Property**



Java 层通过隐藏/系统 API `android.os.SystemProperties` 访问系统属性；底层由 property service 管理。



```Java
String value = SystemProperties.get("persist.example.feature", "default");
SystemProperties.set("persist.example.feature", "enabled");
```



- `ro.*` 属性在启动后只读，不能通过普通方式修改。

- 属性名称、值长限制和 SELinux `property_contexts` 必须符合系统策略。

- 普通三方应用不应依赖隐藏 API；优先使用公开 API、Settings Provider 或 Binder 服务。

设备端检查：



```Bash
adb shell getprop persist.example.feature
adb shell setprop persist.example.feature enabled
```



**\#\#\# 使用 \`ContentObserver\` 监听 Settings**



当多个系统组件需共享配置时，可用 `Settings` \+ `ContentObserver`：



```Java
private static final String KEY = "example_key";

ContentObserver observer = new ContentObserver(new Handler(Looper.getMainLooper())) {
    @Override
    public void onChange(boolean selfChange) {
        int value = Settings.System.getInt(getContentResolver(), KEY, 0);
        Log.d(TAG, "new value=" + value);
    }
};

getContentResolver().registerContentObserver(
        Settings.System.getUriFor(KEY), false, observer);
```



页面或服务结束时必须注销：



```Java
getContentResolver().unregisterContentObserver(observer);
```



写入 `Settings.System`、`Settings.Secure` 或 `Settings.Global` 的权限和可写范围不同。`WRITE_SECURE_SETTINGS` 是签名/特权权限，系统应用还需要正确签名与 privapp allowlist，不能只设置 `sharedUserId`。调试命令：



```Bash
adb shell settings get system example_key
adb shell settings put system example_key 1
```



**\#\# 23\. 开机向导（Setup Wizard）定制**



**\#\#\# Provisioning 状态**



首次开机或恢复出厂设置后，系统根据 provisioning 状态选择开机向导和 Home。常见状态：



```Bash
adb shell settings get global device_provisioned
adb shell settings get secure user_setup_complete
```



- `device_provisioned=1`：设备级初始化完成。

- `user_setup_complete=1`：当前用户完成 Setup Wizard。

测试时修改这两个状态会改变系统启动路径，可能影响账户、策略和测试数据；应仅在可恢复的调试设备上操作。



**\#\#\# 声明 Setup Wizard Activity**



Activity 需要能响应向导类别；是否同时声明 `HOME` 取决于产品的实际流程：



```XML
<activity android:name=".WelcomeActivity" android:exported="true">
    <intent-filter android:priority="2">
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.DEFAULT" />
        <category android:name="android.intent.category.SETUP_WIZARD" />
    </intent-filter>
</activity>
```



`android.intent.category.SETUP_WIZARD` 是识别开机向导的关键。不要为了提高匹配概率盲目同时声明 `HOME`；这可能参与桌面解析并与 Launcher 冲突。



**\#\#\# 完成向导**



完成流程应写入状态并结束 Activity。是否禁用组件要看是否希望该向导未来还能被显式启动：



```Java
private void finishSetup() {
    Settings.Global.putInt(getContentResolver(), Settings.Global.DEVICE_PROVISIONED, 1);
    Settings.Secure.putInt(getContentResolver(), Settings.Secure.USER_SETUP_COMPLETE, 1);

    ComponentName component = new ComponentName(this, WelcomeActivity.class);
    getPackageManager().setComponentEnabledSetting(
            component,
            PackageManager.COMPONENT_ENABLED_STATE_DISABLED,
            PackageManager.DONT_KILL_APP);
    finish();
}
```



应使用 `finish()`，不建议在普通 Activity 中调用 `System.exit(0)`。系统 Settings 写入需要相应系统权限；应用身份、签名、privapp allowlist 和 SELinux 策略都必须正确。



**\#\#\# 在向导中切换导航模式**



导航模式由同一类别的 RRO 控制。系统应用可取得 `IOverlayManager` 后启用指定 Overlay：



```Java
IOverlayManager overlayManager = IOverlayManager.Stub.asInterface(
        ServiceManager.getService(Context.OVERLAY_SERVICE));
overlayManager.setEnabledExclusiveInCategory(
        NAV_BAR_MODE_GESTURAL_OVERLAY, UserHandle.USER_CURRENT);
```



实现前应参考当前版本 Settings 中的 `SystemNavigationGestureSettings` 和 `SystemNavigationPreferenceController`，直接复用当前分支定义的 Overlay 包名与模式判断逻辑。应用需要系统签名，并按产品策略授予 `android.permission.CHANGE_OVERLAY_PACKAGES` 等权限；仅在 manifest 声明权限不会自动获得授权。



**\#\# 24\. GMS 集成与自定义向导适配**



**\#\#\# 集成原则**



GMS 是受授权和兼容性要求约束的 Google 移动服务套件。量产产品应使用与 Android 版本、设备认证和分区策略匹配的正式交付包；不要把来源不明的 Open GApps 包直接用于商用镜像。



典型预置结构可按分区分类：



```Plain Text
vendor/<company>/gms/
├── app/                 # 普通预装 APK
├── priv-app/            # 需要特权权限的 APK
├── system/etc/
│   ├── default-permissions/
│   ├── permissions/
│   ├── preferred-apps/
│   └── sysconfig/
├── framework/            # 必需的共享库（如交付包要求）
└── gms.mk
```



每个预编译 APK 都要明确模块名、签名形式、安装分区和特权属性。以 `Android.mk` 为例：



```Makefile
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := GoogleContactsSyncAdapter
LOCAL_SRC_FILES := GoogleContactsSyncAdapter.apk
LOCAL_MODULE_CLASS := APPS
LOCAL_MODULE_TAGS := optional
LOCAL_CERTIFICATE := PRESIGNED
LOCAL_PRODUCT_MODULE := true
LOCAL_PRIVILEGED_MODULE := false

include $(BUILD_PREBUILT)
```



XML 配置更适合用 `prebuilt_etc` 安装：



```ABAP
prebuilt_etc {
    name: "google_default_permissions",
    product_specific: true,
    sub_dir: "default-permissions",
    src: "default-permissions.xml",
    filename_from_src: true,
}
```



产品配置引用模块时使用追加赋值，避免后一个 `PRODUCT_PACKAGES :=` 覆盖前一个列表：



```Makefile
PRODUCT_PACKAGES += \
    GoogleServicesFramework \
    PrebuiltGmsCore \
    Phonesky \
    SetupWizard \
    google_default_permissions

$(call inherit-product-if-exists, vendor/<company>/gms/gms.mk)
```



**\#\#\# 自定义 Setup Wizard 与 GMS**



Google Setup Wizard 可通过 partner customization 广播识别合作方包。自定义包可声明接收器：



```XML
<receiver
    android:name=".SetupWizardPartnerReceiver"
    android:exported="true">
    <intent-filter>
        <action android:name="com.android.setupwizard.action.PARTNER_CUSTOMIZATION" />
    </intent-filter>
</receiver>
```



是否需要该接口、广播权限、脚本格式和资源 URI 取决于集成的 Setup Wizard 版本及授权文档。不要假设空 `BroadcastReceiver` 就能完成适配；需要按交付版本定义提供正确的 partner 资源与流程。



若使用 Wizard Script，建议将每一步的 action、输入、结果码和下一个 action 显式描述，例如：



```XML
<WizardScript xmlns:wizard="http://schemas.android.com/apk/res/com.google.android.setupwizard"
    wizard:version="2">
    <WizardAction id="welcome"
        wizard:uri="intent:#Intent;action=com.example.setup.WELCOME;end">
        <result wizard:name="navigation"
            wizard:resultCode="111"
            wizard:action="setup_navigation" />
    </WizardAction>
    <WizardAction id="setup_navigation"
        wizard:uri="intent:#Intent;action=com.example.setup.NAVIGATION;end" />
    <WizardAction id="exit"
        wizard:uri="intent:#Intent;action=com.android.setupwizard.EXIT;end" />
</WizardScript>
```



完整链路应至少验证：恢复出厂设置后的启动入口、每一步的返回/跳过行为、导航模式切换、provisioning 状态写入、重启后的不再进入向导，以及已安装 GMS 应用的首次启动行为。



**\#\# 25\. 常用术语速查**



|术语|简要说明|
|---|---|
|AOSP|Android Open Source Project，Android 开源系统代码。|
|SoC|System on Chip，将 CPU、GPU、NPU、基带等集成在一颗芯片中。|
|HAL|Hardware Abstraction Layer，Framework 与硬件实现之间的稳定接口层。|
|OEM / ODM|OEM 负责品牌和产品定义；ODM 负责设计与制造，实际合作关系可重叠。|
|SystemServer|Zygote fork 出的核心 Java 系统进程，承载 AMS、PMS 等系统服务。|
|Bootloader|引导加载程序，负责加载/校验内核及启动链，常提供 Fastboot。|
|SELinux|Android 的强制访问控制机制，用于限制进程和文件等访问权限。|
|Treble|Android 8\.0 起的系统/厂商实现解耦架构，常涉及 system、vendor、product 等分区。|
|Binder|Android 最主要的本地跨进程通信机制。|
|AIDL|描述 Binder 跨进程接口的语言和构建工具。|
|JNI|Java Native Interface，Java/Kotlin 与 C/C\+\+ 交互接口。|
|HIDL|Android 8\.0 引入的 HAL 接口定义机制，已逐步由 AIDL HAL 替代。|
|RRO / SRO|运行时资源覆盖 / 编译时资源覆盖。|
|CTS / VTS|Android 兼容性测试套件 / Vendor Test Suite。|
|Mainline|通过 Google Play 系统更新机制独立更新部分系统组件。|



