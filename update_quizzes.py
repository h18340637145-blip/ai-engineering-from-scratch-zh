import json
import os

new_questions = {
    "phases/21-java-android-foundations/01-java-collections-and-equality/quiz.json": [
        {"stage": "check", "question": "HashMap 在什么情况下会将其内部结构从链表转换为红黑树？", "options": ["达到负载因子时", "链表长度超过8且数组容量大于等于64时", "一旦发生哈希冲突时", "永远不会发生转换"], "correct": 1, "explanation": "在 Java 8 中，当单条链表长度超过 8 并且整个 HashMap 的容量大于等于 64 时，链表会转换为红黑树以优化查找效率。"},
        {"stage": "check", "question": "ConcurrentHashMap 是如何保证线程安全的（Java 8+）？", "options": ["使用全局锁（synchronized方法）", "使用 CAS 和 synchronized 锁住链表或红黑树的首节点", "使用分段锁 (Segment)", "使用 ThreadLocal 隔离数据"], "correct": 1, "explanation": "Java 8 中的 ConcurrentHashMap 弃用了分段锁，改为使用 Node 数组 + CAS + synchronized 锁住桶的首节点来实现更细粒度的并发控制。"}
    ],
    "phases/21-java-android-foundations/02-java-oop-generics-and-strings/quiz.json": [
        {"stage": "check", "question": "String, StringBuffer 和 StringBuilder 中，哪个是线程安全的且可变的？", "options": ["String", "StringBuffer", "StringBuilder", "都不是"], "correct": 1, "explanation": "StringBuffer 是可变且线程安全的（方法由 synchronized 修饰）；StringBuilder 是可变但非线程安全的。"},
        {"stage": "check", "question": "关于 Java 泛型擦除，以下哪项描述是正确的？", "options": ["泛型信息在运行时仍然保留", "所有的泛型类型在编译后都会被替换为 Object 或其上界", "可以在运行时通过反射获取 List<String> 的具体类型信息", "基本数据类型可以直接作为泛型参数"], "correct": 1, "explanation": "Java 的泛型是在编译期实现的，编译后类型参数会被擦除，替换为其上界（如果没有指定上界则是 Object）。"}
    ],
    "phases/21-java-android-foundations/03-reflection-serialization-and-gc/quiz.json": [
        {"stage": "check", "question": "以下哪种垃圾回收算法是现代 JVM/ART 中常用于新生代的？", "options": ["标记-清除算法", "标记-压缩算法", "复制算法", "引用计数法"], "correct": 2, "explanation": "新生代的大部分对象生命周期较短，通常使用复制算法（Copying）来提高收集效率。"},
        {"stage": "check", "question": "在 Java 序列化时，如果不希望某个字段被序列化，应该使用什么关键字？", "options": ["volatile", "static", "transient", "final"], "correct": 2, "explanation": "transient 关键字用于修饰不希望被序列化的成员变量。"}
    ],
    "phases/21-java-android-foundations/04-process-thread-and-ipc/quiz.json": [
        {"stage": "check", "question": "Android 中以下哪种 IPC 机制传输数据的容量限制最严格？", "options": ["Socket", "ContentProvider", "Binder", "共享内存 (Ashmem)"], "correct": 2, "explanation": "普通的 Binder 事务缓冲区大小通常限制在 1MB 左右（甚至更小），不适合传输极大数据。"},
        {"stage": "check", "question": "Android 默认情况下，一个应用的四大组件运行在什么位置？", "options": ["同一个进程的主线程", "各自独立的进程", "同一个进程的不同子线程", "系统服务进程中"], "correct": 0, "explanation": "默认情况下，同一个应用的所有组件都运行在该应用进程的主线程（UI 线程）中。"}
    ],
    "phases/21-java-android-foundations/05-handler-looper-and-concurrency/quiz.json": [
        {"stage": "check", "question": "一个线程中最多可以有几个 Looper 对象？", "options": ["1个", "2个", "没有限制", "由系统内存决定"], "correct": 0, "explanation": "由于 Looper 是通过 ThreadLocal 存储的，一个线程只能绑定一个 Looper 对象。"},
        {"stage": "check", "question": "Handler 发送延迟消息（sendMessageDelayed）的实现原理是什么？", "options": ["启动一个子线程睡眠指定时间后发送", "在 MessageQueue 中按触发时间 (when) 排序，未到时间时底层阻塞等待", "使用系统 AlarmManager 定时唤醒", "不断轮询判断时间是否到达"], "correct": 1, "explanation": "消息入队时会按目标触发时间 (when) 进行排序，Looper 轮询时如果发现最近的消息时间未到，会通过 epoll 机制阻塞等待，不会消耗 CPU。"}
    ],
    "phases/21-java-android-foundations/06-view-rendering-and-touch/quiz.json": [
        {"stage": "check", "question": "View 事件分发机制中，如果 onTouchEvent 返回 true，代表什么？", "options": ["事件被丢弃", "事件已消耗，不再向上级抛出", "事件继续传递给父 View", "触发 onClick 事件"], "correct": 1, "explanation": "onTouchEvent 返回 true 表示当前 View 处理并消耗了该触摸事件，事件终止传递。"},
        {"stage": "check", "question": "在自定义 View 时，通常在哪个方法中决定 View 的最终大小？", "options": ["onDraw", "onLayout", "onMeasure", "onSizeChanged"], "correct": 2, "explanation": "onMeasure 方法用于测量 View 的大小，并需通过 setMeasuredDimension() 设置最终测量的宽和高。"}
    ],
    "phases/21-java-android-foundations/07-list-rendering-and-image-cache/quiz.json": [
        {"stage": "check", "question": "RecyclerView 相比 ListView 性能更好的主要原因之一是什么？", "options": ["自动开启多线程加载", "使用了四级缓存机制（如 Scrap, Cache, ViewCacheExtension, RecycledViewPool）", "完全在 Native 层渲染", "默认不加载图片"], "correct": 1, "explanation": "RecyclerView 提供了更灵活且多层次的缓存机制，能更高效地复用 ViewHolder。"},
        {"stage": "check", "question": "LruCache 内部主要基于什么数据结构实现的？", "options": ["HashMap", "ArrayList", "LinkedHashMap", "LinkedList"], "correct": 2, "explanation": "LruCache 底层依赖 LinkedHashMap 的访问顺序 (access-order) 特性来淘汰最近最少使用的对象。"}
    ],
    "phases/21-java-android-foundations/08-memory-oom-and-anr/quiz.json": [
        {"stage": "check", "question": "引起 Android 内存泄漏的常见原因不包括哪一项？", "options": ["非静态内部类隐式持有外部类引用", "使用 Application Context 代替 Activity Context", "未注销的广播接收器", "单例模式持有 Activity 的引用"], "correct": 1, "explanation": "使用 Application Context 通常是避免内存泄漏的好方法，因为它与应用同生命周期，不会导致 Activity 无法被回收。"},
        {"stage": "check", "question": "在 Android 中，主线程 (UI线程) 发生 ANR 的常见超时时间（Input dispatching timed out）是多少？", "options": ["1秒", "5秒", "10秒", "20秒"], "correct": 1, "explanation": "按键或触摸事件在主线程无响应的 ANR 触发时间通常为 5 秒。"}
    ],
    "phases/21-java-android-foundations/09-activity-window-and-service/quiz.json": [
        {"stage": "check", "question": "从 Activity A 启动 Activity B 时，生命周期的调用顺序是怎样的？", "options": ["A.onPause -> B.onCreate -> B.onStart -> B.onResume -> A.onStop", "A.onPause -> A.onStop -> B.onCreate -> B.onStart -> B.onResume", "B.onCreate -> B.onStart -> B.onResume -> A.onPause -> A.onStop", "A.onStop -> B.onCreate -> B.onStart -> B.onResume"], "correct": 0, "explanation": "当前 Activity 先执行 onPause，随后新 Activity 执行初始化和展示，最后再回调旧 Activity 的 onStop。"},
        {"stage": "check", "question": "如果想要在应用退出后 Service 依然能在后台长时间运行，通常需要怎么做？", "options": ["使用 bindService 启动", "使用 startService 并在 Service 中返回 START_STICKY，且可能需要提升为前台服务", "在 onCreate 中创建一个普通线程即可", "在 AndroidManifest 中设置 persistent=\"true\""], "correct": 1, "explanation": "普通服务在应用进程被杀后会停止。返回 START_STICKY 可让系统尝试重启服务；但在高版本 Android 中，持久后台运行通常需要提升为前台服务 (Foreground Service)。"}
    ],
    "phases/21-java-android-foundations/10-media-jni-network-and-security/quiz.json": [
        {"stage": "check", "question": "在 JNI 开发中，如果不手动释放局部引用 (Local Reference)，可能会导致什么后果？", "options": ["Native 崩溃", "局部引用表溢出 (JNI ERROR: local reference table overflow)", "Java 层的 OutOfMemoryError", "应用被强制卸载"], "correct": 1, "explanation": "JNI 的局部引用表大小是有限的，如果在循环中创建大量局部引用且不释放，会触发局部引用表溢出崩溃。"},
        {"stage": "check", "question": "Android 中用于安全存储密钥，且确保密钥难以被提取的系统组件是什么？", "options": ["SharedPreferences", "SQLite 数据库", "Android Keystore 系统", "外部存储 (SD卡)"], "correct": 2, "explanation": "Android Keystore 提供硬件支持的密钥存储，密钥材料不会进入应用进程内存，极大地提高了安全性。"}
    ],
    "phases/21-java-android-foundations/11-architecture-performance-and-storage/quiz.json": [
        {"stage": "check", "question": "在 MVVM 架构中，ViewModel 的主要职责是什么？", "options": ["直接操作 UI 控件 (如 TextView.setText)", "处理数据逻辑并通过 LiveData/StateFlow 暴露给 View", "负责与数据库进行最底层的 SQL 交互", "管理 Activity 的生命周期流转"], "correct": 1, "explanation": "ViewModel 用于存储和管理与 UI 相关的界面数据，并且在生命周期变化时保留数据，但不应持有任何 View 引用。"},
        {"stage": "check", "question": "以下哪种方式不适合用于持久化存储大型结构化数据？", "options": ["SQLite / Room", "SharedPreferences", "文件存储", "网络云端存储"], "correct": 1, "explanation": "SharedPreferences 会在内存中缓存整个 XML 文件，存储大型或复杂的结构化数据会导致性能问题和内存消耗过大。"}
    ],
    "phases/21-java-android-foundations/12-binder-classloading-build-and-install/quiz.json": [
        {"stage": "check", "question": "Android 中应用程序的类加载器主要是哪个？", "options": ["PathClassLoader", "DexClassLoader", "URLClassLoader", "BootClassLoader"], "correct": 0, "explanation": "Android 应用程序通常由 PathClassLoader 加载，它用于加载已安装的 APK/DEX 文件。DexClassLoader 常用于动态加载。"},
        {"stage": "check", "question": "APK 构建过程中，主要用于将 Java/Kotlin 字节码转换为 Dalvik 字节码的工具是什么？", "options": ["AAPT2", "R8 / D8", "ApkSigner", "Zipalign"], "correct": 1, "explanation": "D8 编译器（或带有混淆功能的 R8）负责将 .class 字节码转换为 .dex 字节码。"}
    ],
    "phases/21-java-android-foundations/13-linux-adb-and-device-operations/quiz.json": [
        {"stage": "check", "question": "使用 adb 命令如何清除某个应用的数据及缓存？", "options": ["adb uninstall <包名>", "adb shell pm clear <包名>", "adb shell rm -rf /data/app", "adb shell am force-stop <包名>"], "correct": 1, "explanation": "pm clear 命令用于重置应用，清除其在 /data/data/ 下的用户数据和缓存，等同于在设置中点击“清除数据”。"},
        {"stage": "check", "question": "Linux 中的 chmod 755 命令代表什么权限？", "options": ["所有者具有读写执行权限，其他人具有读和执行权限", "所有者仅有读权限", "所有人拥有全部权限", "禁止任何写入操作"], "correct": 0, "explanation": "7 (4+2+1) 代表所有者拥有读、写、执行权限；5 (4+1) 代表属组和其他人拥有读和执行权限。"}
    ],
    "phases/21-java-android-foundations/14-system-diagnostics-and-stability/quiz.json": [
        {"stage": "check", "question": "想要查看当前系统中运行了哪些 Activity，通常使用哪个 dumpsys 命令？", "options": ["dumpsys window", "dumpsys meminfo", "dumpsys activity activities", "dumpsys cpuinfo"], "correct": 2, "explanation": "dumpsys activity (或 activity activities) 用于输出 ActivityManager 的状态，包括任务栈和 Activity 信息。"},
        {"stage": "check", "question": "关于 Systrace/Perfetto，以下说法正确的是？", "options": ["它只能追踪 Java 层的异常", "它可以提供设备整个系统的 CPU、线程、I/O 等时序图", "使用它不需要设备开启开发者选项", "它主要用于检测内存泄漏"], "correct": 1, "explanation": "Systrace/Perfetto 收集内核级和框架级的跟踪数据，提供高度详细的性能时序视图。"}
    ],

    "phases/22-android-framework-system-basics/01-boot-chain-and-bootanimation/quiz.json": [
        {"stage": "check", "question": "Android 启动过程中，解析 init.rc 文件并启动关键系统服务的进程是哪个？", "options": ["Zygote", "init 进程", "SystemServer", "Kernel"], "correct": 1, "explanation": "init 是内核启动后进入用户空间的第一个进程，负责挂载文件系统和解析 init.rc，从而启动 Zygote 等服务。"},
        {"stage": "check", "question": "Bootanimation (开机动画) 是由哪个进程启动的？", "options": ["SystemServer", "SurfaceFlinger", "Zygote", "由 init 进程基于 init.rc 触发"], "correct": 3, "explanation": "bootanim 服务在 init.rc 中配置，通常由 SurfaceFlinger 发出信号或 init 直接根据条件触发启动。"}
    ],
    "phases/22-android-framework-system-basics/02-native-log-and-callstack/quiz.json": [
        {"stage": "check", "question": "当 Native 层发生崩溃 (如段错误) 时，系统会生成什么文件来记录崩溃信息？", "options": ["ANR traces", "Tombstone 文件", "Java Exception Log", "Bugreport"], "correct": 1, "explanation": "Native 崩溃由 debuggerd 捕获，并会在 /data/tombstones/ 目录下生成包含调用栈和寄存器信息的 tombstone 文件。"},
        {"stage": "check", "question": "Android 系统的日志守护进程是什么？", "options": ["logcat", "logd", "syslog", "kmsg"], "correct": 1, "explanation": "logd 是 Android 系统底层的日志守护进程，负责收集和存储不同缓冲区 (main, system, crash 等) 的日志。"}
    ],
    "phases/22-android-framework-system-basics/03-zygote-and-system-server/quiz.json": [
        {"stage": "check", "question": "为什么 Android 所有的应用进程都要通过 Zygote 来 fork？", "options": ["为了安全性检查", "为了共享 Zygote 已经预加载的通用类和资源，加快启动速度并节省内存", "因为 init 进程不能直接启动应用", "因为只有 Zygote 拥有 root 权限"], "correct": 1, "explanation": "Zygote 在启动时预加载了常用的类和资源，通过 fork( ) 创建应用进程可以使用 Copy-On-Write 机制，大幅优化了启动速度和内存开销。"},
        {"stage": "check", "question": "SystemServer 进程是由谁启动的？", "options": ["直接由 init 进程启动", "由 Zygote fork 而来", "由 ActivityManagerService 启动", "由 Bootloader 启动"], "correct": 1, "explanation": "SystemServer 是 Zygote 启动后主动 fork 出的第一个核心子进程，承载了绝大多数的 Android 系统服务。"}
    ],
    "phases/22-android-framework-system-basics/04-android-build-and-partition/quiz.json": [
        {"stage": "check", "question": "在 AOSP 构建系统中，目前主要使用哪种构建配置语言来替代旧的 Android.mk？", "options": ["CMake", "Gradle", "Android.bp (Soong)", "Bazel"], "correct": 2, "explanation": "Android 引入了 Soong 构建系统，使用基于 JSON 语法的 Android.bp 文件来替代之前基于 Make 的 Android.mk。"},
        {"stage": "check", "question": "芯片厂商 (SoC Vendor) 提供的硬件驱动和 HAL 实现通常位于哪个分区？", "options": ["system 分区", "vendor 分区", "data 分区", "boot 分区"], "correct": 1, "explanation": "Treble 架构引入后，系统核心逻辑与硬件相关代码分离，厂商特定的底层实现被放置在 vendor 分区中。"}
    ],
    "phases/22-android-framework-system-basics/05-resource-overlays/quiz.json": [
        {"stage": "check", "question": "RRO (Runtime Resource Overlay) 的主要作用是什么？", "options": ["在运行时动态修改系统或应用的资源值 (如颜色、字符串、布局)，而无需修改目标 APK", "在运行时动态替换目标应用的 Java 代码", "提升应用的渲染帧率", "用于修复系统内核漏洞"], "correct": 0, "explanation": "RRO 允许将覆盖包 (Overlay APK) 中的资源映射到目标应用中，是定制系统 UI 和主题的关键机制。"},
        {"stage": "check", "question": "当有多个 Overlay 覆盖同一个资源时，系统如何决定使用哪一个？", "options": ["随机选择", "根据 Overlay 的包名按字母排序", "根据 Overlay 的优先级 (priority) 以及 target 规则来决定解析顺序", "只使用第一个安装的 Overlay"], "correct": 2, "explanation": "Overlay 的生效顺序由其在系统中的优先级配置决定，高优先级的资源会覆盖低优先级的资源。"}
    ],
    "phases/22-android-framework-system-basics/06-system-properties-and-settings/quiz.json": [
        {"stage": "check", "question": "如果想要在系统重启后依然保留一个自定义属性 (property)，其名称应该以什么前缀开头？", "options": ["sys.", "ro.", "persist.", "debug."], "correct": 2, "explanation": "以 persist. 开头的系统属性会被保存在磁盘上，在重启后依然保持最后设定的值。"},
        {"stage": "check", "question": "Android 中的 Settings.Global, Settings.System, Settings.Secure 有何主要区别？", "options": ["只在存储格式上有区别", "权限和作用域不同：Global对所有用户生效，Secure包含敏感信息且第三方不可随便写，System包含普通系统偏好", "它们都只存储在内存中", "Global只能被系统服务读取"], "correct": 1, "explanation": "三者区分了设置项的可见性和读写权限。Global 是设备级别的，Secure 保护敏感数据，System 存储非敏感偏好。"}
    ],
    "phases/22-android-framework-system-basics/07-setup-wizard-and-provisioning/quiz.json": [
        {"stage": "check", "question": "标识设备是否已经完成首次开机引导 (Setup Wizard) 的全局设置项通常是哪一个？", "options": ["adb_enabled", "device_provisioned", "user_setup_complete", "install_non_market_apps"], "correct": 1, "explanation": "Settings.Global.DEVICE_PROVISIONED 用于标识整个设备是否已完成配置。通常还有 Settings.Secure.USER_SETUP_COMPLETE 标识当前用户。"},
        {"stage": "check", "question": "为什么在开机向导完成之前，按下 Home 键无效？", "options": ["Home 键物理屏蔽", "开机向导应用捕获并禁用了 Home 键响应，且系统在未 provisioned 状态下会限制 Home 键跳转", "桌面应用 (Launcher) 尚未被安装", "内核禁用了所有按键输入"], "correct": 1, "explanation": "在 provisioning 未完成时，系统级策略 (PhoneWindowManager 等) 通常会限制 Home 键跳转，以强制用户完成引导。"}
    ],
    "phases/22-android-framework-system-basics/08-gms-integration-and-customization/quiz.json": [
        {"stage": "check", "question": "如果一款 Android 设备想要预装 Google 官方应用商店及服务 (GMS)，必须通过什么认证？", "options": ["FCC 认证", "CTS、GTS 和 VTS 等一系列兼容性测试与认证", "Google Play Store 上架审核", "开源协议审核"], "correct": 1, "explanation": "Google 要求设备制造商通过兼容性测试套件 (CTS) 和 GMS 测试套件 (GTS) 等严格测试，才能合法预装 GMS。"},
        {"stage": "check", "question": "AOSP 原生系统与搭载 GMS 的系统在架构上的一个明显区别是什么？", "options": ["AOSP 无法运行 Java 应用", "GMS 替换了系统的核心网络定位、推送服务 (FCM) 等模块的实现提供者", "GMS 会完全替换掉系统的 SystemServer", "AOSP 没有浏览器"], "correct": 1, "explanation": "GMS 将许多标准服务（如 NLP 网络定位、应用推送）重定向到了 Google 闭源的 Play Services 包中。"}
    ],
    "phases/22-android-framework-system-basics/09-framework-integration-lab/quiz.json": [
        {"stage": "check", "question": "在修改 Android Framework 核心类 (如 Activity.java) 后，想要快速验证而不全编译系统，通常可以编译哪个模块？", "options": ["make kernel", "make framework", "make init", "make bootimage"], "correct": 1, "explanation": "通常可以单独执行 make framework (或 make services) 然后将生成的 framework.jar push 到设备中并重启。"},
        {"stage": "check", "question": "如果修改了 System API 或公开 API，导致构建失败提示 API check failed，应该怎么做？", "options": ["直接忽略错误", "修改系统源码去除 API 检查工具", "执行 make update-api 以更新 current.txt 等 API 记录文件", "将修改的 API 设为 private"], "correct": 2, "explanation": "构建系统会对比当前 API 与预定义的 API 列表。合法的 API 变更需要执行 make update-api (或 m update-api) 提交记录变更。"}
    ]
}

for path, additional_qs in new_questions.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    
    # insert before the first 'post' stage question, or at the end
    insert_idx = len(questions)
    for i, q in enumerate(questions):
        if q.get("stage") == "post":
            insert_idx = i
            break
            
    for q in additional_qs:
        questions.insert(insert_idx, q)
        insert_idx += 1
        
    data["questions"] = questions
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("Quizzes updated successfully.")

