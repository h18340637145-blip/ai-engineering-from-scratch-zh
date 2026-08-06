# Java android 

# **Java 基础知识整理**



> 面向复习和面试的速查笔记：先理解概念，再记住适用场景。
> 
> 



## **1\. 链表**



链表由一系列节点组成。每个节点保存数据，以及与其他节点的连接关系。节点在内存中不要求连续存放。



### **单向链表**



每个节点包含：数据 \+ 指向下一个节点的引用。



```Plain Text
head -> [data | next] -> [data | next] -> [data | null]
```



- 只能从前向后遍历。

- 已知前驱节点时，插入或删除节点通常是 $O(1)$。

- 按下标查找必须从头遍历，时间复杂度为 $O(n)$。

### **双向链表**



每个节点包含：数据 \+ 前驱引用 \+ 后继引用。



```Plain Text
null <- [prev | data | next] <-> [prev | data | next] -> null
```



- 可以向前或向后遍历。

- 已知节点时，删除更方便，因为能直接找到前驱和后继。

- 相比单向链表，多占用一个引用的内存。

- Java 的 `LinkedList` 是双向链表实现。

### **循环链表**



首尾相连的链表称为循环链表，可由单向或双向链表实现。



```Plain Text
[node] -> [node] -> [node]
  ^                     |
  |_____________________|
```



- 没有天然的 `null` 结尾，遍历时要以“回到起始节点”作为结束条件。

- 常用于轮询、约瑟夫环等场景。

- 带虚拟头节点（哨兵节点）的循环链表可减少首尾插入、删除时的特殊判断。

## **2\. 数组、List、Set 和 Map**



### **数组与集合**



|特性|数组|集合|
|---|---|---|
|长度|创建后固定|通常可动态增长|
|元素类型|基本类型或对象|存放对象；基本类型会自动装箱|
|访问方式|下标|API、迭代器或键|
|典型用途|大小确定、追求访问性能|大小不确定、需要丰富操作|



### **List、Set、Map 的关系**



- `List` 和 `Set` 继承自 `Collection`，存储单列元素。

- `Map` 存储键值对，不继承 `Collection`。

- `List` 有序，允许重复元素。

- `Set` 不允许重复元素；是否保持顺序取决于实现类。

- `Map` 的键不能重复，值可以重复。`HashMap` 允许一个 `null` 键和多个 `null` 值；并非所有 `Map` 都允许 `null`。

### **常见 List 实现**



|实现类|底层结构|特点|适用场景|
|---|---|---|---|
|`ArrayList`|动态数组|随机访问快；中间插入、删除通常需要移动元素|读多写少|
|`LinkedList`|双向链表|已定位节点附近的插入、删除快；随机访问慢|频繁在两端或迭代位置增删|
|`Vector`|动态数组|方法大多同步，通常性能低于 `ArrayList`|兼容旧代码；新代码一般不选它|



### **常见 Map 实现**



|实现类|特点|
|---|---|
|`HashMap`|哈希表实现，非线程安全，通常无序，允许 `null`|
|`Hashtable`|旧的同步实现，不允许 `null` 键和值；新代码通常优先考虑 `ConcurrentHashMap`|
|`LinkedHashMap`|在 `HashMap` 基础上维护链表，通常按插入顺序迭代；也可配置为访问顺序|
|`TreeMap`|红黑树实现，按键的自然顺序或指定比较器排序；一般不允许 `null` 键|



### **常见 Set 实现**



|实现类|特点|
|---|---|
|`HashSet`|基于 `HashMap`，无重复元素，通常无序|
|`LinkedHashSet`|基于 `LinkedHashMap`，保持插入顺序|
|`TreeSet`|基于 `TreeMap`，元素有序，元素必须可比较或提供比较器|



**\#\#\# \`HashSet\` 去重原理**



`HashSet` 先通过 `hashCode()` 确定候选位置，再通过 `equals()` 判断元素是否相等：



1. `hashCode()` 不同，两个对象一定不相等。

2. `hashCode()` 相同，还需要调用 `equals()` 继续判断。

3. `equals()` 为 `true` 的两个对象，`hashCode()` 必须相同。

因此，自定义对象作为 `HashSet` 元素或 `HashMap` 键时，应同时重写 `equals()` 和 `hashCode()`。



> `hashCode()` 不是内存地址；它只是一个用于散列定位的整数值。
> 
> 



## **3\.\`==\`、\`equals\(\)\` 和 \`hashCode\(\)\`**



|比较方式|基本类型|引用类型|
|---|---|---|
|`==`|比较数值|比较是否引用同一个对象|
|`equals()`|不适用|比较对象逻辑是否相等，默认实现等同于引用比较|



示例：



```Java
String first = new String("java");
String second = new String("java");

first == second;        // false：不是同一个对象
first.equals(second);   // true：字符串内容相同
```



`Integer` 会缓存 `-128` 到 `127` 范围内的对象，因此该范围内由自动装箱产生的对象可能满足 `==`。业务代码应使用 `equals()` 比较包装类型的数值。



## **4\. 基本数据类型和字符编码**



### **基本类型大小**



|字节数|类型|
|---|---|
|1|`byte`、`boolean`（JVM 规范未精确定义其内存布局）|
|2|`short`、`char`|
|4|`int`、`float`|
|8|`long`、`double`|



### **字符与编码**



- Java 的 `char` 是 16 位 UTF\-16 代码单元，不一定代表一个完整的 Unicode 字符。

- UTF\-8 中，常见中文通常占 3 个字节，扩展字符可能占 4 个字节。

- GBK 中，常见中文通常占 2 个字节。

- 字节数取决于编码，不能只根据“是不是中文”判断。

## **5\. 面向对象三大特性**



\- **封装**：隐藏内部实现，通过公开的方法或接口暴露必要能力，降低耦合并保护状态。

\- **继承**：子类复用父类的可继承成员，并可扩展或重写行为。

\- **多态**：使用父类或接口引用指向不同子类对象；调用同一方法时表现出不同实现。



```Java
List<String> names = new ArrayList<>();
// 变量类型是接口，运行时对象可以替换为其他 List 实现。
```



## **6\.\`String\`、\`StringBuilder\` 和 \`StringBuffer\`**



|类型|可变性|线程安全|典型使用场景|
|---|---|---|---|
|`String`|不可变|是|少量字符串处理、常量、键|
|`StringBuilder`|可变|否|单线程中的大量拼接|
|`StringBuffer`|可变|是（同步）|需要共享同一缓冲区的旧式多线程代码|



字符串频繁拼接时，优先使用 `StringBuilder`，避免反复创建新的 `String` 对象。



## **7\. 内部类**



内部类是定义在另一个类内部的类。外部类可以是 `public`；成员内部类也可以使用访问修饰符。一个源文件中最多只有一个 `public` 顶级类，且文件名需与该类名一致。



|类型|特点|
|---|---|
|非静态成员内部类|依赖外部类实例，可访问外部类所有成员（含私有成员）|
|静态内部类|不依赖外部类实例，只能直接访问外部类静态成员|
|局部内部类|定义在方法或代码块内，仅在该作用域可见；可捕获有效 final 的局部变量|
|匿名内部类|没有类名，适合一次性的接口或类实现；现代 Java 中常可用 Lambda 替代|



## **8\. 抽象类与接口**



|特性|抽象类|接口|
|---|---|---|
|构造方法|可以有|没有|
|实例字段|可以有|字段默认是 `public static final` 常量|
|方法|可包含抽象和具体方法|可包含抽象方法、`default` 方法、`static` 方法和私有辅助方法|
|继承关系|类只能继承一个类|类可以实现多个接口|
|适用场景|有共用状态或部分实现的类层次|定义能力和契约，支持多实现|



## **9\. 泛型：\`extends\` 与 \`super\`**



\- \`? extends T\`：类型为 \`T\` 或其子类，适合**读取** \`T\`。除 \`null\` 外，不能安全地写入具体元素。

\- \`? super T\`：类型为 \`T\` 或其父类，适合**写入** \`T\`。



记忆口诀：**PECS**，Producer Extends，Consumer Super。



```Java
List<? extends Number> producers = List.of(1, 2, 3);
Number value = producers.get(0);

List<? super Integer> consumers = new ArrayList<Number>();
consumers.add(1);
```



## **10\. 反射**



反射是指程序在运行时获取类的信息，并访问其构造器、字段和方法的机制。



常见入口：



```Java
Class<?> type = Class.forName("com.example.User");
Constructor<?> constructor = type.getDeclaredConstructor();
Method method = type.getDeclaredMethod("getName");
```



- 优点：灵活，框架可在运行时按约定加载和调用类型。

- 缺点：类型检查推迟到运行时，通常性能更低，也可能破坏封装。

- 建议：优先使用正常的接口和方法调用；仅在框架、注解处理、插件加载等确有需要时使用反射。

## **11\. 序列化与反序列化**



\- **序列化**：将对象状态转换成可传输或存储的字节序列。

\- **反序列化**：从字节序列恢复对象状态。



Java 原生序列化通过实现 `Serializable` 使用，但新项目通常优先选择 JSON、Protocol Buffers 等明确的数据格式。不要反序列化不可信来源的数据，以免引入安全风险。



## **12\.\`String\` 转 \`Integer\`**



```Java
int number = Integer.parseInt("123");
Integer boxed = Integer.valueOf("123");
```



- `Integer.parseInt()` 返回基本类型 `int`。

- `Integer.valueOf()` 返回包装类型 `Integer`，可能复用缓存对象。

- 默认按十进制解析，也可以指定进制：`Integer.parseInt("ff", 16)`。

- 传入非数字或超出 `int` 范围的字符串会抛出 `NumberFormatException`。

## **13\. 垃圾回收（GC）**



Java GC 的核心判断是：对象是否仍能从 **GC Roots** 到达，而不是简单依据引用计数。



### **常见 GC Roots**



- 线程栈中的局部变量和方法参数。

- 已加载类的静态字段。

- JNI 引用。

- 正在运行的线程及 JVM 内部引用。

从 GC Roots 不可达的对象，才有资格被回收。循环引用本身不会阻止 Java 的可达性分析回收对象。



### **四种引用强度**



|引用类型|回收时机|
|---|---|
|强引用|只要仍可达，就不会被回收|
|软引用|内存不足时可能被回收，适合内存敏感缓存|
|弱引用|下一次 GC 时通常会被回收|
|虚引用|不影响对象生命周期，用于接收回收通知，必须配合 `ReferenceQueue`|



> `finalize()` 已被废弃且不可靠，不应依赖它释放资源。文件、网络连接等资源应使用 `try-with-resources` 主动关闭。
> 
> 





# **Android 基础知识整理**



> 面向复习与面试的速查笔记。Android API 和系统实现会随版本演进，涉及行为差异时应以目标 API 文档和实际设备测试为准。
> 
> 



## **1\. 进程、线程与 IPC**



### **进程和线程**



\- **进程**是资源分配和隔离的单位；一个 Android 应用默认运行在独立 Linux 进程中，但可通过 \`android:process\` 使用多个进程。

\- **线程**是 CPU 调度的单位；一个进程可包含多个线程。

- 主线程负责组件生命周期回调和 UI 操作。耗时 I/O、编解码和复杂计算应移到后台线程，结果再切回主线程更新 UI。

- 线程池能复用线程、限制并发量并统一管理任务，但线程数量仍应按任务类型和设备能力配置。

常用线程方法：



|方法|含义|
|---|---|
|`start()`|创建并启动新线程，随后由 JVM 调用该线程的 `run()`|
|`run()`|普通方法调用；直接调用不会创建新线程|
|`join()`|当前线程等待目标线程结束|
|`sleep()`|当前线程暂停一段时间，不释放已持有的锁|
|`setPriority()`|提供调度提示，不能保证实际执行顺序|



### **常见跨进程通信方式**



|方式|适用场景|要点|
|---|---|---|
|`Intent`|启动组件、传递少量参数|通过显式组件或 `action` 解析；`extras` 不适合大数据|
|Broadcast|一对多事件通知|动态注册与进程生命周期相关；静态注册由系统在匹配广播时拉起组件，受后台广播限制影响|
|`ContentProvider`|跨应用结构化数据共享|通过 `ContentResolver` 和 URI 访问；可控制读写权限|
|Binder / AIDL|客户端与服务端的高效 RPC|适合跨进程方法调用；AIDL 接口应避免大对象与长耗时同步调用|
|`Messenger`|基于 Binder 的串行消息通信|使用 `Handler` 收发 `Message`，实现较简单但服务端默认串行处理|
|Socket|网络或本地套接字通信|适合自定义协议、跨设备或跨语言场景|



> Binder 是 Android 主要 IPC 机制；AIDL 是描述 Binder 接口的一种工具，并不是 Binder 的替代品。
> 
> 



## **2\.\`wait\(\)\` 与 \`sleep\(\)\`**



|对比项|`Object.wait()`|`Thread.sleep()`|
|---|---|---|
|所属类|`Object`|`Thread`|
|锁要求|必须在持有该对象监视器的同步块/方法内调用|不要求 `synchronized`|
|是否释放监视器锁|释放，并进入该对象的等待集|不释放|
|恢复条件|被 `notify()` / `notifyAll()` 唤醒、超时或中断后重新竞争锁|到达睡眠时间或被中断|



`wait()` 应总是放在条件循环中，防止虚假唤醒：



```Java
synchronized (lock) {
    while (!ready) {
        lock.wait();
    }
}
```



## **3\. 字符串类型**



|类型|可变性|线程安全|使用建议|
|---|---|---|---|
|`String`|不可变|是|少量操作、常量、键|
|`StringBuilder`|可变|否|单线程频繁拼接，优先选择|
|`StringBuffer`|可变|是（同步）|共享同一缓冲区的并发场景，现代代码较少使用|



## **4\. View、SurfaceView 与绘制**



**\#\#\# \`View\` 和 \`SurfaceView\`**



- 普通 `View` 在应用窗口的 View 层级中绘制，UI 更新必须在主线程进行。

- `SurfaceView` 拥有独立的 `Surface`，可由后台线程通过 `SurfaceHolder` 锁定 Canvas 后绘制，适合视频、相机预览和高频渲染。

- `SurfaceView` 的 Surface 与普通 View 分层合成，透明、动画、叠放等行为有额外限制。

- `TextureView` 也能显示 `SurfaceTexture` 内容，但属于普通 View 层级，支持变换和动画，代价通常更高。

- `GLSurfaceView` 封装 OpenGL ES 渲染线程和 `Surface` 生命周期，适合 3D 图形。

### **View 绘制流程**



一次布局绘制主要包含：



1\. **measure**：父容器向下传递 \`MeasureSpec\`，子 View 测量期望宽高。

2\. **layout**：父容器确定子 View 的位置；\`ViewGroup\` 在 \`onLayout\(\)\` 中摆放子元素。

3\. **draw**：绘制背景、内容、子 View、前景等；自定义 View 通常重写 \`onDraw\(\)\`。



`requestLayout()` 会请求重新测量和布局，`invalidate()` 会请求重绘。两者不应在高频循环中滥用。



### **触摸事件分发**



触摸序列通常从父容器向子 View 分发：



```Plain Text
Activity -> Window -> DecorView -> 父 ViewGroup -> 子 View
```



|方法|作用|
|---|---|
|`dispatchTouchEvent()`|分发事件，并决定事件是否已被消费|
|`onInterceptTouchEvent()`|`ViewGroup` 决定是否拦截后续事件；普通 `View` 没有此方法|
|`onTouchEvent()`|处理事件，返回 `true` 表示消费|



- 同一手势序列从 `ACTION_DOWN` 开始；一个对象若不消费 `DOWN`，通常不会再收到后续事件。

- 父容器拦截后，子 View 会收到 `ACTION_CANCEL`。

- 子 View 可用 `requestDisallowInterceptTouchEvent()` 请求父容器暂不拦截，但父容器仍可在必要时拦截。

### **滑动冲突**



\- **方向不同**：根据移动距离判断水平或垂直意图，让相应容器拦截，例如横向分页与纵向列表。

\- **方向相同**：根据子容器是否还能向该方向滚动决定由谁处理，例如使用 \`canScrollVertically\(\)\`。

- 优先使用嵌套滚动体系（`NestedScrollingChild` / `NestedScrollingParent`、`CoordinatorLayout`），避免手写分发逻辑失控。

## **5\.\`RecyclerView\` 与 \`ListView\`**



|对比项|`ListView`|`RecyclerView`|
|---|---|---|
|布局能力|主要是垂直列表|通过 `LayoutManager` 支持线性、网格、瀑布流等|
|ViewHolder|推荐手动实现|`ViewHolder` 是适配器模型的核心|
|动画|支持有限|内建 `ItemAnimator`，支持插入、移除、移动动画|
|分割线与点击|常用内置能力|通常通过 `ItemDecoration` 和监听器实现|
|局部更新|常见为 `notifyDataSetChanged()`|支持 `notifyItem...()`；推荐 `ListAdapter` \+ `DiffUtil`|



`RecyclerView` 嵌套卡顿的常用处理：



1. 避免不必要的嵌套，优先扁平化数据和布局。

2. 内层横向列表使用共享 `RecycledViewPool`，并设置合理的预取数量。

3. 在尺寸固定时调用 `setHasFixedSize(true)`。

4. 只有在需求允许时再调用 `setNestedScrollingEnabled(false)`，不能将其视为通用性能开关。

5. 使用稳定 ID、`DiffUtil`，避免整表刷新和复杂 `onBindViewHolder()`。

## **6\. Handler、Looper 与 MessageQueue**



- `Handler` 负责向消息队列投递消息或任务，并在所属 `Looper` 线程执行回调。

- `MessageQueue` 按时间顺序保存消息。

- `Looper.loop()` 持续取出消息并分发给目标 `Handler`。

- 主线程在框架启动时已创建 `Looper` 和 `MessageQueue`；后台线程只有在调用 `Looper.prepare()` 后才拥有它们。

```Plain Text
Handler.post/sendMessage -> MessageQueue -> Looper.loop -> Handler.handleMessage
```



不要让 `Handler`、延迟消息或匿名 Runnable 长期持有 `Activity` / `View`。页面销毁时，应移除不再需要的消息和回调。



## **7\. 内存泄漏、内存溢出与 ANR**



### **内存泄漏与 OOM**



\- **内存泄漏**：不再需要的对象仍被强引用，无法被 GC 回收。

\- **OOM（OutOfMemoryError）**：进程申请内存失败；持续泄漏是常见诱因，但大对象、缓存失控也可直接导致 OOM。



常见泄漏来源及治理方式：



|来源|处理方式|
|---|---|
|单例、静态对象持有 `Activity`|使用 `Application` Context；必要时改为弱引用或显式清理|
|非静态内部类、匿名类、Handler|用静态类 \+ `WeakReference`（仅在语义正确时）；移除消息与回调|
|注册未反注册|在对应生命周期取消广播、观察者、监听器和回调|
|协程、线程、网络请求未取消|在界面销毁时取消任务，避免回调引用已销毁页面|
|动画未停止|在 View 或页面销毁时取消动画|
|`Cursor`、流、数据库等资源|使用 `use` / `try-with-resources` 或对应的 `close()`|
|WebView|停止加载、清理回调并销毁；避免长期持有页面 Context|



推荐使用 LeakCanary 定位泄漏链路。不要为了“释放 Bitmap”而随意把仍在使用的 `ImageView` 内容设为 `null`；应让图片加载库、生命周期和缓存策略共同管理资源。



### **ANR**



ANR 是应用主线程长时间无法响应输入事件或组件回调时触发的系统提示。常见场景包括输入事件超时、前台广播执行过久、Service 执行超时等，具体阈值会因 Android 版本和组件类型而不同。



避免方式：



- 主线程不执行磁盘、网络、数据库、大量 JSON 解析或复杂计算。

- `BroadcastReceiver.onReceive()` 快速返回；耗时工作转交给 `WorkManager`、前台服务或其他合适组件。

- 避免锁竞争、死锁和在主线程等待后台任务。

- 用 Perfetto / System Trace、`dumpsys`、ANR traces 和 StrictMode 定位阻塞点。

## **8\. Activity、Fragment、Window 与 Service**



### **Activity 生命周期**



常见前台路径：



```Plain Text
onCreate -> onStart -> onResume
```



退到后台通常为：



```Plain Text
onPause -> onStop
```



回到前台通常为：



```Plain Text
onRestart -> onStart -> onResume
```



销毁时通常为：



```Plain Text
onPause -> onStop -> onDestroy
```



系统可能直接终止已停止的进程而不调用 `onDestroy()`，因此关键状态应在合适时机持久化，并通过 `ViewModel`、`SavedStateHandle` 或 `onSaveInstanceState()` 恢复。



### **Fragment**



Fragment 的生命周期与宿主 Activity 和自身 View 生命周期相关。常见回调：



```Plain Text
onAttach -> onCreate -> onCreateView -> onViewCreated -> onStart -> onResume
```



View 被销毁时会先调用 `onDestroyView()`，Fragment 实例未必立即销毁。涉及 View 的观察者应绑定 `viewLifecycleOwner`，避免 Fragment 持有已销毁 View。



- `add()`：保留已有 Fragment；通常配合 `show()` / `hide()` 管理显示。

- `replace()`：移除容器中当前 Fragment，再添加新 Fragment；是否保留在返回栈取决于事务是否 `addToBackStack()`。

- 旧版 `FragmentPagerAdapter`、`FragmentStatePagerAdapter` 已废弃；新代码使用 `ViewPager2` \+ `FragmentStateAdapter`。

### **Activity、Window 与 View**



1. Activity 在 `attach()` 阶段关联 `PhoneWindow`。

2. `PhoneWindow` 创建并持有顶层 `DecorView`。

3. `setContentView()` 将布局加载到内容区域，形成 View 树。

4. `WindowManager` / WMS 负责窗口管理；输入事件经 InputDispatcher、窗口与 ViewRootImpl 分发到 View 树，并非由 WMS 直接回调每个 View 的监听器。

### **屏幕旋转**



- 默认情况下，配置变化通常会重建 Activity；应通过 ViewModel、保存状态和资源限定符处理。

- 声明 `android:configChanges` 后，系统将部分配置变化交由 Activity 的 `onConfigurationChanged()` 处理；实际行为与所声明的配置项和目标系统有关。

- 不建议为逃避重建而滥用 `configChanges`，因为需要自行更新资源、布局和状态。

### **Activity 启动模式**



|模式|行为|
|---|---|
|`standard`|每次启动都创建新实例|
|`singleTop`|若目标实例位于栈顶则复用，并回调 `onNewIntent()`；否则新建|
|`singleTask`|任务栈内复用已有目标实例，并清除其上方 Activity；受 task 与文档模式等影响|
|`singleInstance`|实例独占一个任务；现代任务管理下应谨慎使用|



### **Service**



- `startService()` / `startForegroundService()`：服务可独立于绑定者运行，须通过 `stopSelf()` 或 `stopService()` 停止。Android 8\.0\+ 对后台启动服务有限制，长期任务通常需使用前台服务或 `WorkManager`。

- `bindService()`：客户端绑定期间服务存活；客户端通过 `IBinder` 获取接口或使用 AIDL 通信。

- Service 不应强制把 `Context` 转成 Activity 来回调 UI。应使用 Binder 回调、广播、LiveData / Flow 或其他解耦机制，并确保生命周期安全。

## **9\. 多媒体与 JNI**



### **相机与音视频**



\- 新项目优先使用 **CameraX** 或 Camera2；旧 \`Camera\` API 已废弃。

- 典型相机流程：绑定预览输出、配置相机、采集帧、编码或保存。预览目标可使用 `SurfaceView`、`TextureView` 或 `SurfaceTexture`。

- 常见原始视频格式为 YUV（如 `YUV_420_888`、NV21），原始音频格式常为 PCM；具体格式由 API 和设备决定。

- 视频常由 `MediaCodec` 编码为 H\.264 / H\.265 等，录制封装可使用 `MediaMuxer` 或 `MediaRecorder`；音视频同步和时间戳管理不可省略。

### **JNI**



JNI（Java Native Interface）让 Java / Kotlin 代码调用 C / C\+\+，也允许原生代码回调 JVM。



Android 常见流程：



1. 编写 C / C\+\+ 实现及 Java/Kotlin 的 `native` 声明。

2. 使用 CMake 或 `ndk-build` 构建 ABI 对应的 `.so` 库。

3. 通过 `System.loadLibrary()` 加载库。

4. 由 JNI 桥接调用原生函数。

JNI 适合复用原生库、性能敏感编解码等场景，但会增加内存管理、线程附着、异常处理和 ABI 兼容性复杂度。



## **10\. 网络与安全**



### **TCP、UDP、Socket、HTTP 与 Web Service**



|概念|说明|
|---|---|
|TCP|面向连接、可靠、有序的字节流传输；建立连接通常使用三次握手，关闭连接涉及 FIN / ACK 交互|
|UDP|无连接的数据报协议，不保证到达、顺序或不重复，开销低、延迟小|
|Socket|网络编程接口，不等同于“长连接”；TCP Socket 可保持长连接，也可短连接|
|HTTP|应用层请求\-响应协议；可运行在 TCP 或 QUIC 上，HTTP/2 支持多路复用，HTTP/3 基于 QUIC|
|Web Service / SOAP|SOAP 是基于 XML 的消息协议，可运行于 HTTP 等传输层；Web Service 不只等于 SOAP|



### **网络库**



\- **OkHttp**：HTTP 客户端，提供连接池、拦截器、缓存、HTTP/2 和 WebSocket 等能力；它不基于 \`HttpURLConnection\` 封装，而是独立实现。

\- **Retrofit**：基于注解的接口声明框架，通常以 OkHttp 作为底层客户端；负责将接口调用转换为 HTTP 请求和数据转换。

\- **Volley**：适合小型、频繁的网络请求，提供请求队列和图片加载能力；不适合大文件传输或复杂流式场景。

\- **xUtils**：历史上的综合框架；新项目应谨慎评估维护状态、体积和依赖风险。



### **哈希与加密**



|算法|类型|建议|
|---|---|---|
|MD5、SHA\-1|哈希，单向|已不适合安全用途；校验兼容场景除外|
|SHA\-256 / SHA\-3|哈希，单向|用于完整性校验；密码存储应使用专用 KDF，如 bcrypt、scrypt、Argon2|
|AES|对称加密|适合大量数据加密；密钥必须安全生成、保存和协商|
|RSA / ECC|非对称加密|常用于签名或安全协商密钥；RSA 不适合直接加密大数据|



现实系统常用混合加密：通过非对称机制协商或加密临时对称密钥，再用 AES 等算法加密业务数据。



## **11\. 架构与性能优化**



### **MVC、MVP、MVVM**



|架构|核心思想|Android 实践|
|---|---|---|
|MVC|Model、View、Controller 分离|Activity / Fragment 往往同时承担 View 和 Controller，容易膨胀|
|MVP|Presenter 协调 View 与 Model|View 保持被动，Presenter 更易测试，但接口可能较多|
|MVVM|View 观察 ViewModel 暴露的状态|常与 LiveData / Flow、Data Binding 或 Compose 状态配合；ViewModel 不应持有 View 引用|



### **性能优化方向**



\- **布局与渲染**：减少嵌套和过度绘制；按需使用 \`ViewStub\`、\`\<include\>\`、\`\<merge\>\`；避免无意义的 \`wrap\_content\` 链式测量。

\- **启动**：区分首帧必要工作与延后工作；使用 App Startup、异步初始化和基线配置文件，避免主线程阻塞。

\- **内存**：控制缓存大小，及时取消任务、释放资源，避免 Context 和 View 泄漏。

\- **包体**：删除无用资源和依赖，启用 R8，使用资源压缩与适当的图片格式，拆分 ABI / 功能模块。

\- **分析工具**：使用 Android Studio Profiler、Perfetto / System Trace、Macrobenchmark、Layout Inspector 和 \`dumpsys\`，先测量再优化。



## **12\. SharedPreferences 跨进程**



`SharedPreferences` 不保证多进程读写一致性。不同进程各自缓存数据，可能读到旧值或发生竞争写入；已废弃的 `MODE_MULTI_PROCESS` 不能作为可靠方案。



跨进程共享数据应使用：



- `ContentProvider` \+ 数据库或文件。

- Binder / AIDL 服务。

- 由单一进程负责写入，其他进程通过 IPC 读取。

`sharedUserId` 已被废弃，不能用它解决共享偏好或跨应用数据访问问题。



## **13\. Binder、类加载、构建与安装**



### **Binder**



Binder 是 Android 的核心 IPC 驱动与框架。客户端通过代理将请求写入 Binder 驱动，服务端线程池读取并执行，再返回结果。Binder 使用内核驱动和内存映射减少数据复制次数，但仍有事务大小限制，不应传输大位图或大集合。



### **Android 类加载器**



Android 运行 DEX / ODEX 等字节码格式。常见类加载器包括：



- `PathClassLoader`：加载应用安装包和系统路径中的代码。

- `DexClassLoader`：可从指定路径加载 DEX / APK / JAR，常用于插件化等受控场景。

动态加载会带来安全、兼容和调试成本，应只加载可信代码。



### **APK / AAB 构建概览**



现代 Android Gradle Plugin 的构建流程大致为：



1. 合并 Manifest、资源和依赖。

2. AAPT2 编译和链接资源，生成资源表及 `R` 类相关代码。

3. 编译 Java / Kotlin 源码为 `.class`。

4. 处理 AIDL、注解处理和字节码转换。

5. D8 将字节码转换为 DEX；R8 可进行压缩、优化和混淆。

6. 打包为 APK 或 AAB，并使用签名密钥签名；发布包通常还需 `zipalign`。

`dx`、`apkbuilder` 是旧工具，现代构建通常使用 D8、R8、Bundletool 等工具链。



### **安装概览**



Package Manager 解析并校验安装包、安装代码和资源、记录组件与权限信息，并为应用准备数据目录。ART 会在需要时进行 dex 优化或编译；具体路径和优化策略依设备、Android 版本及安装方式而变。安装完成后，系统会更新包管理状态并发送相关系统广播。



## **14\. Linux 与 ADB 常用命令**



### **Linux**



```Bash
pwd                         # 当前目录
cd /path/to/dir             # 切换目录
cd ..                        # 上级目录
cd -                         # 上次目录
ls -la                       # 查看文件（含隐藏文件）
mkdir -p path/to/dir         # 创建目录
cp -r source destination     # 复制目录
mv source destination        # 移动或重命名
find /path -name 'pattern'   # 查找文件
grep -R "text" /path         # 搜索文本
```



`rm -rf` 极具破坏性，执行前必须确认绝对路径和通配符展开结果，尤其不要使用 `rm -rf *` 处理不确定目录。



### **ADB**



```Bash
adb devices
adb start-server
adb kill-server
adb connect HOST:PORT
adb disconnect HOST:PORT
adb install app.apk
adb install -r app.apk
adb uninstall com.example.app
adb shell pm list packages
adb shell am start -n com.example.app/.MainActivity
adb logcat
adb logcat --pid=$(adb shell pidof -s com.example.app)
adb shell dumpsys activity activities
adb shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'
```



## **15\. 图片缓存与性能分析**



### **图片缓存**



典型策略是：内存缓存 \-\> 磁盘缓存 \-\> 网络加载。内存最快，磁盘次之，网络最慢。



- 优先使用成熟图片库（Coil、Glide、Picasso 等），由其处理采样、缓存、取消、生命周期与复用。

- 不建议手工维护“软引用二级缓存”作为主要策略：软引用回收时机不可控，可能造成缓存抖动。

- 大图应按目标尺寸解码，列表中应避免原图解码和频繁创建 Bitmap。

### **性能分析**



- 使用 Perfetto / System Trace 分析线程调度、帧时间、Binder 调用和 I/O。

- 使用 Android Studio CPU Profiler 或采样工具定位热点；`Debug.startMethodTracing()` 可生成方法追踪，但开销较大，不适合代表性线上性能测量。

- 使用 Memory Profiler 和 LeakCanary 排查内存；使用 Macrobenchmark 测量启动和滚动等用户路径。



## **14\. Linux、ADB 与 Git 常用命令**



本节中的 `<...>` 表示需要替换为实际值，例如 `<serial>` 是设备序列号，

`<package>` 是应用包名。多设备连接时，请在 `adb` 后添加

`-s <serial>`，例如：`adb -s <serial> shell`。



### **Linux 文件与搜索**



```Bash
pwd                              # 查看当前目录
cd /path/to/dir                  # 切换目录
cd ..                            # 返回上级目录
cd -                             # 返回上一次目录
ls -la                           # 查看文件（含隐藏文件）
mkdir -p path/to/dir             # 创建多级目录
cp -r source destination         # 复制目录
mv source destination            # 移动或重命名
find /path -name 'pattern'       # 按名称查找文件
grep -Rni "text" /path           # 递归搜索文本，忽略大小写并显示行号
```



`rm -rf` 极具破坏性，执行前必须确认绝对路径和通配符展开结果。



### **设备连接与基本操作**



|命令|说明|
|---|---|
|`adb devices`|查看已连接设备及状态。|
|`adb -s <serial> <command>`|对指定设备执行命令。|
|`adb start-server` / `adb kill-server`|启动 / 停止 ADB 服务。|
|`adb reconnect`|重新连接 USB 设备。|
|`adb connect <ip>:<port>`|连接无线调试设备；设备需先执行 `adb tcpip <port>`。|
|`adb disconnect [<ip>:<port>]`|断开一个或全部无线连接。|
|`adb get-state`|查看状态：`device`、`offline` 或 `unknown`。|
|`adb root`|以 root 重新启动 adbd；仅 userdebug / eng 或允许 root 的设备可用。|
|`adb remount`|将可写分区重新挂载为读写；通常需要 root。|
|`adb reboot [bootloader|recovery]`|重启设备，或进入 bootloader / recovery。|
|`adb shell stop && adb shell start`|重启 Android framework，通常比完整重启更快。|



### **安装、卸载与文件传输**



```Bash
adb install app.apk                         # 安装 APK
adb install -r app.apk                      # 覆盖安装，保留应用数据
adb install -d app.apk                      # 允许降级安装（通常还需 -r）
adb install -s app.apk                      # 安装到外部存储；仅设备支持时可用
adb uninstall <package>                     # 卸载应用及其数据
adb uninstall -k <package>                  # 卸载应用，保留数据和缓存
adb push <local_path> <device_path>         # 从电脑推送文件到设备
adb pull <device_path> <local_path>         # 从设备拉取文件到电脑
```



### **包管理与应用信息**



以下命令在 `adb shell` 中执行；也可以在每条命令前加上 `adb shell`。



```Bash
pm list packages                            # 列出所有已安装包名
pm list packages -3                         # 列出第三方应用包名
pm list packages -s                         # 列出系统应用包名
pm list packages -f                         # 显示包名及 APK 路径
pm list packages -u                         # 包含已卸载但保留数据的包
pm path <package>                           # 查看指定应用的 APK 路径
pm dump <package>                           # 查看包详细信息
pm clear <package>                          # 清除应用数据和缓存
pm disable <package>                        # 禁用应用
pm enable <package>                         # 启用应用
pm list users                               # 查看系统用户
```



### **Activity、Service 与广播**



```Bash
am start -n <package>/<activity>            # 启动 Activity
am start -n <package>/.MainActivity         # 同包名时可简写 Activity
am start -W -S -n <package>/<activity>      # 停止旧进程后启动，并输出启动耗时
am start -n <package>/<activity> --es name wzk  # 携带字符串 Intent extra
am force-stop <package>                     # 强制停止应用所有进程
am startservice -n <package>/<service>      # 启动 Service
am startservice -a <action>                 # 按 action 启动 Service
am broadcast -a <action>                    # 发送广播
am broadcast -a <action> --es name wzk      # 发送带字符串参数的广播
```



Android 8\.0 及以上对后台 Service 和隐式广播有限制。重启设备应使用

`adb reboot`，而不是发送伪造的开机完成广播。



### **当前页面、进程与系统服务**



```Bash
dumpsys activity activities                  # 查看 Activity 栈和 Task 信息
dumpsys activity top                         # 查看最上层 Activity 的 View 层级
dumpsys activity | grep -E 'mResumed|mCurrentFocus|mFocusedActivity'
                                             # 过滤当前前台页面线索
dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'
                                             # 查看当前焦点窗口和应用
dumpsys activity top | grep '#0: ' | tail -n 1
                                             # 尝试查看顶层 Fragment 信息
ps -A                                        # 查看所有进程
pidof <package>                              # 获取应用进程 PID
dumpsys -l                                   # 列出可供 dumpsys 查询的服务
dumpsys <service_name>                       # 查询指定系统服务
service list                                 # 列出 Binder 服务
```



不同 Android 版本的 `dumpsys` 输出会变化。应结合包名、组件名和实际输出判断，

不要依赖某一固定行的格式。



### **输入与屏幕模拟**



```Bash
input text 'hello%sworld'                    # 输入文本；空格用 %s 表示
input tap 500 1040                           # 点击坐标 (500, 1040)
input swipe 100 200 300 400 100              # 从起点滑到终点，耗时 100 ms
input swipe 100 200 100 200 2000             # 原地滑动约 2 秒，常用于模拟长按
input keyevent 25                            # 发送音量降低按键事件
```



```Bash
wm size                                      # 查看屏幕分辨率
wm size <width>x<height>                     # 设置临时分辨率
wm size reset                                # 恢复默认分辨率
wm density                                   # 查看屏幕密度
wm density <density>                         # 设置临时密度
wm density reset                             # 恢复默认密度
wm rotation 0                                # 设置方向：0、1、2、3
wm display-rotation                          # 查看显示旋转状态
```



**\#\#\# \`dumpsys\`、电池与 Settings**



```Bash
dumpsys battery                              # 查看电池状态
dumpsys battery unplug                       # 模拟拔掉电源，并冻结电池状态
dumpsys battery set level 50                 # 设置模拟电量为 50
dumpsys battery reset                        # 恢复真实电池状态

settings list system                         # 列出 system 表所有键值
settings get system <key>                    # 读取 system 表中的值
settings put system <key> <value>            # 写入 system 表中的值
settings list global                         # 列出 global 表所有键值
settings get global <key>                    # 读取 global 表中的值
settings put global <key> <value>            # 写入 global 表中的值
settings get secure <key>                    # 读取 secure 表中的值
settings put secure <key> <value>            # 写入 secure 表中的值
```



写入 Settings 可能需要系统权限，并可能被设备策略或系统版本限制。



### **Logcat 与 Bugreport**



```Bash
adb logcat -c                                # 清空日志缓冲区
adb logcat                                   # 持续输出默认日志
adb logcat -b all > adblogcat.txt            # 抓取所有日志缓冲区到文件
adb logcat --pid=$(adb shell pidof -s <package>)
                                             # 只输出指定应用进程日志
adb logcat -s <TAG>                          # 只输出指定 Tag
adb logcat '*:E'                             # 只输出 Error 及更高优先级
adb logcat ActivityManager:I PowerManagerService:D '*:S'
                                             # 组合 Tag 和优先级过滤
adb logcat | grep -i '<text>'                # 过滤文本，忽略大小写
adb logcat -f /sdcard/logcat.txt             # 写入设备中的文件
adb bugreport bugreport.zip                  # 导出 bugreport
adb shell setprop log.tag.<TAG> DEBUG        # 设置某 Tag 的调试日志开关
```



常用优先级从低到高为：`V`、`D`、`I`、`W`、`E`、`F`、`S`。缓冲区包括

`main`、`system`、`radio`、`events`、`crash` 和 `all`。



### **SELinux 与进程调试**



```Bash
adb shell getenforce                         # 查看 SELinux 状态
adb shell setenforce 0                       # 临时切换为 Permissive；通常需要 root
adb shell setenforce 1                       # 切回 Enforcing
adb shell ps -Af | grep '<package>'          # 查询应用进程详情
adb shell kill <pid>                         # 结束指定进程
```



`setenforce 0` 会降低设备安全性，只应在受控调试环境中临时使用，并在完成后恢复。



### **Framework 构建与部署**



在 Android 源码根目录完成环境初始化后，可按需要构建并部署：



```Bash
make framework-minus-apex
adb root
adb remount
adb push out/target/product/$TARGET_PRODUCT/system/framework/framework.jar /system/framework/
adb shell rm -rf /system/framework/oat /system/framework/arm /system/framework/arm64
adb reboot
```



`framework.jar` 的中间产物通常可在

`out/target/common/obj/JAVA_LIBRARIES/framework_intermediates/classes.jar` 找到。

实际安装路径、分区是否可写及是否需要重刷镜像，取决于设备和构建版本。



### **Git 常用命令**



```Bash
git init                                     # 初始化本地仓库
git clone <url>                              # 克隆远程仓库
git status                                   # 查看工作区和暂存区状态
git add <file>                               # 添加单个文件到暂存区
git add .                                    # 添加当前目录下所有改动到暂存区
git commit -m 'message'                      # 创建本地提交
git commit --amend                           # 修改上一笔提交（未推送时使用）
git remote add origin <url>                  # 添加远程仓库
git pull                                     # 拉取并整合远程改动
git push -u origin <branch>                  # 首次推送并关联远程分支
git push                                     # 推送当前分支
git log --oneline                            # 查看提交历史
git reflog                                   # 查看 HEAD 变更历史，可用于找回提交
git stash push -m 'message'                  # 暂存当前未提交改动
git stash pop                                # 恢复最近一次暂存
git rebase -i HEAD~<count>                   # 交互式整理最近提交
git format-patch -1 HEAD                     # 导出最近一笔提交为补丁
git apply <patch_file>                       # 应用普通补丁，不创建提交
git am <patch_file>                          # 应用 format-patch 补丁，并创建提交
```



`git reset --hard` 会丢弃未提交改动；`git push --force` 会改写远程分支历史。

执行前应确认分支、备份必要改动，并先与协作者沟通。



## **15\. 图片缓存与性能分析**



### **图片缓存**



典型策略是：内存缓存 \-\> 磁盘缓存 \-\> 网络加载。内存最快，磁盘次之，网络最慢。



- 优先使用成熟图片库（Coil、Glide、Picasso 等），由其处理采样、缓存、取消、生命周期与复用。

- 不建议手工维护“软引用二级缓存”作为主要策略：软引用回收时机不可控，可能造成缓存抖动。

- 大图应按目标尺寸解码，列表中应避免原图解码和频繁创建 Bitmap。

### **性能分析**



- 使用 Perfetto / System Trace 分析线程调度、帧时间、Binder 调用和 I/O。

- 使用 Android Studio CPU Profiler 或采样工具定位热点；`Debug.startMethodTracing()` 可生成方法追踪，但开销较大，不适合代表性线上性能测量。

- 使用 Memory Profiler 和 LeakCanary 排查内存；使用 Macrobenchmark 测量启动和滚动等用户路径。





## **15\. 诊断日志与问题定位**



`bugreport`、Event Log 和 ANR trace 是定位系统及应用问题的三类核心资料：



|资料|适合回答的问题|
|---|---|
|Bugreport|设备当时的系统全貌、服务状态、配置和历史日志是什么？|
|Event Log|Activity、进程、广播、电源等关键事件按时间如何发生？|
|ANR trace|发生 ANR 时各线程分别卡在什么调用栈？|



### **Bugreport：系统快照**



Bugreport 由 `dumpstate` 收集，是一个包含系统属性、`dumpsys` 输出、内核信息、

日志缓冲区、网络和电源状态等资料的压缩包。它反映的是抓取时或系统保留的历史状态，

不等同于某一个应用的 logcat。



```Bash
adb bugreport bugreport.zip                  # 导出完整 Bugreport
adb bugreport                                # 由 adb 自动创建并保存文件
adb shell dumpstate > dumpstate.txt          # 直接输出 dumpstate；通常不如 bugreport 完整
unzip -l bugreport.zip                        # 查看压缩包内容
unzip bugreport.zip -d bugreport              # 解压后再检索
grep -RniE 'anr|fatal exception|lmkd|low memory' bugreport/
```



不同 Android 版本的压缩包目录和文本文件名可能不同；解压后通常可从主文本报告开始搜索。

分析时先根据时间戳定位异常，再回看异常前后的系统状态。



#### **常见内容与阅读顺序**



|区域|典型内容|用途|
|---|---|---|
|Header / Build|型号、Build fingerprint、Bootloader、Kernel|确认问题发生的设备与软件版本。|
|System Properties|`ro.build.version.*`、产品和厂商属性|判断 Android 版本、SDK 和产品配置。|
|Activity Manager State|`dumpsys activity`、Task、前台组件、进程状态|确认当时前台应用、任务栈和组件状态。|
|System Log|`main`、`system`、`crash`、`events` 等日志|按时间关联崩溃、ANR、进程启动或被杀。|
|ANR 区域|ANR 原因、进程和线程栈|判断阻塞位置及系统为什么判定超时。|
|Network / Wi\-Fi|网络连接、IP、Wi\-Fi 状态|排查网络不可用、切网及连接异常。|
|Power / Battery|电量、充电状态、WakeLock、电源策略|排查待机、耗电、休眠和唤醒问题。|
|Memory / LMKD|内存统计、进程回收、低内存事件|排查进程被系统回收和内存压力。|
|Sensor / Hardware|传感器、显示、音频等服务状态|排查硬件服务或 HAL 相关问题。|



#### **崩溃示例**



```Plain Text
09-10 10:00:41.941 20183 20183 E AndroidRuntime: FATAL EXCEPTION: main
09-10 10:00:41.941 20183 20183 E AndroidRuntime: Process: com.example.myapp, PID: 20183
09-10 10:00:41.941 20183 20183 E AndroidRuntime: java.lang.NullPointerException: ...
09-10 10:00:41.941 20183 20183 E AndroidRuntime:     at com.example.myapp.MainActivity.onCreate(MainActivity.java:21)
```



- `FATAL EXCEPTION` 表示未捕获异常导致进程崩溃；`main` 表示异常发生在主线程。

- `Process` 与 `PID` 用于关联同一进程的其他日志。

- 首个业务代码栈帧通常是优先检查的位置；示例中为 `MainActivity.java:21`。

### **Event Log：关键生命周期事件**



Event Log 是结构化事件缓冲区，适合快速还原 Activity、Task、进程、广播和电源状态的

时间线。查看命令如下：



```Bash
adb logcat -b events                          # 持续查看 Event Log
adb logcat -b events -v threadtime            # 显示时间、PID、TID、优先级和 Tag
adb logcat -b events -d > events.txt          # 导出当前 Event Log 后退出
adb logcat -b events | grep -E 'am_anr|am_crash|am_proc_(start|died)|am_kill'
adb shell cat /system/etc/event-log-tags      # 查看设备声明的 Event Tag；路径因版本而异
```



典型事件格式：



```Plain Text
06-01 13:44:55.518  7361  8289 I am_create_service: [0,111484394,.StatService,10094,7769]
```



前半部分依次为时间、PID、TID、优先级和 Tag；方括号内参数由该 Tag 定义。参数数量、

顺序及对象名称会因系统版本不同而变化，应以设备上的 `event-log-tags` 和源码定义为准。



#### **常用 ActivityManager Event Tag**



|Tag|含义|排查价值|
|---|---|---|
|`am_proc_start`|启动应用进程|建立进程生命周期起点。|
|`am_proc_bound`|进程已绑定到系统服务|判断应用何时完成 attach。|
|`am_proc_died`|进程死亡|关联崩溃、LMKD 回收或主动退出。|
|`am_kill`|AMS 请求杀死进程|查看被杀原因与优先级变化。|
|`am_crash`|应用崩溃|与 `crash` 缓冲区中的 Java / native 堆栈关联。|
|`am_anr`|应用无响应|查看 ANR 发生时间、进程和超时原因。|
|`am_create_activity` / `am_resume_activity`|创建 / 恢复 Activity|还原页面启动和切换流程。|
|`am_pause_activity` / `am_finish_activity`|暂停 / 结束 Activity|分析退出、返回或启动中断。|
|`am_activity_launch_time`|Activity 启动耗时|关注 `thisTime`、`totalTime` 等耗时字段。|
|`am_activity_fully_drawn_time`|首次完整绘制耗时|评估页面达到可用状态的时间。|
|`am_create_service` / `am_destroy_service`|创建 / 销毁 Service|排查服务频繁重启或被销毁。|
|`am_broadcast_discard_*`|广播被丢弃|判断是否被过滤器、应用状态或后台限制拦截。|
|`am_low_memory` / `am_pss`|低内存与 PSS 信息|关联内存压力和进程回收。|



不要只凭 Tag 名称推断根因。例如 `am_proc_died` 仅表示进程已死亡，还需结合

`am_crash`、`am_kill`、`lmkd` 和 `crash` 日志确定具体原因。



### **ANR Trace：线程阻塞现场**



ANR 是系统检测到应用在规定时间内无法处理输入、广播、Service 或 Provider 相关工作时

生成的无响应报告。核心目标是回答：**主线程正在等待什么，以及谁持有了它需要的资源。**



```Bash
adb shell dumpsys activity processes          # 查看进程和可能的 ANR 信息
adb shell dumpsys activity lastanr            # 部分版本支持，查看最近 ANR
adb pull /data/anr/traces.txt ./traces.txt    # 旧版本常见路径；通常需要 root
adb shell ls -al /data/anr                    # 查看设备实际 ANR 文件
adb bugreport anr_bugreport.zip               # 无 root 时优先用 bugreport 收集
```



Android 版本、SELinux 权限和厂商定制会影响 ANR trace 的位置及可读性。无法读取

`/data/anr` 时，不要尝试修改权限；改用 Bugreport、`dumpsys` 和 logcat 收集信息。



#### **阅读 ANR 的步骤**



1. 记录 ANR 的 `Reason`、时间、包名与 PID，并在 Event Log 中找同一时间的 `am_anr`。

2. 找到该 PID 的 `"main"` 线程；先看最顶部的业务调用栈和线程状态。

3. 判断主线程是在 I/O、Binder 调用、锁等待、`wait()`、耗时计算，还是等待另一个线程。

4. 若是锁等待，继续找持锁线程；若是 Binder 等待，继续看被调用服务是否繁忙或死锁。

5. 对照异常前后的 `am_proc_*`、`am_kill`、GC、内存和 CPU 日志，确认是否由系统压力放大。

常见线索：



|线索|常见含义|下一步|
|---|---|---|
|`Input dispatching timed out`|窗口未能及时处理输入|检查主线程、窗口焦点与 CPU 负载。|
|`Executing service ...`|Service 生命周期回调执行过久|检查 `onCreate()`、`onStartCommand()` 中的同步工作。|
|`Broadcast of Intent ...`|广播接收器执行超时|让 `onReceive()` 快速返回，转移耗时任务。|
|`nativePollOnce`|线程在消息队列轮询|主线程出现它通常是空闲，不必单独视为卡死证据。|
|`BLOCKED` / `waiting to lock`|等待 Java 锁|找到锁拥有者及其调用栈。|
|`BinderProxy.transact`|同步等待 Binder 返回|检查被调进程、服务端线程池和循环调用。|



### **LMKD：低内存进程回收**



LMKD（Low Memory Killer Daemon）在系统内存压力较高时，根据进程优先级和内存状态

选择进程回收。它通常不是应用崩溃的根因，而是系统为恢复可用内存做出的决策。



```Bash
adb logcat -b all -v threadtime | grep -iE 'lmkd|lowmemorykiller|low memory'
adb shell dumpsys meminfo                     # 查看系统和进程内存概览
adb shell dumpsys activity oom                # 查看 OOM adj / 进程优先级
adb shell cat /proc/pressure/memory           # 支持 PSI 的设备可查看内存压力
```



```Plain Text
I lmkd    : Killing 'com.example.myapp' (pid 4567) for memory, reclaiming 5678 KB
```



- `Killing` 表示 LMKD 因内存压力终止了进程，不表示该进程发生 Java 异常。

- `pid` 用于与 `am_proc_died`、应用日志和 tombstone 关联。

- `reclaiming` 是预期回收的内存量；应继续检查回收前后的可用内存、OOM adj 与频率。

- 频繁杀死前台或重要进程，应进一步检查内存泄漏、缓存策略、系统总内存与厂商 OOM 配置。

### **Crash、ANR 与 LMKD 的关键字**



|问题类型|优先搜索的关键字|关键证据|
|---|---|---|
|Java 崩溃|`FATAL EXCEPTION`、`AndroidRuntime`、`am_crash`|异常类型与业务堆栈。|
|Native 崩溃|`Fatal signal`、`tombstone`、`DEBUG`|信号、backtrace 与对应 `.so` 符号。|
|ANR|`ANR in`、`am_anr`、`Input dispatching timed out`|ANR Reason 与各线程 trace。|
|低内存回收|`lmkd`、`Killing`、`am_kill`、`am_proc_died`|回收原因、PID、OOM adj 和内存数据。|



`dead`、`fatal`、`exception` 是宽泛关键字，容易产生大量无关结果；应优先使用上表中的

精确短语并增加包名或 PID 过滤条件。



### **Monkey：稳定性随机测试**



Monkey 会向设备随机或按比例发送用户事件，用于发现崩溃、ANR 和页面状态问题。开始前

应使用测试账号、关闭无关通知，并确保测试设备可恢复。



```Bash
adb shell monkey -p <package> --throttle 100 --monitor-native-crashes --bugreport \
    -v -v 36000 > monkey_test_log.txt
```



|参数|说明|
|---|---|
|`-p <package>`|只向指定包及其允许的组件发送事件。|
|`--throttle 100`|每两个事件之间等待约 100 ms。|
|`36000`|发送的事件数量；实际耗时取决于 throttle、设备性能和系统响应。|
|`--monitor-native-crashes`|监控 native 崩溃。|
|`--bugreport`|检测到异常时请求附加 Bugreport 信息。|
|`-v -v`|提高 Monkey 自身日志详细度。|



测试结束后，在 `monkey_test_log.txt` 中搜索 `CRASH`、`ANR`、`// NOT RESPONDING` 和

`// CRASH`，再用相同时间段的 Bugreport 与 Event Log 复盘。



### **ProtoLog：WindowManager 和 Shell 调试日志**



ProtoLog 是 Android framework / SystemUI 中可按组开启的结构化调试日志，常用于排查

窗口、动画、Transition 和 Shell Task 组织问题。具体 Tag 和命令取决于 Android 版本及

构建类型，user 版本通常不允许开启全部调试日志。



```Bash
adb shell wm logging enable-text WM_DEBUG_ANIM
adb shell wm logging disable-text WM_DEBUG_ANIM
adb shell dumpsys activity service SystemUIService WMShell protolog enable-text WM_SHELL_TASK_ORG
adb shell dumpsys activity service SystemUIService WMShell protolog disable-text WM_SHELL_TASK_ORG
```



使用前先在设备上执行相关命令的帮助或 `dumpsys` 查询确认可用组名。日志开启后可用

`adb logcat` 过滤 `ProtoLog`、`WindowManager` 或 `Shell`；完成定位后应立即关闭，避免

持续日志带来性能和存储开销。



## **16\. 图片缓存与性能分析**



### **图片缓存**



典型策略是：内存缓存 \-\> 磁盘缓存 \-\> 网络加载。内存最快，磁盘次之，网络最慢。



- 优先使用成熟图片库（Coil、Glide、Picasso 等），由其处理采样、缓存、取消、生命周期与复用。

- 不建议手工维护“软引用二级缓存”作为主要策略：软引用回收时机不可控，可能造成缓存抖动。

- 大图应按目标尺寸解码，列表中应避免原图解码和频繁创建 Bitmap。

### **性能分析**



- 使用 Perfetto / System Trace 分析线程调度、帧时间、Binder 调用和 I/O。

- 使用 Android Studio CPU Profiler 或采样工具定位热点；`Debug.startMethodTracing()` 可生成方法追踪，但开销较大，不适合代表性线上性能测量。

- 使用 Memory Profiler 和 LeakCanary 排查内存；使用 Macrobenchmark 测量启动和滚动等用户路径。

