# Java 与 Android 基础课程阶段实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `docs/AndroidFramework/Java android .md` 拆为 14 节可运行、可测验、可复用的 Android Framework 前置课程。

**架构：** 新建 `phases/21-java-android-foundations/`。每节课以一个标准库 Python 模型还原一个 Android/Java 决策点，文档保留「概念—构建—运行—诊断」闭环，题目仅考察源文档涵盖的事实与推理。阶段封面是无文字的教学插画；所有精确流程图用 Mermaid。

**技术栈：** Markdown、Python 3 标准库、`unittest`、Mermaid、内置 imagegen。

---

## 课程文件清单

| 课次 | 目录 | 可运行接口 | 核心验证 |
|---|---|---|---|
| 01 | `01-java-collections-and-equality` | `CollectionAdvisor.recommend()`、`keys_collide()` | List/Set/Map 选择与 `equals`/`hashCode` 契约 |
| 02 | `02-java-oop-generics-and-strings` | `choose_text_container()`、`generic_access()` | String 可变性、PECS、接口与抽象类 |
| 03 | `03-reflection-serialization-and-gc` | `inspect_type()`、`is_collectable()` | 反射边界、序列化风险、GC Roots |
| 04 | `04-process-thread-and-ipc` | `choose_ipc()`、`ThreadOperation.describe()` | 进程线程边界、IPC 选型、`wait`/`sleep` |
| 05 | `05-handler-looper-and-concurrency` | `MessageQueueSimulator` | 延迟消息顺序、移除回调、主线程 Looper |
| 06 | `06-view-rendering-and-touch` | `RenderRequest.plan()`、`TouchDispatcher.dispatch()` | measure/layout/draw、拦截与 CANCEL |
| 07 | `07-list-rendering-and-image-cache` | `ListPerformanceAdvisor`、`ImageCache` | RecyclerView 优化、三级图片缓存 |
| 08 | `08-memory-oom-and-anr` | `IncidentClassifier.classify()` | 泄漏、OOM、ANR 的区别和下一步 |
| 09 | `09-activity-window-and-service` | `LifecyclePlanner`、`LaunchModeResolver` | 生命周期、Window 层级、Service 模式 |
| 10 | `10-media-jni-network-and-security` | `TransportAdvisor`、`CryptoAdvisor` | Media/JNI 边界、TCP/UDP、加密适用面 |
| 11 | `11-architecture-performance-and-storage` | `ArchitectureAdvisor`、`OptimizationPlan` | MVC/MVP/MVVM、测量优先、跨进程存储 |
| 12 | `12-binder-classloading-build-and-install` | `BinderTransaction.check()`、`BuildPipeline` | Binder 大事务、类加载、APK 构建流水线 |
| 13 | `13-linux-adb-and-device-operations` | `AdbCommandPlanner` | 设备选择、包管理、dumpsys、危险命令保护 |
| 14 | `14-system-diagnostics-and-stability` | `IncidentTimeline.analyze()` | bugreport、Event Log、ANR、LMKD、Monkey、ProtoLog |

## 任务 1：建立阶段外观与第 01–03 课（Java 语言基础）

**文件：**
- 创建：`phases/21-java-android-foundations/assets/java-android-foundations-cover.png`
- 创建：`phases/21-java-android-foundations/{01-java-collections-and-equality,02-java-oop-generics-and-strings,03-reflection-serialization-and-gc}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(CollectionAdvisor.recommend("ordered", "read-heavy"), "ArrayList")
self.assertEqual(generic_access("extends"), "read")
self.assertTrue(is_collectable({"thread_stack": False, "static": False, "jni": False}))
```

- [x] **步骤 2：运行测试验证失败**

运行：`python3 -m unittest discover -s phases/21-java-android-foundations -p test_main.py -v`

预期：导入 `main` 后找不到上述接口，测试因「尚未实现课程模型」断言失败。

- [x] **步骤 3：编写最少实现代码**

```python
def generic_access(bound: str) -> str:
    return "read" if bound == "extends" else "write"

def is_collectable(roots: dict[str, bool]) -> bool:
    return not any(roots.values())
```

每课补齐至少 5 个针对正常路径与边界条件的 `unittest`，并在 `main()` 输出可读的诊断演示。

- [x] **步骤 4：运行测试验证通过**

运行：`python3 -m unittest discover -s phases/21-java-android-foundations/01-java-collections-and-equality/code/tests -v`，以及第 02、03 课相同命令。

预期：每课至少 5 个测试均为 `OK`。

- [x] **步骤 5：按课程提交**

```bash
git add phases/21-java-android-foundations/01-java-collections-and-equality
git commit -m "feat(phase-21/01): add java collections equality"
```

对第 02、03 课分别重复独立提交，绝不合并为一笔。

## 任务 2：建立第 04–06 课（并发与渲染）

**文件：**
- 创建：`phases/21-java-android-foundations/{04-process-thread-and-ipc,05-handler-looper-and-concurrency,06-view-rendering-and-touch}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(choose_ipc("rpc", 64), "Binder/AIDL")
self.assertEqual(queue.next_due(10).name, "render")
self.assertEqual(dispatcher.dispatch("DOWN", child_consumes=True), "child")
```

- [x] **步骤 2：运行测试验证失败**

运行：`python3 -m unittest discover -s phases/21-java-android-foundations/04-process-thread-and-ipc/code/tests -v`，以及第 05、06 课相同命令。

预期：接口尚不存在导致明确失败。

- [x] **步骤 3：编写最少实现代码**

```python
def choose_ipc(kind: str, payload_kb: int) -> str:
    if kind == "rpc" and payload_kb <= 1024:
        return "Binder/AIDL"
    return "ContentProvider" if kind == "data" else "Socket"
```

消息队列按 `when_ms` 排序；触摸分发在父级拦截时向子级生成 `CANCEL` 记录。

- [x] **步骤 4：运行测试验证通过**

运行：各课 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：三课主程序退出码为 0，且每课 5 个以上测试通过。

- [x] **步骤 5：按课程提交**

分别提交第 04、05、06 课，提交主题依次为 `feat(phase-21/04): add process ipc`、`feat(phase-21/05): add handler looper`、`feat(phase-21/06): add view touch`。

## 任务 3：建立第 07–10 课（运行时组件与性能）

**文件：**
- 创建：`phases/21-java-android-foundations/{07-list-rendering-and-image-cache,08-memory-oom-and-anr,09-activity-window-and-service,10-media-jni-network-and-security}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(ImageCache.fetch("avatar"), "network")
self.assertEqual(IncidentClassifier.classify("waiting to lock"), "ANR")
self.assertEqual(LifecyclePlanner.foreground_path(), ["onCreate", "onStart", "onResume"])
self.assertEqual(TransportAdvisor.choose("low-latency", False), "UDP")
```

- [x] **步骤 2：运行测试验证失败**

运行：第 07–10 课各自 `python3 -m unittest discover code/tests -v`。

预期：所有新接口尚未定义，测试按缺失功能失败。

- [x] **步骤 3：编写最少实现代码**

```python
def choose(self, goal: str, reliable: bool) -> str:
    return "TCP" if reliable else "UDP"
```

`ImageCache` 固定按 memory → disk → network 返回来源；`IncidentClassifier` 将 `FATAL EXCEPTION`、超时和内存申请失败区分；生命周期模型不得假设 `onDestroy()` 必然执行。

- [x] **步骤 4：运行测试验证通过**

运行：第 07–10 课各自 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：共 20 个以上断言通过，程序无外部依赖。

- [x] **步骤 5：按课程提交**

为每个课程目录单独提交，主题采用 `feat(phase-21/07): add list image cache` 至 `feat(phase-21/10): add media jni network`。

## 任务 4：建立第 11–14 课（构建与诊断工具链）

**文件：**
- 创建：`phases/21-java-android-foundations/{11-architecture-performance-and-storage,12-binder-classloading-build-and-install,13-linux-adb-and-device-operations,14-system-diagnostics-and-stability}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(ArchitectureAdvisor.recommend("observable-state"), "MVVM")
self.assertFalse(BinderTransaction(2_000_000).is_safe())
self.assertIn("-s emulator-5554", AdbCommandPlanner.for_device("emulator-5554", "shell pm list packages"))
self.assertEqual(IncidentTimeline.analyze(["am_anr", "BinderProxy.transact"])["kind"], "ANR")
```

- [x] **步骤 2：运行测试验证失败**

运行：第 11–14 课各自 `python3 -m unittest discover code/tests -v`。

预期：模型尚未存在，测试失败原因是功能缺失。

- [x] **步骤 3：编写最少实现代码**

```python
class BinderTransaction:
    def __init__(self, bytes_count: int) -> None:
        self.bytes_count = bytes_count
    def is_safe(self) -> bool:
        return self.bytes_count <= 1_048_576
```

命令规划器拒绝含有未限定 `rm -rf *` 的命令；诊断时间线以精确关键字区分 Crash、ANR 与 LMKD。

- [x] **步骤 4：运行测试验证通过**

运行：第 11–14 课各自 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：每课至少 5 个测试为 `OK`。

- [x] **步骤 5：按课程提交**

逐课提交第 11–14 课；提交主题以 `feat(phase-21/11): add architecture performance` 至 `feat(phase-21/14): add diagnostics stability` 为准。

## 任务 5：登记课程并完成阶段验证

**文件：**
- 修改：`README.md` 的 Phase 20 之后课程目录
- 修改：`ROADMAP.md` 的 Phase 20 之后阶段状态
- 修改：`glossary/terms.md` 的术语定义（Binder、Looper、ANR、LMKD、Zygote）

- [x] **步骤 1：写入 14 条 Markdown 课程链接和 Phase 21 状态表。**
- [x] **步骤 2：运行 `python3 scripts/audit_lessons.py --phase 21`，预期 0 issue。**
- [x] **步骤 3：运行每课 `main.py` 与 `unittest`，预期 14 个主程序均退出 0。**
- [x] **步骤 4：运行 `python3 scripts/check_readme_counts.py`、`node site/build.js` 和 `git diff --check`。**
- [x] **步骤 5：提交 README、ROADMAP 与术语表；不得提交 `site/data.js`。**

## 自检

- 覆盖源文档的 Java 集合、对象模型、GC、IPC、渲染、组件、网络、构建、ADB 与系统诊断章节。
- 每课均具备 H1、中文学习目标、Mermaid 技术图、6 道标准题型、可运行代码、5 个以上单元测试和 `skill-` 产物。
- 所有图形关系使用 Mermaid；`imagegen` 生成的 PNG 仅作为无文字课程封面。
