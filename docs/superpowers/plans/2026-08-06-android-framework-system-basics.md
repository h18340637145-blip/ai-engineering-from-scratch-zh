# Android Framework 系统基础课程阶段实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `docs/AndroidFramework/Android Framework 基础.md` 的第 17–25 节转为 9 节可运行的 Android Framework 工程课程。

**架构：** 新建 `phases/22-android-framework-system-basics/`。每一课使用 Python 标准库模型或解析器复现一个可验证的系统工程决策，并把源码路径、设备命令和版本差异写入中文课程文档。阶段封面用 imagegen 生成的无文字插画，精确调用链保持 Mermaid。

**技术栈：** Markdown、Python 3 标准库、`unittest`、Mermaid、内置 imagegen。

---

## 课程文件清单

| 课次 | 目录 | 可运行接口 | 核心验证 |
|---|---|---|---|
| 01 | `01-boot-chain-and-bootanimation` | `BootTimeline.find_gap()`、`BootAnimationSpec.parse()` | 启动阶段、`desc.txt`、可中断段 |
| 02 | `02-native-log-and-callstack` | `NativeModuleConfig.validate()` | `LOG_NDEBUG`、CallStack、`libutils`/`liblog` |
| 03 | `03-zygote-and-system-server` | `ZygoteRc.parse()`、`ZygotePlanner.spawn()` | `ro.zygote`、socket、fork 链 |
| 04 | `04-android-build-and-partition` | `ModuleSpec.validate()` | `Android.mk`、Android.bp、分区与特权边界 |
| 05 | `05-resource-overlays` | `OverlayRegistry.resolve()` | SRO/RRO、target、category、priority |
| 06 | `06-system-properties-and-settings` | `PropertyPolicy.can_write()`、`SettingsObserver` | `ro.*`、`persist.*`、Settings 范围与观察者 |
| 07 | `07-setup-wizard-and-provisioning` | `ProvisioningState.next_home()`、`SetupIntentValidator` | device/user 状态与 SETUP_WIZARD 匹配 |
| 08 | `08-gms-integration-and-customization` | `PrebuiltPackage.validate()`、`WizardScriptGraph` | 预置模块、分区、partner customization |
| 09 | `09-framework-integration-lab` | `FrameworkReadinessReport.evaluate()` | 启动、Overlay、向导、权限与验证闭环 |

## 任务 1：建立阶段外观与第 01–03 课（启动与原生运行时）

**文件：**
- 创建：`phases/22-android-framework-system-basics/assets/android-framework-system-cover.png`
- 创建：`phases/22-android-framework-system-basics/{01-boot-chain-and-bootanimation,02-native-log-and-callstack,03-zygote-and-system-server}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(BootAnimationSpec.parse("1080 1920 30\np 0 0 part0").fps, 30)
self.assertTrue(NativeModuleConfig(log_ndebug=0, libs={"libutils", "liblog"}).validate().ok)
self.assertIn("system_server", ZygotePlanner.spawn("zygote64"))
```

- [x] **步骤 2：运行测试验证失败**

运行：第 01–03 课各自 `python3 -m unittest discover code/tests -v`。

预期：测试报告尚未实现的模型接口。

- [x] **步骤 3：编写最少实现代码**

```python
@dataclass
class BootAnimationSpec:
    width: int
    height: int
    fps: int
    parts: list[AnimationPart]
```

解析器拒绝少于三个首行字段的 `desc.txt`；原生模块仅在源代码确实使用 API 时建议加入相应库；Zygote 计划从 init → zygote → system_server/app process 生成序列。

- [x] **步骤 4：运行测试验证通过**

运行：第 01–03 课各自 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：每课至少 5 个测试均通过。

- [x] **步骤 5：按课程提交**

为每课单独提交，主题为 `feat(phase-22/01): add boot animation`、`feat(phase-22/02): add native callstack`、`feat(phase-22/03): add zygote system server`。

## 任务 2：建立第 04–06 课（构建、资源与设置）

**文件：**
- 创建：`phases/22-android-framework-system-basics/{04-android-build-and-partition,05-resource-overlays,06-system-properties-and-settings}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertFalse(ModuleSpec("Demo", "vendor", privileged=True).validate().ok)
self.assertEqual(registry.resolve("navigation", user=0).package, "com.example.gesture")
self.assertFalse(PropertyPolicy.can_write("ro.example.flag", is_system=True))
```

- [x] **步骤 2：运行测试验证失败**

运行：第 04–06 课各自 `python3 -m unittest discover code/tests -v`。

预期：依赖接口缺失而失败。

- [x] **步骤 3：编写最少实现代码**

```python
def can_write(name: str, is_system: bool) -> bool:
    return is_system and not name.startswith("ro.")
```

模块校验器拒绝不匹配的 `privileged` 与分区组合；Overlay 解析器在同一 category 中选择已启用且优先级最高的 RRO；SettingsObserver 只为已注册 URI 发布变化。

- [x] **步骤 4：运行测试验证通过**

运行：第 04–06 课各自 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：三课全部通过且无第三方依赖。

- [x] **步骤 5：按课程提交**

分别提交第 04、05、06 课，主题依次为 `feat(phase-22/04): add build partition`、`feat(phase-22/05): add resource overlays`、`feat(phase-22/06): add system settings`。

## 任务 3：建立第 07–09 课（向导、GMS 与集成演练）

**文件：**
- 创建：`phases/22-android-framework-system-basics/{07-setup-wizard-and-provisioning,08-gms-integration-and-customization,09-framework-integration-lab}/{docs/en.md,code/main.py,code/tests/test_main.py,quiz.json,outputs/skill-*.md}`

- [x] **步骤 1：编写失败的测试**

```python
self.assertEqual(ProvisioningState(0, 0).next_home(), "SetupWizard")
self.assertFalse(SetupIntentValidator(["MAIN", "HOME"]).is_valid())
self.assertTrue(PrebuiltPackage("GmsCore", "product", "PRESIGNED").validate().ok)
self.assertFalse(FrameworkReadinessReport({"boot": True, "overlay": False}).ready)
```

- [x] **步骤 2：运行测试验证失败**

运行：第 07–09 课各自 `python3 -m unittest discover code/tests -v`。

预期：新模型接口尚未实现。

- [x] **步骤 3：编写最少实现代码**

```python
def next_home(self) -> str:
    return "Launcher" if self.device_provisioned and self.user_setup_complete else "SetupWizard"
```

Setup Intent 校验器要求 `MAIN` 与 `SETUP_WIZARD`，并拒绝为提高匹配率而同时声明 `HOME`；预置包校验器要求明确分区、证书和特权标记；集成报告列出失败的系统层及下一条验证命令。

- [x] **步骤 4：运行测试验证通过**

运行：第 07–09 课各自 `python3 code/main.py && python3 -m unittest discover code/tests -v`。

预期：每课主程序退出 0，且 5 个以上测试通过。

- [x] **步骤 5：按课程提交**

逐课提交第 07、08、09 课，主题为 `feat(phase-22/07): add setup wizard`、`feat(phase-22/08): add gms customization`、`feat(phase-22/09): add framework integration`。

## 任务 4：登记课程并完成阶段验证

**文件：**
- 修改：`README.md` 的课程目录
- 修改：`ROADMAP.md` 的课程状态
- 修改：`glossary/terms.md` 的术语定义（BootAnimation、SystemServer、RRO、SRO、System Property、Setup Wizard）

- [x] **步骤 1：写入 9 条 Markdown 课程链接和 Phase 22 状态表。**
- [x] **步骤 2：运行 `python3 scripts/audit_lessons.py --phase 22`，预期 0 issue。**
- [x] **步骤 3：运行 9 个课程 `main.py` 和 `unittest`，预期全部退出 0。**
- [x] **步骤 4：运行 `python3 scripts/check_readme_counts.py`、`node site/build.js` 和 `git diff --check`。**
- [x] **步骤 5：提交 README、ROADMAP 与术语表；不提交 `site/data.js`。**

## 自检

- 覆盖源文档的开机动画、Native 日志、Zygote、构建、分区、Overlay、属性、Setup Wizard 与 GMS 章节。
- 测验题只使用原文或可由原文推导的工程结论，且保持 1 个 pre、3 个 check、2 个 post 的标准结构。
- PNG 仅作为无文字阶段封面；调用链和文件结构均用 Mermaid 展示。
