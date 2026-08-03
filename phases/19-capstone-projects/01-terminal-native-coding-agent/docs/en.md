# 综合项目 01——终端原生编码智能体

> 到 2026 年，编码智能体的形态已经稳定下来：TUI 框架、有状态计划、沙盒化工具界面，以及规划、行动、观察、恢复的循环。远看之下，Claude Code、Cursor 3 和 OpenCode 的架构大同小异。本综合项目要求你端到端构建一个编码智能体——从 CLI 接收任务，最终输出 Pull Request——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 对比。你将理解真正的难点并非模型调用，而是工具循环、沙盒，以及 50 轮运行中的成本上限。

**类型：** 综合项目
**语言：** TypeScript / Bun（智能体框架）、Python（评估脚本）
**前置课程：** 阶段 11（LLM 工程）、阶段 13（工具与协议）、阶段 14（智能体）、阶段 15（自主系统）、阶段 17（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

2026 年，编码智能体成为主导性的 AI 应用类别。Claude Code (Anthropic)、搭载 Composer 2 和 Agent Tabs 的 Cursor 3 (Cursor)、Amp (Sourcegraph)、OpenCode（11.2 万 Star）、Factory Droids 与 Google Jules，都交付了同一架构的不同变体：终端框架、受权限约束的工具界面、沙盒，以及围绕前沿模型构建的「规划—行动—观察」循环。前沿能力的差距很窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%——但工程实践的范围很广。大多数故障并非模型犯错，而是工具循环不稳定、上下文污染、Token 成本失控以及破坏性文件系统操作。

你无法只从外部观察来真正理解这些智能体。你必须亲手构建一个，看到它在第 47 轮因 ripgrep 返回 8 MB 匹配结果而崩溃，然后重新设计截断层。这正是本综合项目的意义。

## 核心概念

智能体框架包含 4 个界面。**规划（Plan）**维护 TodoWrite 风格的状态对象，模型每轮都会完整重写该对象。**行动（Act）**分派工具调用（读取、编辑、运行、搜索、Git）。**观察（Observe）**捕获 stdout、stderr 和退出码，截断结果后将摘要反馈给模型。**恢复（Recover）**处理工具错误，同时避免撑爆上下文窗口或陷入无限循环。2026 年的架构还增加了一项能力：**钩子（Hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` 都是可配置的扩展点，操作者可以在其中注入策略、遥测和护栏。

沙盒使用 E2B 或 Daytona。每个任务都在全新的开发容器中运行，并以读写方式挂载一个 Git 工作树。智能体框架绝不接触宿主机文件系统；无论任务成功还是失败，工作树都会被销毁。成本控制分 3 层执行：每轮 Token 上限、每会话金额预算，以及硬性轮数限制（通常为 50 轮）。可观测层采用遵循 GenAI 语义约定的 OpenTelemetry Span，并将数据发送到自托管 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5（终端中的 React）
- 模型访问：通过 OpenRouter 统一 API 使用 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro，以及用于最困难任务的 Opus 4.5
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙盒：E2B 沙盒 (JS SDK) 或 Daytona 开发容器
- 代码搜索：ripgrep 子进程，以及预编译的 17 种语言 tree-sitter 解析器
- 隔离：每个任务执行一次 `git worktree add`，成功或失败后均清理
- 评估框架：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 自建的 30 任务保留集
- 可观测性：采用 `gen_ai.*` 语义约定的 OpenTelemetry SDK → 自托管 Langfuse
- PR 发布：使用细粒度 Token 的 GitHub App，权限范围仅限目标仓库

## 动手构建

1. **TUI 与命令循环。** 使用 Ink 搭建 Bun 项目，接受 `agent run <repo> "<task>"` 命令。输出分栏视图：顶部为计划面板，中部为工具调用流，底部为 Token 预算。支持通过 Ctrl-C 取消，并在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义带类型的 TodoWrite Schema，其中包含附有备注的 pending、in_progress 和 done 条目。模型每轮通过工具调用重写完整状态，不允许增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具界面。** 定义 6 个工具：`read_file`、带 Diff 预览的 `edit_file`、`ripgrep`、`tree_sitter_symbols`、带超时的 `run_shell`，以及支持 status、diff、commit、push 的 `git`。通过 MCP StreamableHTTP 暴露工具，使框架不依赖具体传输方式。所有工具都返回截断后的输出，每次调用最多 4k Token。

4. **沙盒封装。** 每个任务启动一个 E2B 沙盒，并通过 `git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用都在沙盒内执行，无法访问宿主机文件系统。

5. **钩子。** 实现 2026 年的全部 8 种钩子。至少接入 4 个用户编写的钩子：(a) `PreToolUse` 破坏性命令护栏，阻止在工作树外执行 `rm -rf`；(b) `PostToolUse` Token 统计；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆包含 30 个问题的 SWE-bench Pro Python 子集，逐个运行你的框架，并在 pass@1、每任务轮数和每任务成本上与最小基线 mini-swe-agent 对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 设置硬性上限：50 轮、200k 上下文、每任务 5 美元。达到 150k 时，`PreCompact` 钩子将较早轮次总结为先前状态块，在不丢失计划的前提下为新观察腾出空间。

8. **发布 PR。** 成功后的最后一步是执行 `git push`，再调用 GitHub API 创建 PR，并在正文中写入计划和 Diff 摘要。

## 实际使用

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付成果

交付的 Skill 位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它会在沙盒中运行完整的「规划—行动—观察」循环，并返回 PR URL 和追踪包。本综合项目的评分标准如下：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 相对基线 | 在 30 个配对 Python 任务上对比你的框架与 mini-swe-agent |
| 20 | 架构清晰度 | 对照 Live-SWE-agent 架构审查规划、行动、观察的分离，钩子界面和工具 Schema |
| 20 | 安全性 | 沙盒逃逸测试、权限提示、破坏性命令护栏通过红队测试 |
| 20 | 可观测性 | 追踪完整度（100% 的工具调用都有 Span）、逐轮 Token 统计 |
| 15 | 开发者体验 | 冷启动 < 2 秒；崩溃恢复能继续计划；Ctrl-C 可干净地取消执行中的工具 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 换成由 vLLM 提供服务的 Qwen3-Coder-30B。比较 pass@1 和每任务成本，并报告开源模型表现较差的场景。

2. 添加 `reviewer` 子智能体，在发布 PR 前读取 Diff，并能要求进入修改循环。衡量误报式评审是否会让 SWE-bench 通过率低于单智能体基线（提示：通常会）。

3. 对沙盒进行压力测试：编写一个尝试通过 `curl` 访问外部 URL 的任务，以及一个尝试写入工作树外部的任务。确认二者都被 PreToolUse 钩子阻止，并记录这些尝试。

4. 使用较小的模型 (Haiku 4.5) 实现 `PreCompact` 摘要，衡量经过 3 次压缩后损失了多少计划保真度。

5. 将 MCP StreamableHTTP 传输替换为 stdio，测试冷启动和每次调用延迟，并选出更适合纯本地使用的方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Harness | 「智能体循环」 | 包围模型的代码，负责分派工具、维护计划状态并执行预算限制 |
| Hook | 「智能体事件监听器」 | 由用户编写、框架在 8 种生命周期事件之一发生时运行的脚本 |
| Worktree | 「Git 沙盒」 | 位于独立路径的关联 Git 检出，可直接丢弃而不影响主克隆 |
| TodoWrite | 「计划状态」 | 由模型每轮重写的带类型 pending、in-progress、done 条目列表 |
| StreamableHTTP | 「MCP 传输」 | 2026 年 MCP 修订版：支持双向流式传输的长连接 HTTP，取代 SSE |
| Token ceiling | 「上下文预算」 | 每轮或每会话的输入与输出 Token 上限，触发压缩或终止 |
| pass@1 | 「单次尝试通过率」 | 无需重试或窥探测试集，首次运行便解决的 SWE-bench 任务比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code)——Anthropic 的参考框架
- [Cursor 3 更新日志](https://cursor.com/changelog)——Agent Tabs 与 Composer 2 的产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)——用于 SWE-bench 框架对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent)——使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai)——拥有 11.2 万 Star 的开放框架
- [SWE-bench Pro 排行榜](https://www.swebench.com)——本综合项目的目标评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)——StreamableHTTP 与能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——工具调用和 Token 用量的 Span Schema
