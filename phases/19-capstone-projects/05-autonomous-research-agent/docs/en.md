# 第 05 章 — 自主研究智能体（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 已经真的发表了完整论文。Agent Laboratory 也把实验跑了起来。Allen AI 公开了执行轨迹。2026 年这类系统的标准形态，是在实验树上做计划-执行-验证的树搜索，带预算约束、沙箱化代码执行、带视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格评审集成。这个阶段项目的目标，是把它完整搭出来，每篇论文控制在 30 美元以内跑通，并且扛住 Sakana 记录过的沙箱逃逸红队。

**Type:** Capstone
**Languages:** Python（智能体 + 沙箱）、LaTeX（输出）
**Prerequisites:** 第 2 章（机器学习）、第 3 章（深度学习）、第 7 章（Transformer）、第 10 章（从零实现 LLM）、第 14 章（智能体）、第 15 章（自主系统）、第 16 章（多智能体）、第 18 章（安全）
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 小时

## 问题

自主研究智能体在 2026 年跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 以 Nature 论文的形式发表，生成的论文通过了 workshop 的同行评审。ShinkaEvolve（ICLR 2026）把这条路线扩展到了“演化假设”。AMD 的 Agent Laboratory 则交付了可复现的轨迹。这类智能体并不是魔法，而是在一棵候选实验树上运行的计划-执行-验证循环，外加成本上限、种子绑定的沙箱和自动化审查。真正的功夫在这个循环、预算和安全叙事里。

你要通过实现一个窄领域里的种子课题来学会这套循环，比如在一个 1 亿参数 Transformer 上做 attention sparsity 消融。第一轮跑出来能不能发现新东西并不重要。重要的是基础设施：树搜索、实验沙箱、写作者-审阅者循环、红队报告。Sakana 团队已经记录了沙箱逃逸失败案例；你的智能体也必须通过同样的红队。

## 概念

这个智能体本质上是一个 best-first 树搜索。节点是实验规格：（假设、配置、代码、预期结果）。展开步骤会通过小改动提出子节点，比如换优化器、调 batch size、做组件消融。每个子节点都会在一个新的沙箱里运行，且资源有硬上限。结果会反馈到评分函数里，用（新颖性 × 质量 × 剩余预算）给节点排序。树会一直长到预算耗尽，然后把最优分支写成论文。

写作者是多模态的。它会生成 LaTeX 草稿、编译、渲染图表，再把渲染后的 PDF 交回 Claude Opus 4.7 的视觉模式，检查版式、图表可读性和主张-证据对齐。一个由五个 LLM 裁判组成的评审集会输出 NeurIPS 风格分数（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文就会带着批注退回写作者继续改。

安全是系统的承重墙。每个实验都运行在 E2B 或 Daytona 沙箱里，没有网络外联、有限墙钟时间和固定资源上限。智能体的代码生成步骤还会经过策略层，阻止可能逃出沙箱的系统调用。红队报告要复现 Sakana 文档里提到的攻击面（fork bomb、文件系统逃逸、LLM 写出的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- 编排：带检查点和人工批准闸门的 LangGraph
- 树搜索：针对实验节点的自定义 best-first 搜索（Sakana v2 风格的 AB-MCTS）
- 沙箱：每个实验一个 E2B，Docker-in-Docker 作为回退；资源上限通过 cgroups 控制
- 文献：Semantic Scholar Graph API + OpenAlex + 论文摘要的本地 FAISS 缓存
- 写作者：LaTeX 模板 + Claude Opus 4.7（视觉模式），用于图表批注和版式检查
- 审阅者：由 5 个裁判组成的集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），按权重聚合
- 实验框架：PyTorch 2.5 用于实际实验，W&B 用于日志
- 可观测性：Langfuse 记录智能体轨迹，每篇论文硬预算 30 美元

## Build It

1. **定种子与定领域。** 先选一个种子想法，比如“研究 sub-1B Transformer 的 attention map 中稀疏模式”。然后定义搜索空间：模型、数据集、算力预算。

2. **文献扫描。** 用 Semantic Scholar + OpenAlex 查询最相关、被引用最多的 50 篇论文；把摘要缓存到本地；生成一页纸的领域摘要。

3. **树结构搭建。** 用种子假设初始化根节点。实现 `expand(node) -> children`，每个子节点只做一个小改动（一次只改一个配置）。实现 `score(node)`，把新颖性 × 质量 × 预算做成加权项。

4. **沙箱封装。** 每个实验都跑 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或者等价的 E2B 策略）。种子写进沙箱，输出再以只读方式挂回外部。

5. **计划-执行-验证循环。** `plan` 负责提出子节点。`execute` 负责跑沙箱、收集日志和指标。`verify` 负责对指标做单元检查（loss 有没有下降？消融是否真的隔离了效应？）。失败节点会把失败原因记回树上。

6. **写作者。** 预算跑完后，挑选最佳分支。用 matplotlib 画图。把分支轨迹放进上下文里，通过 Claude Opus 4.7 生成 LaTeX 草稿。编译，再把编译后的 PDF 交回 Opus 4.7 的视觉模式做批注。循环迭代。

7. **审阅集成。** 五位裁判按照 NeurIPS 风格的 rubric，对草稿的（新颖性、严谨性、清晰度、可复现性、影响力）打分。如果均分低于 4.0/5，就把批注退回写作者。最多重写 3 轮。

8. **红队测试。** 构建或接入一组针对沙箱的对抗任务：fork bomb、网络外传尝试、文件系统逃逸、LLM 写出的 shell 元字符。确认这些都被拦住，并整理结果。

9. **可复现性。** 每篇论文都要附上树搜索轨迹 JSON、种子、W&B 运行链接、沙箱配置，以及一个能端到端复现的 README。

## 使用示例

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付

`outputs/skill-ai-scientist.md` 就是最终交付物。给定一个种子想法、一个领域和 30 美元预算，它会跑完整条流水线，并输出一篇可审阅的论文和一整套可复现包。

| 权重 | 标准 | 评测方式 |
|:-:|---|---|
| 25 | 论文质量 | 与已发表 workshop 论文做盲审 rubric 评比 |
| 20 | 实验严谨性 | 基线、种子、消融齐全；每个主张都能在结果表中找到对应单元格 |
| 20 | 成本与算力纪律 | 严格执行每篇论文 30 美元的上限，并且全程由 Langfuse 记录 |
| 20 | 安全性 | 沙箱红队通过；网络策略和 kill-switch 都已验证 |
| 15 | 可复现性 | 用一条命令、相同种子即可重现实验论文 |
| **100** | | |

## 练习

1. 在同一领域里，用三个不同的种子想法跑这条流水线。比较树搜索的重叠部分，找出重复浪费的算力。

2. 给预计成本高于 5 美元的节点，在实验执行前加一个人工在环闸门。测总成本能降多少。

3. 把审阅集成换成单一裁判。用一个已知“坏论文”的留出集测误接受率。

4. 加一个网络外传红队测试：让智能体写代码去 `curl` 外部地址。确认 `--network=none` 策略会拦住它，并把这次尝试记录下来。

5. 把你的树搜索和一个平铺随机基线比较（同样预算，但没有展开策略）。报告新颖性 × 质量的提升幅度。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索 | “AB-MCTS 风格展开” | 在实验节点上做 best-first 探索，并用新颖性 × 质量 × 预算打分 |
| 沙箱 | “实验隔离” | 没有网络、CPU / 内存受限、种子固定、输入只读的容器 |
| 视觉批注 | “先渲染再阅读” | 把论文编译成 PDF，再把 PDF 交给 VLM 做版式和主张-证据批注 |
| 审阅集成 | “自动同行评审” | 多个 LLM 裁判按 NeurIPS rubric 给论文打分，并用加权结果控制流水线 |
| 新颖性分数 | “这新不新？” | 一个会惩罚与 50 篇文献缓存过近的启发式指标 |
| 成本上限 | “$ 预算” | 每篇论文的总花费硬上限；由 Langfuse 计数器和运行前估算共同约束 |
| 红队 | “沙箱逃逸审计” | 一组如果策略出错就会逃出沙箱的对抗任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考级生产研究智能体
- [Sakana AI-Scientist-v1 论文（arXiv:2408.06292）](https://arxiv.org/abs/2408.06292) — 原始方法论文
- [ShinkaEvolve（Sakana ICLR 2026）](https://sakana.ai) — 演化扩展
- [Agent Laboratory（AMD）](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献检索
- [E2B sandboxes](https://e2b.dev) — 参考实验隔离方案
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 审阅集成所编码的 rubric
