# LLM 系统评估与基准测试（LLM Evaluation）

> 不可衡量即无法改进：建立针对 RAG、Agent 与生成质量的自动化评估体系。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 1 - Lesson 9
**Time:** ~75 分钟

## 学习目标

- 理解基于 LLM-as-a-Judge（以模型为裁判）的评测机制及其偏置消除方法
- 掌握 RAG 评测四大核心指标：上下文相关度、答案忠实度、答案相关度与上下文召回率（Ragas 框架）
- 构建包含单元测试、回归测试与 Golden Dataset（黄金数据集）的自动化 Eval 管道
- 使用 Exact Match、BLEU/ROUGE 与自定义断言做系统验证

## 核心问题

对于生成式模型，简单的字符串相等断言无法衡量回答的好坏。没有科学的评估测试体系，Prompt 改进或模型切换就如同盲人摸象，难以确保生产稳定性。

## 动手构建

参见 `code/main.py`。
