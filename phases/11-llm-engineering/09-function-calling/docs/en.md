# 工具调用与函数调用（Function Calling & Tool Use）

> 赋予模型连接外部世界的能力：从纯文本生成迈向 API 执行与自动化工具代理。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 3（结构化输出）
**Time:** ~75 分钟

## 学习目标

- 理解 Function Calling 的核心循环：模型识别意图 -> 返回结构化参数 -> 客户端执行 -> 结果写回模型
- 掌握编写符合 JSON Schema 的工具定义（Tools Definition）
- 实现多工具调用（Parallel / Multi-turn Tool Calling）与循环纠错机制
- 建立安全的工具执行沙箱与权限审批控制

## 核心问题

LLM 本身无法直接查询实时天气、读写数据库或操作本地文件。Function Calling 允许模型在推理过程中暂停并输出结构化的工具调用请求，由客户端执行后将结果反馈给模型，从而完成复杂的真实任务。

## 动手构建

参见 `code/main.py`。
