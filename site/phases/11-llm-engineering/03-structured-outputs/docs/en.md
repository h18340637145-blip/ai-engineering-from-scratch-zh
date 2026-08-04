# 结构化输出（Structured Outputs）

> 让大模型输出可确定解析的 JSON，从根源消除脆弱的正则表达式与解析报错。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 1（Prompt 工程）
**Time:** ~60 分钟

## 学习目标

- 掌握通过 Pydantic / Zod 定义结构化输出 Schema 的方法
- 理解约束解码（Constrained Decoding）与上下文无关文法（CFG）的底层原理
- 使用 OpenAI / Hono 原生 JSON Mode 与 Structured Outputs API
- 实现结构化数据的校验、错误自动重试与格式修复机制

## 核心问题

自由格式的文本响应难以直接集成到软件系统中。传统做法依赖于在 Prompt 中强求 JSON 格式并用正则表达式提取，但这极易因 Token 截断、标点遗漏或格式漂移导致运行时解析崩溃。

## 动手构建

参见 `code/main.py` 与 `code/main.ts`。
