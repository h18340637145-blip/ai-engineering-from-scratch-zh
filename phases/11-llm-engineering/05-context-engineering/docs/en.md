# 上下文工程（Context Engineering）

> 上下文窗口就是模型的短期工作内存：如何高密度、低噪声地装载关键信息。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 1（Prompt 工程）
**Time:** ~60 分钟

## 学习目标

- 掌握上下文窗口的预算规划与 Token 挤压策略
- 理解“大海捞针”（Needle in a Haystack）现象与中间丢失（Lost in the Middle）难题
- 实现基于 XML 标签与标记语言的上下文分割与注入防护
- 搭建动态对话历史滑动窗口与增量摘要机制

## 核心问题

上下文窗口不仅价格昂贵，而且并非无限容量。将无关或冗余信息填满上下文会导致模型注意力稀释、响应延迟暴涨以及关键事实遗漏。精细化管理上下文是构建生产级 AI 应用的关键。

## 动手构建

参见 `code/main.py`。
