# 安全防护网与内容审核（Guardrails & Safety）

> 构建双向防护屏障：防范 Prompt 注入攻击、敏感数据泄露与不合规输出。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 1, Lesson 9
**Time:** ~75 分钟

## 学习目标

- 防范直接与间接 Prompt 注入攻击（Prompt Injection & Jailbreaking）
- 实现输入侧 PII（个人身份敏感信息）脱敏与输出侧有害内容分类过滤
- 使用 Guardrails AI / Llama Guard 构建双向检测防护层
- 设计优雅的安全拦截降级响应机制

## 核心问题

大模型可能会被恶意用户通过越狱 Prompt 诱导输出有害言论，或者在读取外部不可信网页时遭遇间接 Prompt 注入攻击导致隐私泄露。系统必须配备独立的隔离与检测墙。

## 动手构建

参见 `code/main.py`。
