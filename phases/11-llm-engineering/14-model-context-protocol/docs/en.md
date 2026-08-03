# 模型上下文协议（MCP - Model Context Protocol）

> Anthropic 提出的统一开放标准：连接大模型与本地及远程数据源、工具的通用接口。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 9（Function Calling）
**Time:** ~75 分钟

## 学习目标

- 理解 MCP 的核心设计理念：Server-Client 架构、Resources、Tools 与 Prompts
- 使用 Python / TypeScript SDK 从零构建自定义 MCP Server
- 实现 MCP 协议的 JSON-RPC 2.0 通信层与 Stdio / SSE 传输通道
- 在 Claude Desktop / Cursor 等客户端中连接并调试 MCP 服务

## 核心问题

在 MCP 出现前，每个 AI 框架和应用都需要为数据库、Git 仓库或外部 API 编写专属的集成分接器。MCP 提供了一套标准化的协议，让大模型应用能够一键插拔式访问各种数据与工具。

## 动手构建

参见 `code/main.py`。
