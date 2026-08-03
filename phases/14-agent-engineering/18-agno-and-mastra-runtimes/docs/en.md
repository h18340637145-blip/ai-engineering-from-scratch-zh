# Agno 与 Mastra：高性能与 TypeScript 原生 Agent 运行时

> 探索面向生产的高性能 Agent 运行时：Agno（轻量级 Python 框架）与 Mastra（TypeScript/Next.js 全栈 Agent 框架）。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 Lesson 01
**Time:** ~60 分钟

## 学习目标

- Identify Agno's performance targets and when they matter.
- Name Mastra's three primitives — Agents, Tools, Workflows — and the supported server adapters.
- Explain why a stateless session-scoped FastAPI backend is the recommended Agno production path.
- Pick Agno vs Mastra for a given stack (Python-first vs TypeScript-first).

## 问题切入

LangGraph, AutoGen, CrewAI are framework-heavy. Teams that want "just the agent loop, fast, in my runtime" reach for Agno (Python) or Mastra (TypeScript). Both trade some of the framework-owned primitives for raw speed and a tighter fit to the surrounding stack.

## 核心概念

### Agno

- Python runtime, formerly Phi-data.
- "No graphs, chains, or convoluted patterns — just pure python."
- Performance targets from their docs: ~2μs agent instantiation, ~3.75 KiB memory per agent, ~23 model providers.
- Production path: stateless session-scoped FastAPI backend. Each request starts a fresh agent; session state lives in a DB.
- Native multimodal (text, image, audio, video, file) and agentic RAG.

The speed targets matter when you have thousands of short-lived agents per second (chat fan-in, evaluation pipelines). They matter less when one agent runs for 10 minutes.

### Mastra

- TypeScript, built on Vercel AI SDK.
- Three primitives: **Agents**, **Tools** (Zod-typed), **Workflows**.
- Unified Model Router — 3,300+ models across 94 providers (March 2026).
- Composite storage: memory, workflows, observability to different backends; ClickHouse recommended for observability at scale.
- Apache 2.0 with `ee/` directories under source-available enterprise license.
- Server adapters for Express, Hono, Fastify, Koa; first-class Next.js and Astro integration.
- Ships Mastra Studio (localhost:4111) for debugging.
- 22k+ GitHub stars, 300k+ weekly npm downloads at 1.0 (Jan 2026).

### Positioning

Neither is trying to be LangGraph. They compete on:

- **Language fit.** Agno for Python-first teams; Mastra for TypeScript-first.
- **Runtime ergonomics.** Agno = near-zero overhead; Mastra = integrated with the Vercel ecosystem.
- **Observability.** Both integrate with Langfuse/Phoenix/Opik (Lesson 24) but Mastra Studio is first-party.

### When to pick each

- **Agno** — Python backend, many short-lived agents, strong perf requirements, FastAPI shop.
- **Mastra** — TypeScript backend, Next.js / Vercel deploy, unified multi-provider model routing, Zod-typed tools.
- **LangGraph** (Lesson 13) — when durable state and explicit graph reasoning matter more than raw speed.
- **OpenAI / Claude Agent SDK** — when you want the provider's productized shape (Lessons 16–17).

### Where this pattern goes wrong

- **Perf-for-perf's-sake.** Picking Agno because "2μs" sounds good when the workload is one slow agent call per request. Overhead is not the bottleneck.
- **Ecosystem lock-in.** Mastra's Vercel-flavored integration is a plus on Vercel, a minus elsewhere.
- **Enterprise license confusion.** Mastra's `ee/` directories are source-available, not Apache 2.0. Read the licenses if you're planning to fork.

## 动手实现

This lesson is primarily comparative — no single code artifact would do both frameworks justice. See `code/main.py` for a side-by-side toy: a minimal "run an agent, stream the output, persist session" flow implemented twice (once Agno-shaped, once Mastra-shaped).

Run it:

```
python3 code/main.py
```

Two structurally different but functionally equivalent traces.

## 应用场景

- **Agno** — Python backend that needs speed and FastAPI shape.
- **Mastra** — TypeScript backend with many providers and workflow primitives.
- Both ship first-party observability hooks. Both integrate with Langfuse.

## 产出成果

`outputs/skill-runtime-picker.md` picks Agno, Mastra, LangGraph, or a provider SDK based on stack, latency budget, and operational shape.

## 练习题

1. Read Agno's docs. Port the stdlib ReAct loop (Lesson 01) to Agno. What disappeared? What stayed?
2. Read Mastra's docs. Port the same loop to Mastra. What changed in tool typing (Zod vs nothing)?
3. Benchmark: measure agent instantiation latency on your stack. Does Agno's 2μs matter to your workload?
4. Design a migration: if you've been running CrewAI in Python, what breaks if you move to Agno?
5. Read Mastra's `ee/` license terms. What restrictions would affect an open-source fork?

## 核心术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## 深入阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets, FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — the stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — observability comparisons cited by Mastra integrations
