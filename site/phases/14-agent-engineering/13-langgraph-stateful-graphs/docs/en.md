# LangGraph：基于状态图与检查点机制的 Agent 编排

>  LangGraph 将 Agent 系统构建为显式状态机（StateGraph），引入持久化检查点（Checkpoints）以支持人机回退与时间旅行调试。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 Lesson 01, Lesson 12
**Time:** ~60 分钟

## 学习目标

- Describe LangGraph's core model: state machine with typed state, function nodes, conditional edges, and post-node checkpoints.
- Name the four capabilities the docs highlight: durable execution, streaming, human-in-the-loop, comprehensive memory.
- Explain the three orchestration topologies LangGraph supports: supervisor, peer-to-peer (swarm), hierarchical (nested subgraphs).
- Implement a stdlib state graph with typed state, conditional edges, and a checkpoint/resume cycle.

## 问题切入

Agents and workflows share a problem: when a 40-step run fails at step 38, you want to resume from step 38, not start over. Second-class state models leave operators hacking retries around a library that assumes fresh runs.

LangGraph's design answer: state is a first-class typed object, mutations are explicit, and checkpoints persist after every node. Resume is a `load_state(session_id)` call.

## 核心概念

### The graph

A graph is defined by:

- **State type.** A typed dict (or Pydantic model) that every node reads and mutates.
- **Nodes.** Pure functions `(state) -> state_update`. Updates are merged into state after return.
- **Edges.** Conditional or direct transitions between nodes.
- **Entry and exit.** `START` and `END` sentinel nodes mark the boundary.

Example: an agent with `classify`, `refund`, `bug`, `sales`, `done` nodes — a routing workflow as a graph.

### Durable execution

After each node returns, the runtime serializes the state and writes it to a checkpointer (SQLite, Postgres, Redis, custom). On failure at step N, the runtime can `resume(session_id)` and pick up from step N+1 with exact state.

The LangGraph docs explicitly highlight production users where this matters: Klarna, Uber, J.P. Morgan. The claim isn't the graph shape; it's that the graph shape plus checkpointing makes recovery cheap.

### Streaming

Every node can yield partial output. The graph streams per-node-delta events to the caller so UIs update as the graph runs.

### Human-in-the-loop

Inspect and modify state between nodes. Implementations: pause before a critical node, surface state to a human, accept modifications, resume. The checkpointer makes this easy because state is already serialized.

### Memory

Short-term (within a run — conversation history in state) and long-term (across runs — persistent via the checkpointer plus a separate long-term store). LangGraph integrates with external memory systems (Mem0, custom) via tools.

### Three topologies

1. **Supervisor.** Central router LLM dispatches to specialist subagents. `create_supervisor()` in `langgraph-supervisor` (though the LangChain team in 2026 recommends doing this through tool calls directly for more context control).
2. **Swarm / peer-to-peer.** Agents hand off directly via a shared tool surface. No central router.
3. **Hierarchical.** Supervisors managing sub-supervisors, implemented as nested subgraphs.

### Where this pattern goes wrong

- **Checkpoints too small.** Only checkpointing conversation turns leaves tool state and memory writes unrecoverable. Full state must serialize.
- **Non-deterministic nodes.** Resume assumes node inputs produce the same state update. Random seeds, wall-clock, external APIs must be captured.
- **Over-use of conditional edges.** A graph with every edge conditional is a state machine that cannot be reasoned about. Prefer linear chains with occasional branches.

## 动手实现

`code/main.py` implements a stdlib stateful graph:

- `State` — a typed dict with `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callable taking state and returning an update dict.
- `StateGraph` — nodes + edges + conditional edges + run + resume.
- `SQLiteCheckpointer` (in-memory fake) — serializes state after every node; `load(session_id)` restores.
- A demo graph: classify -> branch(refund / bug / sales) -> human gate -> send.

Run it:

```
python3 code/main.py
```

The trace shows the first run failing at the human gate, persistence, then resume producing the final output.

## 应用场景

- **LangGraph** — the reference, production-ready. Use `create_react_agent`, `create_supervisor`, or build your own graph.
- **AutoGen v0.4** (Lesson 14) — actor model alternative for high-concurrency scenarios.
- **Claude Agent SDK** (Lesson 17) — managed harness with built-in session store.
- **Custom** — when you need exact control over state shape or checkpointer backend.

## 产出成果

`outputs/skill-state-graph.md` generates a LangGraph-shaped state graph in any target runtime with checkpointing and resume wired in.

## 练习题

1. Add a conditional edge from `classify` to `end` when classification confidence is below a threshold. Resume the run after a human sets `route` manually.
2. Swap the SQLite-like fake for a real SQLite checkpointer. Measure per-step serialization overhead.
3. Implement parallel edges: two nodes run concurrently, merge by a custom reducer. What does immutable state buy here?
4. Read `langgraph-supervisor` reference. Port the toy to `create_supervisor`. Compare the trace shapes.
5. Add streaming: each node yields partial state while it runs. Print the deltas as they arrive.

## 核心术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## 深入阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — the reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store and subagents
