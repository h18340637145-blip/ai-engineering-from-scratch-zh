# 综合项目 02——面向代码库的 RAG（跨仓库语义搜索）

> 2026 年，所有成熟的工程组织都在运行能够理解语义而非只匹配字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图、Aider 的 repomap，以及 Pinterest 的内部 MCP，都采用相同形态：摄取多个仓库，使用 tree-sitter 解析，对函数级和类级分块做嵌入，执行混合搜索与重排，并给出带引用的答案。本综合项目要求你构建这样一个系统：处理 10 个仓库中的 200 万行代码，并能在每次 Git 推送后可靠完成增量重建索引。

**类型：** 综合项目
**语言：** Python（摄取）、TypeScript (API + UI)
**前置课程：** 阶段 5（NLP 基础）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿编码智能体都配备了代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 Token 上下文有所帮助，但不能消除排序检索的必要性。对原始分块执行朴素余弦搜索时，生成代码、单体仓库中的重复内容，以及很少被导入的长尾符号都会污染结果。生产级方案是在 AST 感知分块上执行稠密检索与 BM25 相结合的混合搜索，并使用重排器和符号引用图提供支持。

你需要通过索引一组真实仓库来学习，而不是只处理一个教程仓库；同时衡量 MRR@10、引用忠实度和增量新鲜度。故障模式主要来自基础设施：包含 10 万个文件的单体仓库、一次改动一半文件的推送，以及必须跨越 4 个仓库才能正确回答的查询。

## 核心概念

AST 感知摄取流水线使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界而不是固定 Token 窗口处分块。每个分块都有 3 种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及简短的自然语言摘要。摘要增加了第 3 种可检索模态：用户可能询问「X 如何获得授权」，摘要中会出现「authz」，即使代码里只有 `check_permission`。

检索采用混合方式。一个查询会同时触发稠密搜索和 BM25 搜索，合并 top-k 后，将并集交给交叉编码器重排器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排后的列表进入长上下文综合器（启用提示词缓存的 Claude Sonnet 4.7，或自托管 Llama 3.3 70B），并要求每项声明都引用文件和行号范围。后置过滤器会拒绝没有引用的答案。

增量新鲜度是基础设施问题。Git 推送会触发 Diff，判断哪些文件和符号发生变化。只有受影响的分块重新嵌入，受影响的跨文件符号边（导入、方法调用）也会重新计算。这样无需每次提交都重新处理 200 万行代码，索引仍能保持一致。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack

- Parsing: tree-sitter with 17 language grammars (Python, TS, Rust, Go, Java, C++, etc.)
- Dense embeddings: Voyage-code-3 (hosted) or nomic-embed-code-v1.5 (self-host), bge-code-v1 fallback
- Sparse index: Tantivy (Rust) with BM25F, field-weighted on symbol name vs body
- Vector DB: Qdrant 1.12 with hybrid search, or pgvector + pgvectorscale for teams under 50M vectors
- Chunk summary model: Claude Haiku 4.5 or Gemini 2.5 Flash, prompt-cached
- Re-ranker: Cohere rerank-3 or bge-reranker-v2-gemma-2b self-hosted
- Orchestration: LlamaIndex Workflows for ingestion, LangGraph for query agent
- Synthesizer: Claude Sonnet 4.7 (1M context) with prompt caching
- Symbol graph: Neo4j (managed) or kuzu (embedded) for import and call edges
- Observability: Langfuse spans per retrieval + synthesis step

## Build It

1. **Ingestion walker.** Iterate git history on every push hook. Collect changed files. For each file, parse with tree-sitter, extract function and class nodes with their full source span. Emit chunk records `{repo, path, start_line, end_line, symbol, body}`.

2. **Chunk summarizer.** Batch chunks into Haiku 4.5 calls with prompt caching on the system preamble. Prompt: "Summarize this function in one sentence, naming its public contract and side effects." Store summary alongside the chunk.

3. **Embedding pool.** Two parallel queues: dense (Voyage-code-3 batch 128) and summary (same model, but on the summary string). Write vectors to Qdrant with payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **BM25 index.** Field-weighted Tantivy index: symbol name weight 4, symbol body weight 1, summary weight 2. Enables "find the function named X" queries alongside "find the function that does X".

5. **Symbol graph.** For each chunk, record edges: imports (this file uses symbol Y from repo Z), calls (this function calls method M on class C), inheritance. Store in kuzu. Used at query time to expand retrieval across repo boundaries.

6. **Query agent.** LangGraph with three nodes. `retrieve` fires dense + BM25 in parallel, deduplicates by (repo, path, symbol). `rerank` runs the cross-encoder on top-50 and keeps top-10. `synth` calls Claude Sonnet 4.7 with the reranked chunks in context, caches the system prompt, requires file:line citations.

7. **Citation enforcement.** Parse the model output; any claim without a `(repo/path:start-end)` anchor gets flagged for re-ask or dropped. Return cited-only answer to the user.

8. **Incremental re-index.** On each webhook, compute the symbol-level diff. Only re-embed chunks whose text changed. Recompute symbol edges for chunks whose imports changed. Measure: a 50-file push re-indexed in under 60 seconds for a 2M-LOC fleet.

9. **Eval.** Label 100 cross-repo questions with gold file:line answers. Measure MRR@10, nDCG@10, citation faithfulness (fraction of claims with verifiable anchors), and p50/p99 latency.

## Use It

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

Deliverable skill `outputs/skill-codebase-rag.md`. Given a corpus of repos, it stands up the ingestion pipeline, the hybrid index, and the query agent, and returns a cited answer for any cross-repo question. Rubric:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Retrieval quality | MRR@10 and nDCG@10 on a 100-question held-out set |
| 20 | Citation faithfulness | Fraction of answer claims with verifiable file:line anchors |
| 20 | Latency and scale | p95 query latency at 10k QPS on the indexed corpus size |
| 20 | Incremental indexing correctness | Time from git push to searchable on a 50-file commit |
| 15 | UX and answer formatting | Citation clickability, snippet previews, follow-up affordance |
| **100** | | |

## Exercises

1. Swap Voyage-code-3 for nomic-embed-code self-hosted. Measure the MRR@10 delta. Report whether the gap closes with re-ranking enabled.

2. Inject 20% generated code (LLM-produced boilerplate) into the corpus and re-evaluate. Observe retrieval poisoning. Add a "generated" flag to the payload and down-weight those hits.

3. Benchmark Qdrant hybrid search vs pgvector + pgvectorscale at your corpus size. Report p99 at batch size 1.

4. Add a sampling-based drift check: weekly, rerun the 100-question eval. Alert on MRR@10 drop > 5%.

5. Extend to cross-language symbol resolution: a Python function that calls a Go service over gRPC. Use the symbol graph to link them.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AST-aware chunking | "Function-level splits" | Cutting code at tree-sitter node boundaries instead of fixed token windows |
| Hybrid search | "Dense + sparse" | Run BM25 and vector search in parallel, merge top-k, rerank |
| Cross-encoder rerank | "Second-stage rank" | Model that scores each (query, candidate) pair together, more accurate than cosine |
| Prompt caching | "Cached system prompt" | 2026 Claude / OpenAI feature that discounts repeat prefix tokens up to 90% |
| Symbol graph | "Code graph" | Edges for imports, calls, inheritance across files and repos |
| Citation faithfulness | "Grounded answer rate" | Fraction of claims a user can verify by clicking the anchor and reading the referenced span |
| Incremental re-index | "Push-to-search time" | Wall-clock from git push to the changed symbols being queryable |

## Further Reading

- [Sourcegraph Amp](https://ampcode.com) — production cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — the reference deep-dive for this capstone
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — commercial symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference
