# 模型智能路由：根据复杂性与成本动态派发请求

> 设计级联路由模型：简单查询派发至轻量模型，复杂推理自动路由至顶级模型，兼顾质量与成本。

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 Lesson 01
**Time:** ~60 分钟

## 学习目标

- Explain model cascading: cheap-first with confidence check, escalate on low confidence.
- Enumerate the four routing signals (task classification, prompt length, embedding similarity to known-hard set, self-confidence from first-pass).
- Compute expected blended cost at target routing split and quality loss tolerance.
- Name the drift-monitoring metric (online quality gate) that catches cheap-model creep.

## 问题切入

Your service costs $80k/month on GPT-5. Your analytics show 70% of queries are simple: "what time is it in Paris?" "rephrase this sentence." A Haiku-class model handles those perfectly at 3% of the cost. 30% need GPT-5's reasoning — coding, math, multi-step planning.

If you route the 70% to cheap and 30% to expensive, your bill drops ~65% at the same product quality. This is routing. The trick is building the broker without regressing quality.

## 核心概念

### Four routing signals

1. **Task classification**: simple/complex/codegen/math/chat. Can be a rules-based classifier, a small LLM (Haiku-class at $0.25/M), or embedding similarity to labeled buckets. Output: route = cheap / balanced / frontier.

2. **Prompt length**: prompts >4K tokens often need frontier for coherence. Prompts <500 tokens usually don't.

3. **Embedding similarity to known-hard set**: if the query is close (cosine > 0.88) to a known-hard bucket, escalate to frontier directly.

4. **Self-confidence from first-pass**: send to cheap; if model's log-probs show low confidence OR it refuses OR outputs hedging language, retry on frontier. Adds P95 latency on ~10% of traffic but saves 50%+ on the other 90%.

### Three patterns

**Pre-route** (classifier up front): ~5-10ms latency added; fastest overall.

**Cascade** (cheap-first, escalate on low confidence): ~1.2x median latency (cheap run plus verify), ~2x on escalated. Best quality floor.

**Ensemble route** (run cheap and frontier in parallel for a sample, reward-model pick): highest quality, highest cost; use only for critical A/B.

### Implementation

AI gateways (Phase 17 · 19) expose routing. LiteLLM has `router` config with fallback and cost-routing. Portkey has guards + routing. Kong AI Gateway has plugin-based routing. OpenRouter's model marketplace exposes a recommendation API.

Open-source: RouteLLM (LMSYS), Not Diamond (commercial), Prompt Mule.

### The 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

Most of the improvement is serving efficiency — the core lessons in Phase 17 · 04-09 turned into provider-side cost drops. Routing lets you capture those gains at the app layer instead of waiting for all your users to migrate to the cheap tier.

### Drift is the real risk

Your route sends 40% to the cheap model. Over six months, the task distribution shifts (users get more sophisticated, ask longer questions). The router doesn't notice because its classifier was trained on Q1 data. Quality drops silently. Nobody complains loud enough. You find out in a competitor benchmark you lost.

Gate routes by online quality metrics:

- User thumbs-up / thumbs-down per route.
- Automated LLM-judge on a held-out sample (5%) per route.
- Escalation rate: if cascade is kicking up-route >30%, the cheap model is being over-routed.
- Refusal rate per route.

### Numbers you should remember

- 2026 routing savings at iso-quality: 20-60% case studies.
- LLM price drop 2022-2026: ~10x per year aggregate.
- GPT-4-level 2022 vs 2026: ~$20/M → ~$0.40/M.
- Cascade latency impact: ~1.2x median, ~2x escalated (~10% of traffic).

## 应用场景

`code/main.py` simulates pre-route, cascade, and ensemble on a mixed workload. Reports blended cost, quality loss, and escalation rate.

## 产出成果

This lesson produces `outputs/skill-router-plan.md`. Given workload and quality budget, picks a routing pattern and signals.

## 练习题

1. Run `code/main.py`. At what accuracy floor does cascade beat pre-route?
2. Your user base is 30% enterprise (complex queries), 70% free tier (simple). Design the routing split. What online metric gates it?
3. A route drops quality by 2% but saves 40%. Is that a ship? Depends on product — argue both.
4. Implement a confidence check using logprobs from OpenAI / Anthropic APIs. What's the threshold you start with?
5. Over six months, escalation rate climbs from 8% to 22%. Diagnose three causes and the fix for each.

## 核心术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## 深入阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
