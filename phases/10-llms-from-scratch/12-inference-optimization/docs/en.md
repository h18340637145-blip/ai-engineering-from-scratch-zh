# 推理性能优化（Inference Optimization）

> 提升 Token 生成吞吐量：连续批处理（Continuous Batching）、PagedAttention 与 TensorRT-LLM。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 12, Phase 10 Lesson 11
**Time:** ~75 分钟

## 学习目标

- 掌握 vLLM 架构核心：PagedAttention 如何解决显存碎片化问题
- 理解 Continuous Batching（连续批处理 / 动态批处理）提高 GPU 利用率的机制
- 比较 Chunked Prefill 与 Decode 阶段算力特性的差异
- 使用 Benchmarking 工具评估 TPS（Tokens Per Second）与首字延迟（TTFT）

## 动手构建

参见 `code/main.py`。
