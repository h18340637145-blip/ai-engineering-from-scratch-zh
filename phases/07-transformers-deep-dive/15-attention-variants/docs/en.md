# 注意力机制变体：MHA, GQA, MQA 与线性注意力

> 在表达能力与显存带宽之间寻找最佳平衡点。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 3, Lesson 12
**Time:** ~60 分钟

## 学习目标

- 理解 Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA) 的结构设计
- 深入分析 GQA 如何显着降低 KV Cache 显存占用并提高长文本推理吞吐量
- 理解滑动窗口注意力（Sliding Window Attention）与线性注意力（Linear Attention）的原理
- 在 PyTorch 中实现 GQA 维度切分与广播计算

## 核心问题

随着上下文窗口拉长到 128K 以上，传统 MHA 的 KV Cache 会消耗数十 GB 显存。MQA 让所有 Query 头共享一个 KV 头，极大地节省了显存但可能损伤准确率。GQA（如 LLaMA 3 所采用）则在每 G 个 Query 头共享 1 个 KV 头，完美兼顾了速度与质量。
