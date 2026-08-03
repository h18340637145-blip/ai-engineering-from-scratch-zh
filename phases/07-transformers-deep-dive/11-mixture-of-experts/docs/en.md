# 混合专家模型（Mixture of Experts, MoE）

> 激活小部分专家参数，享受万亿参数模型的容量。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 5（完整 Transformer 架构）
**Time:** ~75 分钟

## 学习目标

- 理解稀疏激活（Sparse Activation）与 Router/Gating 网络的工作原理
- 实现 Top-k 门控机制与 Load Balancing Loss（负载均衡损失）
- 理解 Expert Capacity（专家容量）与 Token 丢弃策略
- 分析 DeepSeek-V3 / Mixtral 等现代 MoE 架构的并行计算与内存权衡

## 核心问题

随着模型参数规模剧增，密集型（Dense）模型在每次前向传播时需要计算所有参数，推理与训练成本极高。MoE 将 FFN 层替换为多个独立的“专家”网络，并通过 Router 为每个 Token 选择最优的 Top-2 专家，从而在保持计算 FLOPs 较低的同时极大扩展模型容量。
