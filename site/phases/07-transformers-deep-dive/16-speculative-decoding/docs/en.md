# 投机采样/投机解码（Speculative Decoding）

> 用小模型提建议，大模型做校验：打破自回归推理的串行瓶颈。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 7, Lesson 12
**Time:** ~60 分钟

## 学习目标

- 理解投机解码（Speculative Decoding）的核心思想：用小的 Draft 模型生成候选 Token，大模型并行验证
- 掌握基于拒绝采样（Rejection Sampling）的概率修正算法，确保输出分布与纯大模型完全一致
- 实现一个完整的投机解码仿真器，并计算加速比（Acceptance Rate 与 Speedup）
- 理解 Medusa / EAGLE 等单模型投机采样的最新进展

## 核心问题

自回归生成是内存带宽受限的：每次生成 1 个 Token，都要将整个大模型的几十 GB 权重从显存读取到芯片中。投机解码利用小模型（草稿模型）快速串行生成 K 个 Token，然后大模型只需要运行一次前向传播即可同时验证这 K 个 Token，将显存读取次数降低数倍。
