# KV Cache 与 FlashAttention 优化

> 从 `O(N^2)` 内存瓶颈到 IO 觉察的硬件级硬件优化。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 2, Lesson 7
**Time:** ~90 分钟

## 学习目标

- 实现自回归生成中的 KV Cache 缓存机制，消除重复键值计算
- 深入理解内存带宽限制（Memory Bandwidth Bound）与 SRAM/HBM 传输瓶颈
- 理解 FlashAttention 的分块（Tiling）与在线 Softmax（Online Softmax）算法原理
- 在 PyTorch 中对比启用与未启用 KV Cache 和 FlashAttention 的推理吞吐量

## 核心问题

自回归推理时，如果不缓存先前的 Key 和 Value，每生成一个新 Token 都要重新计算前面所有 Token 的 K 和 V，生成 N 个 Token 的时间复杂度达到 `O(N^3)`。KV Cache 将其降低到 `O(N^2)`。
然而，注意力矩阵 `Q K^T` 在高上下文下会产生大量的 HBM（显存）读写。FlashAttention 通过在 GPU SRAM 高速缓存中分块计算 Softmax，避免了将大的 N×N 注意力矩阵写入 HBM，实现了数倍的提速。
