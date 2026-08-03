# 大作业：从零搭建完整 Transformer

> 综合运用前面所有知识，手写一个包含所有现代特性的 Transformer。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lessons 1-13
**Time:** ~120 分钟

## 学习目标

- 手写整合 RoPE、SwiGLU、Grouped Query Attention (GQA) 与 RMSNorm 的现代 Transformer
- 编写高效且带 KV Cache 的自回归解码生成器
- 训练模型完成一个微型小说的生成或小语种翻译任务
- 导出模型权重并验证与 PyTorch 原生参考实现的一致性

## 核心问题

本课将前述分散的知识点（RoPE、GQA、SwiGLU、RMSNorm、KV Cache）融会贯通，组装出一个符合当下前沿（LLaMA 3 / Qwen 2.5 风格）的标准 Transformer 架构。
