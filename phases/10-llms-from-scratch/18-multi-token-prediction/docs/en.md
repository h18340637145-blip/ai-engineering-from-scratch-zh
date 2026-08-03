# 多 Token 并行预测（Multi-Token Prediction - MTP）

> 从单 Token 预测到多 Token 并行生成：DeepSeek-V3 架构的核心突破。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 4, Phase 10 Lesson 20
**Time:** ~60 分钟

## 学习目标

- 理解训练阶段同时预测未来 N 个 Token 的多头 Loss 架构
- 掌握在推理时直接将 MTP 头作为免费投机采样 Draft Model 的方法

## 动手构建

参见 `code/main.py`。
