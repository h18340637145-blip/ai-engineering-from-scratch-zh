# 大模型量化技术（Quantization）

> 从 FP16 到 INT8 / INT4：在保留模型威力的同时减少 75% 显存消耗。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 4
**Time:** ~75 分钟

## 学习目标

- 理解数值格式：FP32, FP16, BF16, INT8, INT4 与 NF4 (NormalFloat4)
- 掌握 PTQ（训练后量化）与 QAT（量化感知训练）的差异
- 从零实现对称与非对称量化（Symmetric & Asymmetric Quantization）与 Scale / Zero-point 计算
- 理解 SmoothQuant, GPTQ, AWQ 与 BitsAndBytes 的算法机制

## 动手构建

参见 `code/main.py`。
