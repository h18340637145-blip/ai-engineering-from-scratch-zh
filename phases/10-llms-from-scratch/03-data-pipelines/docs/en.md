# 大模型预训练数据管道（Pre-training Data Pipelines）

> 垃圾进，垃圾出：构建百亿 Token 级别的清洗、去重、过滤与分块数据流水线。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 1-2
**Time:** ~75 分钟

## 学习目标

- 掌握预训练数据的清洗流程：HTML 提取、语言识别与启发式质量过滤
- 实现基于 MinHash 和 LSH（局部敏感哈希）的高效文本去重
- 理解字节级与文档级拼接（Document Packing）与滑动窗口打包策略
- 构建高吞吐量的二进制数据集（如 HDF5 / Memmap）加载器

## 动手构建

参见 `code/main.py`。
