# 从零构建 BPE Tokenizer（Building a Tokenizer）

> 手写一个完整的 Byte-Level BPE 分词器：从频次统计、对合并到编码与解码。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 1（Tokenizers 基础）
**Time:** ~90 分钟

## 学习目标

- 仅使用 Python 标准库实现 Byte-pair encoding (BPE) 训练算法
- 编写高效的对合并（Pair Merging）与编码 / 解码逻辑
- 处理特殊标记（Special Tokens，如 `<|endoftext|>`）的正则预切分与隔离
- 将自制 Tokenizer 的编码结果与 `tiktoken` / HuggingFace `tokenizers` 进行对齐验证

## 动手构建

参见 `code/main.py`。
