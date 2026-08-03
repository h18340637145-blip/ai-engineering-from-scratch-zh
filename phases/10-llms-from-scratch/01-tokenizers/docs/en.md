# Tokenizer 基础与原理（Tokenizers）

> 语言模型看不见字符——它们只看得到整数 ID。Tokenizer 就是文本到张量之间的翻译官。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 1 Lesson 1（向量与矩阵）
**Time:** ~60 分钟

## 学习目标

- 深入理解字符级、词级与子词级（Subword）分词的优缺点
- 掌握 BPE（Byte Pair Encoding）、WordPiece 与 Unigram 分词算法原理
- 理解字节级回退（Byte-level fallback）如何保证 100% 的字符覆盖率
- 分析分词器在非英语语言与代码表示上的词表效率差异

## 核心问题

神经网络无法直接对字符串进行数学计算。Tokenizer 将文本切分为词块（Tokens），并映射到 fixed-size 的词汇表整数索引中。这一转换的质量直接决定了模型的上下文效率与表示能力。

## 动手构建

参见 `code/main.py`。
