# T5 与 BART：编码器-解码器架构（T5 & BART）

> 将所有 NLP 任务统一为文本到文本（Text-to-Text）变换。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 Lessons 5-7
**Time:** ~60 分钟

## 学习目标

- 深入理解 Encoder-Decoder Transformer 架构在序列生成任务中的优势
- 掌握 T5 的 Text-to-Text 统一框架与相对位置偏置（Relative Position Bias）
- 理解 BART 的去噪自编码（Denoising Autoencoder）预训练目标
- 对比纯编码器（BERT）、纯解码器（GPT）与编码器-解码器（T5）在各类任务上的权衡

## 核心问题

对于翻译、摘要总结和问答等输入输出长度不一且依赖双向强编码的任务，完整的 Encoder-Decoder 结构具有独特的优势。T5 将分类、翻译、总结全部转换为 "prefix: input text -> output text" 格式。
