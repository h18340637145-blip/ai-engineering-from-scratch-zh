# DeepSeek-V3 架构全景解析（DeepSeek-V3 Walkthrough）

> MLA (Multi-head Latent Attention)、DeepSeekMoE 与无辅助损失负载均衡。

**Type:** Learn / Reference
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 11, Phase 10 Lessons 18-19
**Time:** ~90 分钟

## 学习目标

- 深入剖析多头隐空间注意力（MLA）的低秩压缩与 RoPE 解耦机制
- 掌握 DeepSeekMoE 的细粒度专家切分与无辅助损失（Auxiliary-loss-free）负载均衡策略
- 总结前沿 LLM 架构演进的关键趋势

## 动手构建

参见 `code/main.py`。
