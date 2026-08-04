# Vision Transformer (ViT)

> 一张图片胜过 16x16 个 Words：将图像切分为 Patch，像处理文本一样处理视觉信号。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 5（完整 Transformer 架构）
**Time:** ~60 分钟

## 学习目标

- 理解图像 Patch 化（Patch Extraction）与线性投影过程
- 实现带有 [CLS] 标记和 1D 位置编码的 ViT 模型
- 分析 ViT 缺乏 CNN 归纳偏置（如平移不变性）的原因及其对数据量的依赖
- 完成一个基于 PyTorch 的 ViT 图像分类任务

## 核心问题

以往计算机视觉由 CNN 主导。ViT（Dosovitskiy et al., 2020）证明，只需将图像切割为 `16x16` 的图像块展开成向量，直接送入标准 Transformer 编码器，就能在海量数据上超越顶级 CNN。
