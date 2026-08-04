# 音频 Transformer 与 Whisper（Audio Transformers & Whisper）

> 将对数梅尔谱图视为图像，使用 Transformer 实现强鲁棒性语音识别。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 8（T5/BART）
**Time:** ~60 分钟

## 学习目标

- 理解语音信号的特征提取：从时域波形到对数梅尔谱图（Log-Mel Spectrogram）
- 掌握 Whisper 架构：基于 Encoder-Decoder 的多任务语音 Transformer
- 理解音频掩码、时间下采样与转录文本生成的同步机制
- 使用 Whisper 完成语音转文字与跨语言翻译流程

## 核心问题

语音信号是高维连续一维序列。Whisper 通过将音频处理为二维梅尔频谱图，并结合海量弱监督数据，实现了对口音、背景噪音和专业术语极强的鲁棒性。
