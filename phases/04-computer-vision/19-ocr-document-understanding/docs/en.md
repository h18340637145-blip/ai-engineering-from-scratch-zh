# OCR 与文档理解

> OCR 是一个三阶段的流程 —— 检测文本框、识别字符、然后进行布局。每个现代的 OCR 系统都会重新排序这些阶段或将其合并。

**类型:** 学习 + 使用
**语言:** Python
**先决条件:** 第 4 阶段第 06 课（检测），第 7 阶段第 02 课（自注意力）
**时间:** ~45 分钟

## 学习目标

- 追踪经典 OCR 流程（检测 -> 识别 -> 布局）以及现代端到端替代方案（Donut, Qwen-VL-OCR）
- 为序列到序列 OCR 训练实现 CTC（连接时序分类）损失
- 使用 PaddleOCR 或 EasyOCR 进行生产文档解析，无需训练
- 区分 OCR、布局解析和文档理解 —— 并根据任务选择合适的工具

## 问题

包含大量文本的图像无处不在：收据、发票、身份证、扫描书籍、表格、白板、标志、截图。从中提取结构化数据 —— 而不仅仅是字符，而是“这是总金额” —— 是应用视觉领域中价值最高的问题之一。

该领域分为三个技能层次：

1. **OCR 本身**：将像素转换为文本。
2. **布局解析**：将 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从布局中提取结构化字段（“invoice_total = $42.50”）。

每一层都有经典和现代的方法，而“我想从图像中得到文本”与“我需要从这张收据中得到总金额”之间的差距，比大多数团队意识到的要大得多。

## 概念

### 经典流程

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```- **文本检测** 生成每行或每个单词的四边形。
- **识别** 将每个区域裁剪为固定高度，运行一个 CNN + BiLSTM + CTC 来生成字符序列。
- **布局** 重建阅读顺序（拉丁文为从上到下、从左到右；阿拉伯语、日文则不同）。

### 一段话中的 CTC

OCR 识别从固定长度的特征图中生成一个可变长度的序列。CTC（Graves 等，2006）允许你在没有字符级对齐的情况下训练这个模型。模型在每个时间步输出（词汇表 + 空格）的分布；CTC 损失对所有在合并重复项并删除空格后还原为目标文本的对齐方式进行边际化处理。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```CTC 是 CRNN 在 2015 年取得成功并在 2026 年仍然训练大多数生产 OCR 模型的原因。

### 现代端到端模型

- **Donut** (Kim 等人，2022) — 一个 ViT 编码器 + 一个文本解码器；读取图像并直接输出 JSON。没有文本检测器，没有布局模块。
- **TrOCR** — ViT + 变换器解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** — 完整的视觉-语言模型，针对 OCR 任务进行了微调；在 2026 年复杂文档中准确率最佳。
- **PaddleOCR** — 经典的 DB + CRNN 管道，封装在成熟的生产包中；仍然是开源的主力工具。

端到端模型需要更多的数据和计算资源，但可以跳过多阶段流水线的误差累积。

### 布局解析

对于结构化文档，运行一个布局检测器（LayoutLMv3, DocLayNet），对每个区域进行标记：标题、段落、图表、表格、脚注。阅读顺序随后变成“按布局顺序遍历区域，然后连接”。

对于表格单，使用 **键值提取** 模型（Donut 用于视觉丰富的文档，LayoutLMv3 用于普通扫描件）。它们接受图像 + 检测到的文本 + 位置，并预测结构化的键值对。

### 评估指标

- **字符错误率 (CER)** — 编辑距离 / 参考长度。数值越低越好。生产目标：干净扫描件上 < 2%。
- **词错误率 (WER)** — 在词级别上相同。
- **结构化字段的 F1** — 用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 的编辑距离** — 用于端到端文档解析；Donut 论文引入了标准化树编辑距离。

## 构建它

### 第一步：CTC 损失 + 贪婪解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效的 CuDNN 实现。贪婪解码器比束搜索更简单，通常其字符错误率（CER）与后者相差在 1% 以内。

### 步骤 2：Tiny CRNN 识别器

用于行 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN将高度最大池化为1）。宽度是CTC的时间维度。

### 步骤3：合成OCR

为端到端的烟雾测试生成黑白数字字符串。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

一个真实的OCR数据集会添加字体、噪声、旋转、模糊和颜色。上面的流程是相同的。

### 步骤4：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

损失应该在 200 步内从约 3 降到约 0.2，这是在这些简单的合成数据上。

## 使用它

三种生产路径：

- **PaddleOCR** — 成熟、快速、多语言。一行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — Python 原生、多语言、基于 PyTorch。
- **Tesseract** — 经典；当模型难以处理时，仍适用于旧的扫描文档。

对于端到端的文档解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于具有可重复结构的收据、发票和表单，对 Donut 进行微调。对于任意文档或需要推理的 OCR，当前默认使用类似 Qwen-VL-OCR 的 VLM。

## 发布它

本课将产出：

- `outputs/prompt-ocr-stack-picker.md` — 一个根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR 的提示。
- `outputs/skill-ctc-decoder.md` — 一个从头编写贪婪和集束搜索 CTC 解码器的技能，包括长度归一化。

## 练习

1. **(简单)** 在 500 步上训练 TinyCRNN 以处理 5 位随机数字字符串。在保留集上报告 CER。
2. **(中等)** 用集束搜索（beam_width=5）替代贪婪解码。报告 CER 的变化。集束搜索在哪些输入上表现更好？
3. **(困难)** 使用 PaddleOCR 处理一组 20 张收据，提取条目信息，并针对 {item_name, price} 对计算与人工标注真实值的 F1 值。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------|
| OCR | “从像素中提取文本” | 将图像区域转换为字符序列 |
| CTC | “无需对齐的损失” | 一种无需每时间步标签即可训练序列模型的损失函数；对对齐进行边缘化 |
| CRNN | “经典 OCR 模型” | 卷积特征提取器 + BiLSTM + CTC；2015 年的基准模型，至今仍在生产中使用 |
| Donut | “端到端 OCR” | ViT 编码器 + 文本解码器；直接从图像中输出 JSON |
| 布局解析 | “查找区域” | 在文档中检测并标记标题/表格/图表/段落区域 |
| 阅读顺序 | “文本序列” | 将识别出的区域按句子顺序排列；对拉丁文简单，对混合布局复杂 |
| CER / WER | “错误率” | 在字符或单词粒度下，基于编辑距离和参考长度的度量 |
| VLM-OCR | “能阅读的 LLM” | 为 OCR 任务训练或提示的视觉语言模型；目前在复杂文档上最先进的模型 |

## 进一步阅读

- [CRNN (Shi 等, 2015)](https://arxiv.org/abs/1507.05717) — 最初的 CNN+RNN+CTC 架构
- [CTC (Graves 等, 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 最初的 CTC 论文；算法思想密集
- [Donut (Kim 等, 2022)](https://arxiv.org/abs/2111.15664) — 无需 OCR 的文档理解转换器
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源的生产 OCR 堆栈
