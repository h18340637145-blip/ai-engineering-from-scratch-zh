# 开放词汇视觉 — CLIP

> 将图像编码器和文本编码器一起训练，使得匹配的（图像，描述）对在共享空间中落在同一个点上。这就是全部的技巧。

**类型:** 构建 + 使用
**语言:** Python
**先决条件:** 第四阶段第14课（ViT），第四阶段第17课（自监督）
**时间:** ~45分钟

## 学习目标

- 解释CLIP的双塔架构和对比训练目标
- 使用预训练的CLIP（或SigLIP）进行零样本分类，而无需任何任务特定的训练
- 从零开始实现零样本分类：编码类别提示，计算余弦相似度，取最大值
- 区分CLIP、SigLIP、OpenCLIP和LLaVA/LLaMA-vision模型 —— 在2026年各自的应用场景

## 问题

传统分类器是封闭词汇的：一个1000类的ImageNet模型只能预测1000个标签。每个新类别都需要标记数据和重新训练头部。

CLIP（Radford等，OpenAI 2021）表明，使用从网络上抓取的4亿个（图像，描述）对进行训练，可以生成一个模型，它可以在推理时对任何用自然语言描述的类别集合进行分类。你只需写一个句子，就可以给出一个新的类别。

这种能力 —— 零样本迁移 —— 就是为什么现代视觉系统都从CLIP家族的检查点开始。检测（Grounding DINO，OWL-ViT）、分割（CLIPSeg，SAM）、检索、内容审核、VLM和文本到图像生成都基于CLIP风格的联合嵌入。

## 概念

### 双塔结构```mermaid
flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```两个编码器都以线性投影结束，投影到相同的嵌入维度（CLIP-B/32 为 512，CLIP-L/14 为 1024）。进行 L2 归一化并计算余弦相似度。

### 目标

给定一个包含 N 个（图像，字幕）对的批次，构建一个 NxN 的相似度矩阵。训练两个编码器，使对角线（匹配对）具有高相似度，而非对角线（不匹配）具有低相似度。```
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```对称，因为图像到文本和文本到图像的检索都应该有效。`tau`（温度）通常被学习为一个标量参数，初始化为 0.07。

### SigLIP：更好的损失函数

SigLIP（Zhai 等，2023）将 softmax 替换为每对的 sigmoid：```
loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise
```每对损失移除了CLIP所需的批量级归一化。SigLIP在小批量尺寸下训练效果更好，在相同数据量下可匹配或超越CLIP。

### 零样本分类

给定一个训练好的CLIP：

1. 对于每个类别，构造一个提示：“一张{类}的照片”。
2. 使用文本编码器对所有类别提示进行编码 -> `T` 形状 (C, d)。
3. 对测试图像进行编码 -> `I` 形状 (1, d)。
4. 相似度 = `I @ T.T` 形状 (1, C)。
5. 取最大值 -> 预测类别。

提示工程很重要。OpenAI为ImageNet发布了80个提示模板（“一张{类}的照片”，“一张模糊的{类}的照片”，“一张{类}的草图”，...）。对每个类别的所有模板的嵌入进行平均，可额外获得1-3%的top-1准确率提升。

### 2026年CLIP风格模型的应用场景

- **零样本分类** —— 直接使用。
- **图像检索** —— 一次性对所有图像进行编码，在推理时嵌入查询。
- **文本条件检测** —— Grounding DINO、OWL-ViT将CLIP文本塔封装在检测器周围。
- **文本条件分割** —— CLIPSeg；SAM通过CLIP使用文本提示输入。
- **视觉-语言模型（VLMs）** —— LLaVA、Qwen-VL、InternVL将CLIP家族的视觉编码器连接到大语言模型（LLM）中。
- **文本到图像生成** —— Stable Diffusion、DALL-E 3基于CLIP文本嵌入进行条件生成。

一旦拥有一个共享的嵌入空间，每一个视觉+语言任务都变成了距离计算。

## 构建它

### 第一步：一个微小的双塔模型

真实的CLIP是ViT + transformer。为了本课程的需要，这两个塔是基于预提取特征的小型MLP，这样在CPU上也能看到训练信号。```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```两个投影，共享维度输出，学习温度。与真实的 CLIP API 形状相同。

### 步骤 2：对比损失```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```对称。更高的 logit_scale = 更尖锐的 softmax = 更自信但存在不稳定的風險。

### 步骤 3：零样本分类器```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```每一步占一行。这是在生产 CLIP 检查点上使用的精确零样本流程。

### 步骤 4：合理性检查```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```损失应该接近 `log(N) = log(8) = 2.08`，对于一个随机初始化的模型 —— 在尚未学习任何结构时的对称交叉熵目标。

## 使用方法

OpenCLIP 是 2026 年社区的默认选择：

 /no_think

<>

损失应该接近 `log(N) = log(8) = 2.08`，对于一个随机初始化的模型 —— 在尚未学习任何结构时的对称交叉熵目标。

## 使用方法

OpenCLIP 是 2026 年社区的默认选择：

 /no_think```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```SigLIP 更新，小规模训练效果更好，并且是新工作的首选：`google/siglip-base-patch16-224`。Hugging Face 提供两者。

## 发布它

本课程将产出以下内容：

- `outputs/prompt-zero-shot-class-picker.md` — 一个提示，可以根据给定的类别列表和领域为零样本 CLIP 设计类别模板。
- `outputs/skill-image-text-retriever.md` — 一种技能，可以使用任何 CLIP 检查点构建图像嵌入索引，支持文本查询和图像查询。

## 练习

1. **(简单)** 使用预训练的 OpenCLIP ViT-B/32，并使用 80 个模板提示集在 CIFAR-10 上进行零样本分类。报告 top-1 准确率；它应该在 85-90% 左右。
2. **(中等)** 在相同的 CIFAR-10 任务中比较单模板（"a photo of a {}"）与 80 个模板的平均嵌入。量化差距并解释为什么模板有帮助。
3. **(困难)** 构建一个零样本图像检索索引：使用 CLIP 对 1,000 张图像进行嵌入，构建一个 FAISS 索引，并使用自然语言描述进行查询。报告你手动编写的 20 个查询的检索 recall@5。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|-----------|
| Two-tower | "双编码器" | 分开的图像和文本编码器，最后通过共享维度的投影头连接 |
| Zero-shot | "没有特定任务训练" | 推理时仅根据文本描述的类别进行分类；不接触标签 |
| Temperature / logit_scale | "tau" | 在 softmax 之前对相似性矩阵进行缩放的学习标量 |
| Prompt template | "A photo of a {}" | 包裹类别名称的自然语言；平均多个模板可以提高零样本准确率 |
| CLIP | "图像+文本模型" | 2021 年 OpenAI 模型；2026 年领域内的标准词汇 |
| SigLIP | "Sigmoid CLIP" | 将 softmax 替换为每对的 sigmoid；在小批量训练中表现更好 |
| OpenCLIP | "开源复现" | 在 LAION 上训练的社区 CLIP 变体；开源流程的默认选择 |
| VLM | "视觉-语言模型" | CLIP 家族的编码器加上一个 LLM，训练用于回答关于图像的问题 |

## 进一步阅读

- [CLIP: 从自然语言监督中学习可迁移的视觉模型 (Radford 等, 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: 用于语言-图像预训练的 Sigmoid 损失 (Zhai 等, 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 社区代码库
- [DINOv2 vs CLIP vs MAE: 特征对比](https://huggingface.co/blog/dinov2) — Hugging Face 的指南，包含并排使用案例
