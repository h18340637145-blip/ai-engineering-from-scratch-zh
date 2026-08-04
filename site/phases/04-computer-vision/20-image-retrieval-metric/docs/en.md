# 图像检索与度量学习

> 检索系统通过嵌入空间中的距离对候选对象进行排序。度量学习是塑造这个空间的学科，使得距离具有你想要的含义。

**类型:** 构建
**语言:** Python
**先决条件:** 第四阶段第14课（ViT），第四阶段第18课（CLIP）
**时间:** ~45分钟

## 学习目标

- 解释三元组、对比和基于代理的度量学习损失，并为给定的数据集选择合适的损失
- 正确实现L2归一化和余弦相似度，并审计“相同项目”和“相同类别”检索之间的差异
- 构建一个FAISS索引，通过文本和图像进行查询，并对保留查询集报告recall@K
- 使用DINOv2、CLIP和SigLIP作为现成的嵌入主干，并知道何时每种方法更优

## 问题

检索在生产视觉系统中无处不在：重复检测、反向图像搜索、视觉搜索（“查找相似产品”）、人脸再识别、用于监控的人再识别、电子商务中的实例级匹配。产品的问题始终相同：“给定这个查询图像，对我的目录进行排序。”

两个设计决策决定了整个系统的结构。嵌入——什么模型生成向量。索引——如何在大规模下查找最近邻。两者在2026年都是商品（DINOv2用于嵌入，FAISS用于索引），这提高了标准：困难的部分是为你的应用定义*什么才算相似*，然后塑造嵌入空间使得距离匹配。

这种塑造就是度量学习。它是一个小但高杠杆的学科。

## 概念

### 一瞥检索```mermaid
flowchart LR
    Q["Query image<br/>or text"] --> ENC["Encoder"]
    ENC --> EMB["Query embedding"]
    EMB --> IDX["FAISS index"]
    CAT["Catalogue images"] --> ENC2["Encoder (same)"] --> IDX_BUILD["Build index"]
    IDX_BUILD --> IDX
    IDX --> RANK["Top-k nearest<br/>by cosine / L2"]
    RANK --> OUT["Ranked results"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style IDX fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```### 四种损失函数家族

| 损失函数 | 需要 | 优点 | 缺点 |
|---------|------|------|-----|
| **对比损失（Contrastive）** | (anchor, positive) + negatives | 简单，适用于任何配对标签 | 没有大量负样本时收敛较慢 |
| **三元组损失（Triplet）** | (anchor, positive, negative) | 直观；可以直接控制边距 | 难三元组挖掘成本较高 |
| **NT-Xent / InfoNCE** | 配对 + 批量挖掘负样本 | 可扩展到大批次 | 需要大批次或动量队列 |
| **基于代理（ProxyNCA）** | 仅需类别标签 | 快速、稳定、无需挖掘 | 在小数据集上可能过度拟合代理 |

对于大多数生产使用场景，建议从预训练的骨干网络开始，仅当现成的嵌入在测试集上表现不佳时，再添加度量学习的微调。

### 正式定义三元组损失```
L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
```将锚点 `a` 拉近至正样本 `p`，推远至负样本 `n`，并使用一个 `margin` 来确保间隔。三图结构可以推广到任何相似性排序。

挖掘重要性：容易的三元组（`n` 已经远离 `a`）贡献零损失；只有困难的三元组才能训练网络。半困难挖掘（`n` 比 `p` 远，但仍在边界内）是 2016 年 FaceNet 的配方，至今仍占主导地位。

### 余弦相似性 vs L2 距离

两种度量方式，两种惯例：

- **余弦相似性**：向量之间的夹角。需要 L2 归一化的嵌入。
- **L2 距离**：欧几里得距离。适用于原始或归一化的嵌入，但通常与 L2 归一化 + 平方 L2 一起使用。

对于大多数现代网络，两者是等价的：当 `||a - b||^2 = 2 - 2 cos(a, b)` 时 `||a|| = ||b|| = 1`。选择与你的嵌入训练相匹配的惯例；混合使用它们会静默地改变“最近”的含义。

### Recall@K

标准的检索指标：```
recall@K = fraction of queries where at least one correct match is in the top K results
```并排报告 recall@1、@5、@10 的结果。如果 recall@10 高于 0.95，但 recall@1 低于 0.5，这意味着嵌入空间结构正确，但排序存在噪声 —— 尝试更长的微调过程或添加重新排序步骤。

对于重复检测，precision@K 更加重要，因为每一个误报都会成为用户可见的错误。对于视觉搜索，recall@K 是产品信号。

### 用一段话介绍 FAISS

Facebook AI Similarity Search。最近邻搜索的默认库。三种索引选择：

- `IndexFlatIP` / `IndexFlatL2` — 暴力搜索，精确，无需训练。适用于最多约 1M 个向量。
- `IndexIVFFlat` — 将数据划分为 K 个单元格，仅搜索最近的几个单元格。近似，快速，需要训练数据。
- `IndexHNSW` — 基于图的，对于大量查询最快，但索引大小较大。

对于 100k 个向量，你可能希望使用 `IndexFlatIP` 进行余弦相似度计算。对于 10M 个向量，你希望使用 `IndexIVFFlat`。对于 100M 个向量以上，可结合乘积量化（`IndexIVFPQ`）。

### 实例级与类别级检索

两个截然不同的问题，却有着相同的名称：

- **类别级** —— “在目录中找到猫。” 类条件相似性；现成的 CLIP / DINOv2 嵌入效果很好。
- **实例级** —— “在目录中找到 *这个确切的产品*。” 需要对同一类别中视觉相似对象进行细粒度区分；现成的嵌入效果不佳；使用度量学习进行微调至关重要。

在选择模型之前，始终要确认你解决的是哪一种问题。

## 构建它

### 步骤 1：三元组损失```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()
```一行。适用于 L2 归一化或原始嵌入。

### 步骤 2：半硬挖掘

给定一组嵌入和标签，为每个锚点找到最难的半硬负样本。```python
def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx
```每个锚点会获得最难的同类正样本和一个半硬负样本，该负样本距离正样本更远，但仍在边界内。

### 步骤 3：Recall@K```python
def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()
```在 L2 归一化嵌入上通过内积进行 top-k 检索等同于通过余弦相似度进行 top-k 检索。报告至少有一个正确邻居的查询的平均比例。

### 步骤 4：整合起来```python
import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()
```经过几百步训练后，嵌入向量的聚类会形成每个类别一个聚类。

## 使用它

2026年的生产环境堆栈：

- **DINOv2 + FAISS** — 通用的视觉检索。开箱即用。
- **CLIP + FAISS** — 当查询为文本时使用。
- **微调后的 DINOv2 + FAISS** — 实例级检索、人脸识别、时尚、电商。
- **Milvus / Weaviate / Qdrant** — 基于 FAISS 或 HNSW 的托管向量数据库包装器。

对于最先进的实例检索，方案是：使用 DINOv2 作为主干，添加一个嵌入头，使用实例标记的对在 triplet 或 InfoNCE 损失下微调，用 FAISS 进行索引。

## 部署它

本课将产出：

- `outputs/prompt-retrieval-loss-picker.md` — 一个提示，根据给定的检索问题选择 triplet / InfoNCE / ProxyNCA。
- `outputs/skill-recall-at-k-runner.md` — 一个技能，编写一个干净的评估框架，用于 recall@K，包含 train/val/gallery 的划分和正确的数据契约。

## 练习

1. **(简单)** 运行上面的玩具示例。使用 PCA 在训练前后绘制嵌入向量，观察六个聚类的形成。
2. **(中等)** 添加一个 ProxyNCA 损失实现：每个类别一个学习到的“代理”，在余弦相似度上使用标准交叉熵。在玩具数据上比较 ProxyNCA 与 triplet loss 的收敛速度。
3. **(困难)** 取 1,000 张 ImageNet 验证图像，通过 HuggingFace 使用 DINOv2 进行嵌入，构建一个 FAISS 扁平索引，并针对相同的图像作为查询（应为 1.0）以及使用 ImageNet 标签作为真实值的保留分割，报告 recall@{1, 5, 10}。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|--------------|
| Metric learning | “塑造空间” | 训练一个编码器，使其输出空间中的距离反映目标相似性 |
| Triplet loss | “拉近和推远” | L = max(0, d(a, p) - d(a, n) + margin); 标准的度量学习损失 |
| Semi-hard mining | “有用的负样本” | 距离锚点比正样本远但仍在 margin 内的负样本；经验上最有效 |
| Proxy-based loss | “类别原型” | 每个类别一个学习到的代理；在与代理的相似度上使用交叉熵；不需要配对挖掘 |
| Recall@K | “Top-K 命中率” | 查询中至少有一个正确结果出现在 Top K 中的比例 |
| Instance retrieval | “找到这个确切的东西” | 细粒度匹配；现成特征通常表现不佳 |
| FAISS | “最近邻库” | Facebook 的最近邻库；支持精确和近似索引 |
| HNSW | “图索引” | 分层可导航小世界；快速近似最近邻，内存开销小 |

## 进一步阅读

- [FaceNet: 人脸识别的统一嵌入（Schroff 等，2015）](https://arxiv.org/abs/1503.03832) — triplet loss / semi-hard mining 论文
- [为行人重识别辩护：Triplet Loss（Hermans 等，2017）](https://arxiv.org/abs/1703.07737) — triplet 微调实践指南
- [FAISS 文档](https://github.com/facebookresearch/faiss/wiki) — 每个索引，每个权衡
- [SMoT: 度量学习分类法（Kim 等，2021）](https://arxiv.org/abs/2010.06927) — 现代损失及其连接的综述
