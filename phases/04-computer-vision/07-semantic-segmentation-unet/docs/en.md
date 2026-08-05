# 语义分割 — U-Net

> 分割是在每个像素上进行分类。U-Net通过将下采样编码器与上采样解码器配对，并在它们之间连接跳跃连接，使这一过程得以实现。

**类型:** 构建
**语言:** Python
**前提条件:** 第四阶段第03课（CNNs），第四阶段第04课（图像分类）
**时间:** ~75 分钟

## 学习目标

- 区分语义分割、实例分割和全景分割，并根据给定的问题选择合适的任务
- 在PyTorch中从零开始构建U-Net，包括编码器模块、瓶颈层、带有转置卷积的解码器和跳跃连接
- 实现像素级别的交叉熵损失、Dice损失以及当前医疗和工业分割默认使用的组合损失
- 按类别读取IoU和Dice指标，并诊断低分是否来自于小物体的召回率、边界准确性或类别不平衡

## 问题描述

分类输出每张图像一个标签。检测输出每张图像几个框。分割输出每个像素一个标签。对于大小为 `H x W` 的输入，输出是一个形状为 `H x W`（语义）或 `H x W x N_instances`（实例）的张量。这意味着每张图像有数百万个预测，而不是一个。

分割的结构就是为什么它几乎驱动了所有密集预测的视觉产品：医学成像（肿瘤掩膜）、自动驾驶（道路、车道、障碍物）、卫星（建筑足迹、作物边界）、文档解析（布局区域）、机器人（可抓取区域）。这些任务都无法通过在物体周围画一个框来解决；它们需要精确的轮廓。

架构问题的陈述很简单，但解决起来并不容易：你需要网络同时看到图像的全局上下文（这是什么场景）和局部像素细节（哪个像素是道路而不是人行道）。标准的CNN通过空间压缩来获取上下文，这会丢失细节。U-Net是同时获得两者的设计。

## 概念

### 语义 vs 实例 vs 全景

```mermaid
flowchart LR
    IN["Input image"] --> SEM["Semantic<br/>(pixel → class)"]
    IN --> INS["Instance<br/>(pixel → object id,<br/>only foreground classes)"]
    IN --> PAN["Panoptic<br/>(every pixel → class + id)"]

    style SEM fill:#dbeafe,stroke:#2563eb
    style INS fill:#fef3c7,stroke:#d97706
    style PAN fill:#dcfce7,stroke:#16a34a
```- **语义分割**（Semantic）说“这个像素是道路，那个像素是汽车。”两辆相邻的汽车会被合并成一个模糊的块。
- **实例分割**（Instance）说“这个像素是汽车#3，那个像素是汽车#5。”忽略背景信息（“stuff” = 天空、道路、草地）。
- **全景分割**（Panoptic）结合两者：每个像素都获得一个类别标签，每个实例都有一个唯一的ID，同时对“stuff”和“things”进行分割。

本课讲解语义分割。下节课（Mask R-CNN）讲解实例分割。

### U-Net 的结构

```mermaid
flowchart LR
    subgraph ENC["Encoder (contracting)"]
        E1["64<br/>H x W"] --> E2["128<br/>H/2 x W/2"]
        E2 --> E3["256<br/>H/4 x W/4"]
        E3 --> E4["512<br/>H/8 x W/8"]
    end
    subgraph BOT["Bottleneck"]
        B1["1024<br/>H/16 x W/16"]
    end
    subgraph DEC["Decoder (expanding)"]
        D4["512<br/>H/8 x W/8"] --> D3["256<br/>H/4 x W/4"]
        D3 --> D2["128<br/>H/2 x W/2"]
        D2 --> D1["64<br/>H x W"]
    end
    E4 --> B1 --> D4
    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
    E4 -. skip .-> D4
    D1 --> OUT["1x1 conv<br/>classes"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style BOT fill:#fef3c7,stroke:#d97706
    style DEC fill:#dcfce7,stroke:#16a34a
```

编码器将空间分辨率降低四次，同时将通道数翻倍。解码器则相反：将空间分辨率翻倍四次，同时将通道数减半。跳跃连接在每一层分辨率上将编码器的特征与解码器的特征拼接在一起。最终的 1x1 卷积在完整分辨率上映射 `64 -> num_classes`。

为什么需要跳跃连接：当解码器试图输出像素级预测时，它只看到了小的特征图。如果没有跳跃连接，它无法准确地定位边缘，因为这些信息在编码器中被压缩掉了。跳跃连接将编码器在下传过程中计算的高分辨率特征图传递给解码器。

### 反转卷积与双线性上采样

解码器需要扩展空间维度。有两种选择：

- **反转卷积** (`nn.ConvTranspose2d`) — 可学习的上采样。历史上的 U-Net 默认方法。如果步长和核尺寸不能整除，可能会产生棋盘格伪影。
- **双线性上采样 + 3x3 卷积** — 光滑上采样后接一个卷积。伪影更少，参数更少，现在是现代默认方法。

两者都可以在实际中看到。对于第一个 U-Net，双线性方法更安全。

### 像素网格上的交叉熵

对于具有 C 个类别的语义分割，模型的输出是 `(N, C, H, W)`。目标是 `(N, H, W)`，包含整数类别 ID。交叉熵与分类情况相同，只是在每个空间位置上应用：

```
Loss = mean over (n, h, w) of -log( softmax(logits[n, :, h, w])[target[n, h, w]] )
```

`F.cross_entropy` 在 PyTorch 中原生处理这种形状。无需重塑。

### Dice 损失及其必要性

交叉熵损失将每个像素视为同等重要。当某一类在图像中占据主导地位时（例如医学影像中 99% 是背景，1% 是肿瘤），这种做法是错误的。网络可以通过在所有位置预测背景而达到 99% 的准确率，但这样的模型仍然毫无用处。

Dice 损失通过直接优化预测掩膜和真实掩膜之间的重叠程度来解决这个问题：

```
Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + epsilon)
Dice_loss = 1 - Dice
```

其中 `p` 是某类的 sigmoid/softmax 概率图，而 `y` 是二值真实标签掩膜。只有当重叠完全吻合时，损失值才为零。由于它是基于比例的，因此类别不平衡问题无关紧要。

在实践中，使用 **联合损失**：

```
L = L_cross_entropy + lambda * L_dice       (lambda ~ 1)
```

交叉熵损失在训练初期提供稳定的梯度；Dice损失则将训练后期的重点放在与掩膜形状的匹配上。这种组合是医学影像领域的默认选择，在任何类别不平衡的数据集上都难以被超越。

### 评估指标

- **像素准确率** — 预测正确的像素百分比。计算成本低。和分类中的准确率一样，在类别不平衡的数据上表现不佳。
- **每类IoU** — 每个类别的掩膜的交并比；跨类别平均 = mIoU。
- **Dice（像素F1）** — 类似于IoU；`Dice = 2 * IoU / (1 + IoU)`。医学影像领域更偏好Dice，而社区驱动更偏好IoU；它们之间是单调相关的。
- **边界F1** — 衡量预测边界与真实边界之间的接近程度，即使微小的偏移也会受到惩罚。对于高精度任务，如半导体检测，非常重要。

报告每类的IoU，而不仅仅是mIoU。平均IoU会在其他九个类别达到85%时，将一个类别隐藏在15%的水平。

### 输入分辨率的权衡

U-Net的编码器将分辨率减半四次，因此输入必须能被16整除。医学图像通常为512x512或1024x1024。自动驾驶图像的裁剪尺寸为2048x1024。U-Net的内存消耗随着`H * W * C_max`的增加而增加，当输入为1024x1024且瓶颈通道数为1024时，前向传播过程已经使用了数GB的VRAM。

两种标准的解决方法：
1. 对输入进行分块处理 — 使用重叠的256x256块进行处理，然后拼接。
2. 用扩张卷积代替瓶颈层，保持更高的空间分辨率，同时扩大感受野（DeepLab系列）。

对于第一个模型，使用256x256输入和64通道基础的U-Net可以在8GB VRAM上舒适地训练。

## 构建它

### 第一步：编码器模块

两个3x3的卷积层，带有批量归一化和ReLU激活。第一个卷积层改变通道数；第二个卷积层保持通道数不变。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

此模块在整个过程中被重复使用。`bias=False` 因为 BN 的 beta 处理了偏置。

### 步骤 2：下采样和上采样模块

```python
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

仅空间形状检查（`shape[-2:]`）处理那些维度不能被16整除的输入；一个安全的 `F.interpolate` 在拼接前对张量进行对齐。比较完整形状时也会触发通道数差异，这应该是一个明显的错误，而不是静默插值。

### 第3步：U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16 + base * 8, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)

net = UNet(in_channels=3, num_classes=2, base=32)
x = torch.randn(1, 3, 256, 256)
print(f"output: {net(x).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

输出形状 `(1, 2, 256, 256)` — 与输入相同的空域尺寸，`num_classes` 个通道。在 `base=32` 大约有 7.7M 个参数。

### 第 4 步：损失函数

```python
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.item(), "dice": dc.item()}
```Dice 按类计算然后取平均（宏 Dice）。`eps` 防止在批次中不存在的类别上出现除以零的情况。

### 步骤 5：IoU 指标

```python
@torch.no_grad()
def iou_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (targets == c)
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious[c] = (inter / union) if union > 0 else torch.tensor(float("nan"))
    return ious
```

返回一个长度为 C 的向量。`nan` 表示批次中不存在的类别 —— 计算 mIoU 时不要对这些类别进行平均。

### 第 6 步：用于端到端验证的合成数据集

在彩色背景上生成形状，使网络必须学习形状，而不是像素颜色。

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader

def synthetic_segmentation(num_samples=200, size=64, seed=0):
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    for i in range(num_samples):
        bg = rng.uniform(0, 1, (3,))
        images[i] = bg
        masks[i] = 0
        num_shapes = rng.integers(1, 4)
        for _ in range(num_shapes):
            cls = int(rng.integers(1, 3))
            color = rng.uniform(0, 1, (3,))
            cx, cy = rng.integers(10, size - 10, size=2)
            r = int(rng.integers(4, 12))
            yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
            if cls == 1:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            else:
                mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            images[i][mask] = color
            masks[i][mask] = cls
        images[i] += rng.normal(0, 0.02, images[i].shape)
        images[i] = np.clip(images[i], 0, 1)
    return images, masks


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask
```

三个类别：背景（0）、圆形（1）、方形（2）。网络必须学会区分形状。

### 第7步：训练循环

```python
def train_one_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    loss_sum, total = 0.0, 0
    iou_sum = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = combined_loss(logits, y, num_classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        iou_sum += iou_per_class(logits, y, num_classes).nan_to_num(0)
    return loss_sum / total, iou_sum / len(loader)
```

在合成数据集上运行10-30个epoch，观察形状类别的mIoU值超过0.9。注意`nan_to_num(0)`将批次中不存在的类别视为零；为了获得更准确的每类IoU值，在评估时应通过存在性进行掩码，并使用`torch.nanmean`跨批次计算，而不是在这里进行平均。

## 使用方法

在生产环境中，`segmentation_models_pytorch`（"smp"）将任何标准分割架构与任何torchvision或timm的主干网络进行封装。只需三行代码：

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
)
```

在实际工作中也值得了解以下内容：
- **DeepLabV3+** 用扩张卷积取代最大池化下采样，使瓶颈层保持分辨率；在卫星和驾驶数据上边界检测更快。
- **SegFormer** 用分层的Transformer取代卷积编码器；在许多基准测试中目前是SOTA。
- **Mask2Former** / **OneFormer** 在单一架构中统一了语义分割、实例分割和全景分割。

这三种模型都可以在 `smp` 或 `transformers` 中直接替换，使用相同的数据加载器。

## 部署它

本课将产出以下内容：

- `outputs/prompt-segmentation-task-picker.md` — 一个提示，用于在语义分割、实例分割和全景分割之间进行选择，并为特定任务命名架构。
- `outputs/skill-segmentation-mask-inspector.md` — 一项技能，用于报告类别分布、预测掩码的统计信息，以及预测不足或边界模糊的类别。

## 练习

1. **(简单)** 为一个二分类分割任务（前景与背景）实现 `bce_dice_loss`。在合成的二分类数据集上验证，当前景仅占5%像素时，联合损失比BCE单独使用时收敛得更快。
2. **(中等)** 用 `nn.ConvTranspose2d` 上采样模块替换 `nn.Upsample + conv` 上采样模块。在合成数据集上分别训练两者并比较mIoU。观察转置卷积版本中棋盘格伪影出现的位置。
3. **(困难)** 使用一个真实的分割数据集（Oxford-IIIT Pets、Cityscapes mini split 或医学子集），训练U-Net直到达到 `smp.Unet` 参考值的2个IoU点以内。报告每个类别的IoU，并识别出哪些类别从将Dice加入损失函数中获益最多。

## 重要术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|--------------|
| 语义分割 | "为每个像素打标签" | 每个像素分类为C个类别；同一类别的实例会合并 |
| 实例分割 | "为每个对象打标签" | 分离同一类别的不同实例；仅包括前景 |
| 全景分割 | "语义 + 实例" | 每个像素都有一个类别；每个对象实例还有一个唯一ID |
| 跳跃连接 | "U-Net桥" | 将编码器特征连接到对应分辨率的解码器特征；保留高频细节 |
| 转置卷积 | "反卷积" | 可学习的上采样；可能会产生棋盘格伪影 |
| Dice损失 | "重叠损失" | 1 - 2|A ∩ B| / (|A| + |B|)；直接优化掩码重叠，对类别不平衡具有鲁棒性 |
| mIoU | "平均交并比" | 各类别IoU的平均值；分割任务的社区标准度量指标 |
| 边界F1 | "边界精度" | 仅在边界像素上计算的F1分数；对精度要求高的任务很重要 |

## 进一步阅读

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) — 原始论文；大家复制的图在第2页
- [Fully Convolutional Networks (Long et al., 2015)](https://arxiv.org/abs/1411.4038) — 第一篇将分割作为端到端卷积问题的论文
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — 生产级分割的参考；包括所有标准架构和所有标准损失
- [Lessons learned from training SOTA segmentation (kaggle.com competitions)](https://www.kaggle.com/code/iafoss/carvana-unet-pytorch) — 说明为什么TTA、伪标签和类别权重在真实数据中重要的指南
