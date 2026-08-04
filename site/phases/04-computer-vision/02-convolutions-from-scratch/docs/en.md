# 从零开始构建卷积

> 卷积是一种很小的密集层，你可以在图像上滑动它，每个位置都共享相同的权重。

**类型:** 构建
**语言:** Python
**先决条件:** 第三阶段（深度学习核心），第四阶段第01课（图像基础）
**时间:** ~75 分钟

## 学习目标

- 仅使用 NumPy 从零开始实现二维卷积，包括嵌套循环版本和一个向量化 `im2col` 版本
- 对于任何输入大小、核大小、填充和步长的组合，计算输出空间大小，并解释 `(H - K + 2P) / S + 1` 公式
- 手动设计核（边缘检测、模糊、锐化、Sobel），并解释为什么每个核会产生它所具有的激活模式
- 将卷积堆叠成一个特征提取器，并将堆叠的深度与感受野的大小联系起来

## 问题

在一个 224x224 的 RGB 图像上，一个全连接层每个神经元需要 224 * 224 * 3 = 150,528 个输入权重。一个具有 1,000 个单元的隐藏层已经拥有 1.5 亿个参数 —— 在你学到任何有用的东西之前。更糟糕的是，该层无法意识到图像左上角和右下角的狗是相同的模式。它将每个像素位置视为独立的，这正好是图像处理的错误做法：将猫向右移动三个像素不应该迫使网络重新学习这个概念。

图像模型需要的两个属性是 **平移等变性**（输入移动时输出也移动）和 **参数共享**（相同的特征检测器在任何地方都运行）。全连接层无法提供这两者。卷积可以免费提供两者。

卷积并不是为了深度学习而发明的。它就是 JPEG 压缩、Photoshop 中的高斯模糊、工业视觉中的边缘检测，以及所有曾经发布的音频滤波器所使用的相同操作。卷积神经网络从 2012 年到 2020 年在 ImageNet 上占据主导地位的原因是，卷积是对数据的正确先验假设：邻近的值相关，相同的模式可以在任何地方出现。

## 概念

### 一个核，滑动

二维卷积采用一个称为核（或滤波器）的小权重矩阵，在输入上滑动，并在每个位置计算逐元素乘积的总和。这个总和变成一个输出像素。```mermaid
flowchart LR
    subgraph IN["Input (H x W)"]
        direction LR
        I1["5 x 5 image"]
    end
    subgraph K["Kernel (3 x 3)"]
        K1["learned<br/>weights"]
    end
    subgraph OUT["Output (H-2 x W-2)"]
        O1["3 x 3 map"]
    end
    I1 --> |"slide kernel<br/>compute dot product<br/>at each position"| O1
    K1 --> O1

    style IN fill:#dbeafe,stroke:#2563eb
    style K fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```一个具体的 3x3 示例，在 5x5 输入上（无填充，步长 1）：```
Input X (5 x 5):                Kernel W (3 x 3):

  1  2  0  1  2                   1  0 -1
  0  1  3  1  0                   2  0 -2
  2  1  0  2  1                   1  0 -1
  1  0  2  1  3
  2  1  1  0  1

The kernel slides across every valid 3 x 3 window. Output Y is 3 x 3:

 Y[0,0] = sum( W * X[0:3, 0:3] )
 Y[0,1] = sum( W * X[0:3, 1:4] )
 Y[0,2] = sum( W * X[0:3, 2:5] )
 Y[1,0] = sum( W * X[1:4, 0:3] )
 ... and so on
```那个公式 —— **共享权重、局部性、滑动窗口** —— 就是全部的核心思想。其他的一切都是琐碎的细节。

### 输出尺寸公式

给定输入空间尺寸 `H`，核尺寸 `K`，填充 `P`，步长 `S`：```
H_out = floor( (H - K + 2P) / S ) + 1
```记住这一点。你将根据架构的不同，计算它数十次。

| Scenario | H | K | P | S | H_out |
|----------|---|- --|---|---|--- ----|
| Valid conv, no padding | 32 | 3 | 0 | 1 | 30 |
| Same conv (preserves size) | 32 | 3 | 1 | 1 | 32 |
| Downsample by 2 | 32 | 3 | 1 | 2 | 16 |
| Pool 2x2 | 32 | 2 | 0 | 2 | 16 |
| Large receptive field | 32 | 7 | 3 | 2 | 16 |

“Same padding”表示选择P，使得当S等于1时，H_out等于H。对于奇数K，P等于(K - 1) / 2。这就是为什么3x3卷积核占主导地位——它们是最小的具有中心点的奇数卷积核。

### Padding

没有padding时，每个卷积都会缩小特征图。堆叠20个这样的卷积，你的224x224图像会变成184x184，这在边界上浪费了计算资源，并且使需要匹配形状的残差连接变得更加复杂。```
Zero padding (P = 1) on a 5 x 5 input:

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0       Now the kernel can centre on pixel
  0  1  0  2  1  3  0       (0, 0) and still have three rows and
  0  2  1  1  0  1  0       three columns of values to multiply.
  0  0  0  0  0  0  0
```实践中遇到的模式：`zero`（最常见），`reflect`（镜像边缘，避免生成模型中的硬边界），`replicate`（复制边缘），`circular`（环绕，用于环形问题）。

### 步长

步长是滑动的步长。`stride=1` 是默认值。`stride=2` 会将空间维度减半，是经典的方式在 CNN 中进行下采样而不使用单独的池化层——每个现代架构（ResNet、ConvNeXt、MobileNet）都会在某些地方用步长卷积代替最大池化。```
Stride 1 on a 5 x 5 input, 3 x 3 kernel:

  starts: (0,0) (0,1) (0,2)        -> output row 0
          (1,0) (1,1) (1,2)        -> output row 1
          (2,0) (2,1) (2,2)        -> output row 2

  Output: 3 x 3

Stride 2 on the same input:

  starts: (0,0) (0,2)              -> output row 0
          (2,0) (2,2)              -> output row 1

  Output: 2 x 2
```### 多个输入通道

真实图像有三个通道。对RGB输入进行3x3卷积实际上是一个3x3x3的体积：每个输入通道一个3x3的切片。在每个空间位置，你将对所有三个切片进行乘法和求和操作，并加上一个偏置。```
Input:   (C_in,  H,  W)        3 x 5 x 5
Kernel:  (C_in,  K,  K)        3 x 3 x 3 (one kernel)
Output:  (1,     H', W')       2D map

For a layer that produces C_out output channels, you stack C_out kernels:

Weight:  (C_out, C_in, K, K)   e.g. 64 x 3 x 3 x 3
Output:  (C_out, H', W')       64 x 3 x 3

Parameter count: C_out * C_in * K * K + C_out   (the + C_out is biases)
```最后一行是你在规划模型时将要计算的部分。一个针对三通道输入的 64 通道 3x3 卷积层有 `64 * 3 * 3 * 3 + 64 = 1,792` 个参数。很便宜。

### im2col 技巧

嵌套循环虽然易于阅读，但速度很慢。GPU 希望进行大规模的矩阵乘法。技巧是：将输入中每个感受野窗口展平成一个大矩阵的一列，将核展平成一行，这样整个卷积就变成一次矩阵乘法。```mermaid
flowchart LR
    X["Input<br/>(C_in, H, W)"] --> IM2COL["im2col<br/>(extract patches)"]
    IM2COL --> COLS["Cols matrix<br/>(C_in * K * K, H_out * W_out)"]
    W["Weight<br/>(C_out, C_in, K, K)"] --> FLAT["Flatten<br/>(C_out, C_in * K * K)"]
    FLAT --> MM["matmul"]
    COLS --> MM
    MM --> OUT["Output<br/>(C_out, H_out * W_out)<br/>reshape to (C_out, H_out, W_out)"]

    style X fill:#dbeafe,stroke:#2563eb
    style W fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```每个生产环境中的卷积（conv）实现都是这种基本形式的某种变体，再加上缓存分块（cache-tiling）技巧（直接卷积、Winograd、大内核的FFT卷积）。理解im2col，你就理解了核心。

### 接收域

一个3x3的卷积层会查看9个输入像素。堆叠两个3x3的卷积层，第二层中的一个神经元会查看5x5的输入像素。三个3x3的卷积层会得到7x7的接收域。一般来说：```
RF after L stacked K x K convs (stride 1) = 1 + L * (K - 1)

With strides:   RF grows multiplicatively with stride along each layer.
```“3x3 all the way down”之所以有效（如VGG、ResNet、ConvNeXt）的原因是，两个3x3的卷积层所看到的输入区域与一个5x5的卷积层相同，但参数更少，并且中间多了一个非线性变换。```figure
convolution-kernel
```## 构建它

### 步骤 1：填充数组

从最简单的原始操作开始：一个函数，用于在 H x W 的数组周围填充零。```python
import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```尾轴技巧 `x.shape[:-2]` 表示相同的函数可以在 `(H, W)`、`(C, H, W)` 或 `(N, C, H, W)` 上无修改地工作。

### 步骤 2：使用嵌套循环的二维卷积

参考实现 —— 慢，但明确无误。这基本上就是 `torch.nn.functional.conv2d` 所做的事情。```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```四个嵌套循环（输出通道、行、列，加上对 C_in、kh、kw 的隐式求和）。这是你要检查每个更快实现的基准。

### 步骤 3：使用手工设计的核进行验证

构建一个垂直 Sobel 核，将其应用于一个合成的阶梯图像，观察垂直边缘被点亮。```python
def synthetic_step_image():
    img = np.zeros((1, 16, 16), dtype=np.float32)
    img[:, :, 8:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```在第7列（从左到右亮度增加）上期望出现较大的正值，其他位置应为零。这个单一的打印结果是你验证数学是否正确的依据。

### 第4步：im2col

将输入中每个内核大小的窗口转换为矩阵中的一列。对于`C_in=3, K=3`，每一列包含27个数字。```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```它仍然是一个 Python 循环，但现在繁重的计算将由一个向量化 matmul 来完成。

### 步骤 5：通过 im2col + matmul 实现快速卷积

用一个矩阵乘法来替代四重循环。```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```正确性检查：运行两种实现并进行比较。```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)

print(f"max abs diff: {np.max(np.abs(y_naive - y_im2col)):.2e}")
````max abs diff` 应该接近 `1e-5` —— 差异是浮点数累加顺序的问题，而不是错误。

### 第6步：一组手工设计的内核

五个滤波器展示了在任何训练之前，单个卷积层能够表达的内容。```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    x = img2d[None].astype(np.float32)
    w = kernel[None, None]
    return conv2d_im2col(x, w, padding=1)[0]
```应用于任何灰度图像，模糊可以柔化图像，锐化可以增强边缘，Sobel-x 可以突出垂直边缘，Sobel-y 可以突出水平边缘。这些正是 AlexNet 和 VGG 中 *第一个* 训练好的卷积层最终学到的模式 —— 因为一个优秀的图像模型，无论后续任务是什么，都需要边缘和斑块检测器。

## 使用方法

PyTorch 的 `nn.Conv2d` 用 autograd、CUDA 内核和 cuDNN 优化封装了同样的操作。形状语义完全一致。```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"weight shape: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"bias shape:   {tuple(conv.bias.shape)}")
print(f"param count:  {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)
y = conv(x)
print(f"\ninput  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}")
```将 `padding=1` 换成 `padding=0`，输出变为 222x222。将 `stride=1` 换成 `stride=2`，输出变为 112x112。使用你上面记住的相同公式。

## 发布它

本课将产出以下内容：

- `outputs/prompt-cnn-architect.md` — 一个提示，给定输入尺寸、参数预算和目标感受野，设计一个由 `Conv2d` 层组成的堆叠，并在每一步都正确设置 K/S/P。
- `outputs/skill-conv-shape-calculator.md` — 一项技能，可以逐层遍历网络规格，并返回每一块的输出形状、感受野和参数数量。

## 练习

1. **(简单)** 给定 128x128 灰度输入和一组 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]`，手动计算每层的输出空间尺寸和感受野。使用 PyTorch 的 `nn.Sequential` 对虚拟卷积进行验证。
2. **(中等)** 扩展 `conv2d_naive` 和 `conv2d_im2col` 以接受一个 `groups` 参数。展示 `groups=C_in=C_out` 可以重现深度可分离卷积，并且其参数数量为 `C * K * K` 而不是 `C * C * K * K`。
3. **(困难)** 手动实现 `conv2d_im2col` 的反向传播：给定输出的梯度，计算 `x` 和 `w` 的梯度。使用 `torch.autograd.grad` 在相同的输入和权重上进行验证。技巧：im2col 的梯度是 `col2im`，并且需要对重叠窗口进行累积。

## 术语表

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 卷积 | "滑动一个滤波器" | 在每个空间位置上应用一个可学习的点积，使用共享权重；数学上是互相关，但大家都称之为卷积 |
| 核/滤波器 | "特征检测器" | 一个形状为 (C_in, K, K) 的小权重张量，与输入窗口的点积生成一个输出像素 |
| 步长 | "跳跃的距离" | 两个连续核放置之间的步长；步长为 2 会将每个空间维度减半 |
| 填充 | "边缘的零" | 在输入周围添加额外的值，以便核可以对齐在边界像素上；`same` 填充使输出尺寸等于输入尺寸 |
| 感受野 | "神经元看到的内容" | 给定输出激活所依赖的原始输入的区域，随着深度和步长的增加而扩大 |
| im2col | "GEMM 技巧" | 将每个感受野窗口重新排列为列，使卷积变成一个大的矩阵乘法 —— 所有快速卷积核的核心 |
| 深度可分离卷积 | "每个通道一个核" | 一个具有 `groups == C_in` 的卷积，每个输出通道仅由对应的输入通道计算得出；MobileNet 和 ConvNeXt 的骨干网络 |
| 平移等变性 | "输入移动，输出也移动" | 输入移动 k 像素，输出也移动 k 像素的性质；共享权重会自然带来这一性质 |

## 进一步阅读

- [深度学习卷积算术指南 (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285) —— 每个课程都默然复制的关于填充/步长/扩张的权威图解
- [CS231n: 视觉识别的卷积神经网络](https://cs231n.github.io/convolutional-networks/) —— 经典的讲义笔记，包括最初的 im2col 解释
- [注释卷积网络 (fast.ai)](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb) —— 从手动卷积到训练后的数字分类器的笔记本
- [卷积神经网络的感受野算术 (Dang Ha The Hien)](https://distill.pub/2019/computing-receptive-fields/) —— 以论文质量进行交互式解释的感受野计算说明
