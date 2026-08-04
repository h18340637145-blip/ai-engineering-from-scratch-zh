# 图像基础 —— 像素、通道、颜色空间

> 图像是一组光样本的张量。你将来使用的所有视觉模型都始于这一事实。

**类型:** 构建
**语言:** Python
**先决条件:** 第一阶段第12课（张量运算），第三阶段第11课（PyTorch入门）
**时间:** ~45分钟

## 学习目标

- 解释一个连续场景如何被离散化为像素，以及采样/量化决策如何设定所有下游模型的上限
- 以NumPy数组形式读取、切片和检查图像，并熟练地在HWC和CHW布局之间切换
- 在RGB、灰度、HSV和YCbCr之间进行转换，并说明每种颜色空间存在的原因
- 严格按照torchvision的要求应用像素级预处理（归一化、标准化、调整大小、通道优先）

## 问题

你将阅读的每篇论文、下载的每个预训练权重、调用的每个视觉API都假定输入具有特定的编码。传递一个`uint8`图像给期望`float32`的模型，它仍然会运行，但会静默地产生垃圾结果。将BGR图像输入一个在RGB上训练的网络，准确率会下降10个百分点。当模型期望通道优先的输入却收到通道最后的输入时，第一个卷积层会将高度视为特征通道。这些情况都不会引发错误。它们只会破坏你的指标，而你将花一周时间寻找一个隐藏在文件加载方式中的错误。

一旦你知道卷积是在什么上滑动，卷积本身并不复杂。困难的部分在于“图像”对相机、JPEG解码器、PIL、OpenCV、torchvision和CUDA内核意味着不同的东西。每个工具链都有自己的轴顺序、字节范围和通道约定。无法理清这些差异的视觉工程师会部署出有缺陷的流水线。

本课修复了基础，以便后续内容可以在此基础上构建。到课程结束时，你将知道像素是什么，为什么每个像素有三个数字而不是一个，“使用ImageNet统计数据进行归一化”实际上做了什么，以及如何在本阶段其他课程所假设的两种或三种布局之间转换。

## 概念

### 整个预处理流程一览

每个生产级的视觉系统都是相同的可逆变换序列。如果某一步出错，模型看到的输入就与它训练时的输入不同。```mermaid
flowchart LR
    A["Image file<br/>(JPEG/PNG)"] --> B["Decode<br/>uint8 HWC"]
    B --> C["Convert<br/>colorspace<br/>(RGB/BGR/YCbCr)"]
    C --> D["Resize<br/>shorter side"]
    D --> E["Center crop<br/>model size"]
    E --> F["Divide by 255<br/>float32 [0,1]"]
    F --> G["Subtract mean<br/>Divide by std"]
    G --> H["Transpose<br/>HWC → CHW"]
    H --> I["Batch<br/>CHW → NCHW"]
    I --> J["Model"]

    style A fill:#fef3c7,stroke:#d97706
    style J fill:#ddd6fe,stroke:#7c3aed
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#bfdbfe,stroke:#2563eb
```两个红色和蓝色的框是80%的静默故障所在：缺少标准化和错误的布局。

### 一个像素是一个样本，而不是一个方块

相机传感器计算落在微小探测器网格上的光子数量。每个探测器在极短的时间内整合光线，并发出与击中它的光子数量成比例的电压。然后传感器将该电压离散化为一个整数。一个探测器对应一个像素。```
Continuous scene                 Sensor grid                     Digital image
(infinite detail)                (H x W detectors)               (H x W integers)

    ~~~~~                        +--+--+--+--+--+                 210 198 180 155 120
   ~   ~   ~                     |  |  |  |  |  |                 205 195 178 152 118
  ~ light ~      ---->           +--+--+--+--+--+     ---->       200 190 175 150 115
   ~~~~~                         |  |  |  |  |  |                 195 185 170 148 112
                                 +--+--+--+--+--+                 188 180 165 145 108
```在这个步骤中会发生两个选择，它们决定了后续所有处理的上限：

- **空间采样**决定了场景每度有多少个探测器。太少的话，边缘会出现锯齿（混叠现象）。太多的话，存储和计算需求将急剧增加。
- **强度量化**决定了电压被分组的精细程度。8 位提供 256 个等级，是显示的标准。10、12、16 位可以提供更平滑的渐变，在医学成像、HDR 和原始传感器流水线中非常重要。

像素不是一个带颜色的有面积的方块。它是一个单独的测量值。当你调整大小或旋转时，你实际上是在重新采样这个测量网格。

### 为什么是三个通道

一个探测器会统计整个可见光谱范围内的光子数量——也就是灰度图像。为了获得颜色，传感器会用红、绿、蓝三种滤光片的马赛克覆盖整个网格。经过去马赛克处理后，每个空间位置都有三个整数：红滤光片探测器的响应值、绿滤光片探测器的响应值和蓝滤光片探测器的响应值。这三个整数构成了像素的 RGB 三元组。```
One pixel in memory:

    (R, G, B) = (210, 140, 30)   <- reddish-orange

An H x W RGB image:

    shape (H, W, 3)     stored as   H rows of W pixels of 3 values
                                    each in [0, 255] for uint8
```三并不是魔法。深度相机增加了一个 Z 通道。卫星增加了红外和紫外波段。医学扫描通常有一个通道（X 光、CT）或许多（高光谱）。通道的数量是最后一个轴；卷积层学会在该轴上进行混合。

### 两种布局惯例：HWC 和 CHW

相同的张量，两种排列方式。每个库都会选择其中一种。```
HWC (height, width, channels)           CHW (channels, height, width)

   W ->                                    H ->
  +-----+-----+-----+                     +-----+-----+
H |R G B|R G B|R G B|                   C |R R R R R R|
| +-----+-----+-----+                   | +-----+-----+
v |R G B|R G B|R G B|                   v |G G G G G G|
  +-----+-----+-----+                     +-----+-----+
                                          |B B B B B B|
                                          +-----+-----+

   PIL, OpenCV, matplotlib,              PyTorch, most deep learning
   almost every image file on disk       frameworks, cuDNN kernels
```CHW 的存在是因为卷积核在 H 和 W 上滑动。将通道轴放在前面意味着每个核可以看见每个通道的连续 2D 平面，这可以很好地向量化。磁盘格式保留 HWC，因为这与传感器输出的扫描线方式一致。

你将无数次输入的一行转换：```
img_chw = img_hwc.transpose(2, 0, 1)      # NumPy
img_chw = img_hwc.permute(2, 0, 1)        # PyTorch tensor
```内存布局，可视化：```mermaid
flowchart TB
    subgraph HWC["HWC — pixels stored interleaved (PIL, OpenCV, JPEG)"]
        H1["row 0: R G B | R G B | R G B ..."]
        H2["row 1: R G B | R G B | R G B ..."]
        H3["row 2: R G B | R G B | R G B ..."]
    end
    subgraph CHW["CHW — channels stored as stacked planes (PyTorch, cuDNN)"]
        C1["plane R: entire H x W of red values"]
        C2["plane G: entire H x W of green values"]
        C3["plane B: entire H x W of blue values"]
    end
    HWC -->|"transpose(2, 0, 1)"| CHW
    CHW -->|"transpose(1, 2, 0)"| HWC
```### 字节范围和数据类型

三种惯例占主导地位：

| 惯例 | 数据类型 | 范围 | 常见出现位置 |
|------|---------|------|-------------|
| 原始 | `uint8` | [0, 255] | 磁盘上的文件，PIL，OpenCV输出 |
| 归一化 | `float32` | [0.0, 1.0] | 在`img.astype('float32') / 255`之后 |
| 标准化 | `float32` | 大约[-2, +2] | 在减去均值并除以标准差之后 |

卷积网络是在标准化的输入上进行训练的。ImageNet统计数据 `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` 是整个ImageNet训练集上三个通道的算术平均值和标准差，计算是在[0, 1]归一化像素上进行的。将原始的`uint8`输入到期望标准化浮点数的模型中，是应用视觉领域中最常见的静默失败原因。

### 颜色空间及其存在的原因

RGB是捕获格式，但它不总是模型最有用的表示方式。```
 RGB               HSV                       YCbCr / YUV

 R red             H hue (angle 0-360)       Y luminance (brightness)
 G green           S saturation (0-1)        Cb chroma blue-yellow
 B blue            V value/brightness (0-1)  Cr chroma red-green

 Linear to         Separates color from      Separates brightness from
 sensor output     brightness. Useful for    color. JPEG and most video
                   color thresholding, UI    codecs compress the chroma
                   sliders, simple filters   channels harder because the
                                             human eye is less sensitive
                                             to chroma detail than to Y.
```对于大多数现代的卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think

<>

对于大多数现代卷积神经网络（CNN），你输入的是RGB。你遇到其他颜色空间的情况包括：

- **HSV** — 经典的计算机视觉代码，基于颜色的分割，白平衡。
- **YCbCr** — 阅读JPEG内部结构，视频处理流程，仅在Y通道上运行的超分辨率模型。
- **灰度** — 光学字符识别（OCR），文档模型，任何颜色作为干扰变量而非信号的情况。

从RGB转换为灰度是加权求和，而不是平均，因为人眼对绿色比对红色或蓝色更敏感：

 /no_think```
Y = 0.299 R + 0.587 G + 0.114 B       (ITU-R BT.601, the classic weights)
```### 宽高比、调整大小和插值

每个模型都有一个固定的输入尺寸（大多数ImageNet分类器为224x224，现代检测器为384x384或512x512）。你的图像很少会匹配这个尺寸。有三个重要的调整大小选项：

- **将较短边调整到目标尺寸，然后居中裁剪** —— 标准的ImageNet方法。保留宽高比，丢弃边缘像素的一条条带。
- **调整大小并填充** —— 保留宽高比和所有像素，添加黑色条带。检测和OCR的标准方法。
- **直接调整到目标尺寸** —— 拉伸图像。成本低，会扭曲几何形状，适合许多分类任务。

插值方法决定了当新网格与旧网格不一致时，如何计算中间像素：```
Nearest neighbour     fastest, blocky, only choice for masks/labels
Bilinear              fast, smooth, default for most image resizing
Bicubic               slower, sharper on upscaling
Lanczos               slowest, best quality, used for final display
```经验法则：训练时使用双线性，查看的资源使用双三次或兰佐斯插值，包含整数类别 ID 的任何内容使用最近邻插值。```figure
conv-output-size
```## 构建它

### 步骤 1：加载图像并检查其形状

使用 Pillow 加载任何 JPEG 或 PNG 图像，将其转换为 NumPy 数组，并打印你得到的结果。为了有一个可以离线运行的确定性示例，可以合成一个图像。```python
import numpy as np
from PIL import Image

def synthetic_rgb(h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    g = yy * 255
    b = (1 - yy) * xx * 255
    rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)

arr = synthetic_rgb()
# Or load from disk:
# arr = np.asarray(Image.open("your_image.jpg").convert("RGB"))

print(f"type:   {type(arr).__name__}")
print(f"dtype:  {arr.dtype}")
print(f"shape:  {arr.shape}     # (H, W, C)")
print(f"min:    {arr.min()}")
print(f"max:    {arr.max()}")
print(f"pixel at (0, 0): {arr[0, 0]}")
```预期输出：`shape: (H, W, 3)`, `dtype: uint8`, 范围 `[0, 255]`。无论字节来自摄像头、JPEG 解码器还是合成生成器，这都是其在磁盘上的标准表示形式。

### 步骤 2：分离通道并重新排列布局

分别提取 R、G、B 通道，然后将格式从 HWC 转换为 CHW 以用于 PyTorch。```python
R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]
print(f"R shape: {R.shape}, mean: {R.mean():.1f}")
print(f"G shape: {G.shape}, mean: {G.mean():.1f}")
print(f"B shape: {B.shape}, mean: {B.mean():.1f}")

arr_chw = arr.transpose(2, 0, 1)
print(f"\nHWC shape: {arr.shape}")
print(f"CHW shape: {arr_chw.shape}")
```三个灰度平面，每个通道一个。CHW只是重新排列轴的顺序；当内存布局允许时，严格来说不需要复制数据。

### 步骤3：灰度和HSV转换

加权求和的灰度，然后手动进行RGB到HSV的转换。```python
def rgb_to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)

def rgb_to_hsv(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    rmax = mask & (cmax == r)
    gmax = mask & (cmax == g)
    bmax = mask & (cmax == b)
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0

    s = np.where(cmax > 0, delta / cmax, 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)

gray = rgb_to_grayscale(arr)
hsv = rgb_to_hsv(arr)
print(f"gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
print(f"hsv   shape: {hsv.shape}")
print(f"hue range: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] degrees")
print(f"sat range: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
print(f"val range: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")
```Hue 以度数、饱和度和明度的形式在 [0, 1] 范围内表示。这与 OpenCV 的 `hsv_full` 惯例一致。

### 第 4 步：归一化、标准化并反转它

从原始字节转换为预训练 ImageNet 模型所期望的精确张量，然后再转换回来。```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x

def deprocess_imagenet(chw_float32):
    x = chw_float32.transpose(1, 2, 0)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

x = preprocess_imagenet(arr)
print(f"preprocessed shape: {x.shape}     # (C, H, W)")
print(f"preprocessed dtype: {x.dtype}")
print(f"preprocessed mean per channel:  {x.mean(axis=(1, 2)).round(3)}")
print(f"preprocessed std  per channel:  {x.std(axis=(1, 2)).round(3)}")

roundtrip = deprocess_imagenet(x)
max_diff = np.abs(roundtrip.astype(int) - arr.astype(int)).max()
print(f"roundtrip max pixel diff: {max_diff}    # should be 0 or 1")
```每个通道的均值应接近于零，标准差应接近于一。预处理/去预处理对就是每个 torchvision `transforms.Normalize` 调用在内部实际执行的操作。

### 步骤 5：使用三种插值方法进行调整大小

在放大时比较最近邻、双线性以及双三次插值方法，以便能够观察到差异。```python
target = (arr.shape[0] * 3, arr.shape[1] * 3)

nearest = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.NEAREST))
bilinear = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.BILINEAR))
bicubic = np.asarray(Image.fromarray(arr).resize(target[::-1], Image.BICUBIC))

def local_roughness(x):
    gy = np.diff(x.astype(float), axis=0)
    gx = np.diff(x.astype(float), axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())

for name, out in [("nearest", nearest), ("bilinear", bilinear), ("bicubic", bicubic)]:
    print(f"{name:>8}  shape={out.shape}  roughness={local_roughness(out):6.2f}")
```Nearest 在粗糙度上得分最高，因为它保留了硬边缘。Bilinear 是最平滑的。Bicubic 处于两者之间，在不出现阶梯状伪影的情况下保留了感知的锐度。

## 使用它

`torchvision.transforms` 将以上所有内容整合到一个可组合的管道中。下面的代码完全重现了 `preprocess_imagenet` 的功能，同时还包括调整大小和裁剪。```python
import torch
from torchvision import transforms
from PIL import Image

img = Image.fromarray(synthetic_rgb(256, 256))

pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

x = pipeline(img)
print(f"tensor type:  {type(x).__name__}")
print(f"tensor dtype: {x.dtype}")
print(f"tensor shape: {tuple(x.shape)}      # (C, H, W)")
print(f"per-channel mean: {x.mean(dim=(1, 2)).tolist()}")
print(f"per-channel std:  {x.std(dim=(1, 2)).tolist()}")

batch = x.unsqueeze(0)
print(f"\nbatched shape: {tuple(batch.shape)}   # (N, C, H, W) — ready for a model")
```四个步骤，严格按照以下顺序：`Resize(256)` 将较短的边缩放至 256；`CenterCrop(224)` 从中间截取一个 224x224 的区域；`ToTensor()` 除以 255 并将 HWC 转换为 CHW；`Normalize` 减去 ImageNet 的均值并除以标准差。反转这个顺序会悄无声息地改变传入模型的内容。

## 发布它

这节课将产生：

- `outputs/prompt-vision-preprocessing-audit.md` — 一个提示，它可以将任何模型卡或数据集卡转换为团队必须遵守的精确预处理不变性的检查表。
- `outputs/skill-image-tensor-inspector.md` — 一种技能，给定任何图像形状的张量或数组，可以报告其数据类型、布局、范围，以及它看起来是原始的、归一化的还是标准化的。

## 练习

1. **(简单)** 使用 OpenCV (`cv2.imread`) 和 Pillow 加载一个 JPEG 图像。打印两个形状和 `(0, 0)` 处的像素。解释通道顺序的差异，然后写一个一行转换，使 OpenCV 数组与 Pillow 的一致。
2. **(中等)** 编写 `standardize(img, mean, std)` 及其逆函数，使它们一起通过 `roundtrip_max_diff <= 1` 测试，适用于任何 uint8 图像。你的函数必须能够在 HWC 的单个图像和 NCHW 的批量上以相同的调用方式运行。
3. **(困难)** 取一个三通道的 ImageNet 标准化张量，将其通过一个 1x1 卷积层，该层学习将 RGB 混合为单个灰度通道的加权组合。将权重初始化为 `[0.299, 0.587, 0.114]`，将其冻结，并验证输出与你手动计算的 `rgb_to_grayscale` 在浮点误差范围内是否一致。还有哪些其他的经典颜色空间转换可以写成 1x1 卷积？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 像素 | “一个彩色的方块” | 一个网格位置上的光强度样本——三个数字表示颜色，一个数字表示灰度 |
| 通道 | “颜色” | 图像张量中堆叠的并行空间网格之一；在 HWC 中是最后一个轴，在 CHW 中是第一个轴 |
| HWC / CHW | “形状” | 图像张量的轴顺序；磁盘和 PIL 使用 HWC，PyTorch 和 cuDNN 使用 CHW |
| 归一化 | “缩放图像” | 除以 255 使像素位于 [0, 1] 范围内——这是必要的但不充分 |
| 标准化 | “零中心” | 按通道减去均值并除以标准差，使输入分布与模型训练时的分布匹配 |
| 灰度转换 | “平均通道” | 一个加权和，系数为 0.299/0.587/0.114，与人类亮度感知匹配 |
| 插值 | “如何调整大小选取像素” | 当新网格与旧网格不匹配时决定输出值的规则——标签使用最近邻，训练使用双线性，显示使用双三次 |
| 宽高比 | “宽除以高” | 区分“调整大小并填充”和“调整大小并拉伸”的比例 |

## 进一步阅读

- [Charles Poynton — 颜色空间导览](https://poynton.ca/PDFs/Guided_tour.pdf) — 最清晰的技术讲解，解释为什么有这么多颜色空间以及何时使用每个颜色空间
- [PyTorch Vision Transforms 文档](https://pytorch.org/vision/stable/transforms.html) — 你在生产中实际组合的完整变换流水线
- [JPEG 工作原理 (Colt McAnlis)](https://www.youtube.com/watch?v=F1kYBnY6mwg) — 对色度子采样、DCT 以及为什么 JPEG 编码 YCbCr 而非 RGB 的清晰视觉导览
- [ImageNet 预处理规范 (torchvision 模型)](https://pytorch.org/vision/stable/models.html) — `mean=[0.485, 0.456, 0.406]` 的权威来源，以及为什么动物园里的每个模型都期望它
