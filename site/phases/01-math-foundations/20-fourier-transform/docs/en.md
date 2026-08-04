# 傅里叶变换与频域分析

> 从时域信号到频域谱图。掌握离散傅里叶变换（DFT）、FFT、卷积定理与梅尔频谱。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 19 (复数与旋转位置编码)
**Time:** ~45 分钟

## Learning Objectives

- 从头实现 DFT，并与 O(N log N) 的 Cooley-Tukey FFT 进行验证
- 解释频率系数：从信号中提取幅度、相位和功率谱
- 应用卷积定理通过 FFT 乘法执行卷积
- 将傅里叶频率分解与 Transformer 位置编码和 CNN 卷积层联系起来

## The Problem

音频记录是一段时间内的压力测量序列。股票价格是一段时间内每天的值序列。图像是一块空间上的像素强度网格。所有这些都是时域（或空间域）中的数据。你看到的是随着某个索引变化的值。

但许多模式在时域中是不可见的。这个音频信号是一个纯音还是和弦？这个股票价格是否有一个每周周期？这张图像是否具有重复的纹理？这些问题与频率内容有关，而时域隐藏了这些信息。

傅里叶变换将数据从时域转换到频域。它接受一个信号，并将其分解为不同频率的正弦波。每个正弦波都有一个幅度（强度）和一个相位（起始位置）。傅里叶变换会告诉你这两个信息。

这在机器学习中很重要，因为频域思维无处不在。卷积神经网络执行卷积，这在频域中是乘法。Transformer 的位置编码使用频率分解来表示位置。音频模型（语音识别、音乐生成）在频谱图上操作，这些是声音的频率表示。时间序列模型寻找周期性模式。理解傅里叶变换会为你提供与所有这些工作的词汇。

## The Concept

### DFT 的定义

给定 N 个样本 x[0], x[1], ..., x[N-1]，离散傅里叶变换（DFT）产生 N 个频率系数 X[0], X[1], ..., X[N-1]：

$$ X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N} $$```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1
```每个 X[k] 都是一个复数。它的模 |X[k]| 告诉你频率 k 的幅度。它的相位角 phase(X[k]) 告诉你该频率的相位偏移。

关键见解：`e^(-2*pi*i*k*n/N)` 是一个频率为 k 的旋转相量。DFT 计算信号与每个 N 个等间距频率之间的相关性。如果信号包含频率 k 的能量，相关性就很大。如果没有，它就接近于零。

### 每个系数的含义

**X[0]：直流分量。** 这是所有样本的总和——与平均值成比例。它表示信号的恒定（零频率）偏移。```
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples
```**X[k] for 1 <= k <= N/2: 正频率。** X[k] 表示每 N 个样本 k 个周期的频率。更高的 k 值意味着更高的频率（更快的振荡）。

**X[N/2]: 尼奎斯特频率。** 用 N 个样本可以表示的最高频率。超过这个频率后，会出现混叠现象——高频伪装成低频。

**X[k] for N/2 < k < N: 负频率。** 对于实值信号，X[N-k] = conj(X[k])。负频率是正频率的镜像。这就是为什么有用的信息在前 N/2 + 1 个系数中。

### 逆 DFT

逆 DFT 从频率系数重建原始信号：```
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1
```与正向 DFT 的唯一不同之处在于：指数中的符号为正（而非负），并且有一个 1/N 的归一化因子。

逆向 DFT 是完美的重构。没有信息丢失。你可以从时域转换到频域，再返回时域而没有任何误差。DFT 是一种基变换——它用不同的坐标系统重新表达相同的信息。

### FFT：使其快速

如上所述的 DFT 是 O(N²) 的复杂度：为了计算每个 N 个输出系数，都需要对 N 个输入样本求和。当 N = 100 万时，这需要 10¹² 次操作。

快速傅里叶变换（FFT）可以在 O(N log N) 的复杂度下计算出同样的结果。当 N = 100 万时，这只需要大约 2000 万次操作，而不是 10000 亿次。这使得频域分析成为可能。

Cooley-Tukey 算法（最常见的 FFT 算法）通过分治法实现：

1. 将信号分成偶数索引和奇数索引的样本。
2. 递归地对每一半计算 DFT。
3. 使用“旋转因子” e^(-2*pi*i*k/N) 将两个半大小的 DFT 结合起来。```
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples
```对称性意味着每次递归做 O(N) 的工作，共有 log2(N) 层递归。总时间复杂度：O(N log N)。```mermaid
graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 samples"] -->|"split even/odd"| E["Even: x[0,2,4,6]"]
        X -->|"split even/odd"| O["Odd: x[1,3,5,7]"]
        E -->|"4-pt FFT"| EK["E[0..3]"]
        O -->|"4-pt FFT"| OK["O[0..3]"]
        EK -->|"combine with twiddle factors"| XK["X[0..7]"]
        OK -->|"combine with twiddle factors"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64 multiplications"]
        C2["FFT: O(N log N) = 24 multiplications"]
    end
```FFT 要求信号长度为 2 的幂。在实践中，信号会被零填充到下一个 2 的幂。

### 频谱分析

**功率谱** 是 |X[k]|² -- 每个频率系数的模平方。它显示每个频率处的能量大小。

**相位谱** 是 angle(X[k]) -- 每个频率的相位偏移。对于大多数分析任务，你关注的是功率谱，而忽略相位。```
Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)
```### 频率分辨率

DFT 的频率分辨率取决于样本数 N 和采样率 fs。```
Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)
```要分辨两个相近的频率，需要更多的样本。要捕捉高频信号，需要更高的采样率。

### 卷积定理

这是信号处理中最重要的结果之一，与卷积神经网络（CNNs）直接相关。

**时域中的卷积等于频域中的逐点相乘。**```
x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication
```为什么这很重要：

- 两个长度分别为 N 和 M 的信号直接进行卷积需要 O(N*M) 次操作。
- 基于 FFT 的卷积需要 O(N log N)：将两个信号都进行变换，相乘，再变换回来。
- 对于大尺寸的卷积核，FFT 卷积速度明显更快。
- 这正是具有大感受野的卷积层中发生的情况。

注意：DFT 计算的是循环卷积（信号会环绕）。对于线性卷积（无环绕），在计算前请将两个信号都填充零至长度 N + M - 1。```mermaid
graph LR
    subgraph "Time Domain"
        TA["Signal x[n]"] -->|"convolve (slow: O(NM))"| TC["Output y[n]"]
        TB["Filter h[n]"] -->|"convolve"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"multiply (fast: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"multiply"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC
```### 窗函数

DFT 假设信号是周期性的——它将 N 个样本视为一个无限重复信号的一个周期。如果信号的起始和结束值不相同，这会在边界处产生不连续性，表现为虚假的高频内容。这种现象称为频谱泄漏。

窗函数通过在计算 DFT 之前将信号在两端衰减至零，从而减少泄漏。

常用窗函数：

| 窗函数 | 形状 | 主瓣宽度 | 旁瓣电平 | 使用场景 |
|--------|-------|----------------|-----------------|----------|
| 矩形窗 | 平坦（无窗） | 最窄 | 最高 (-13 dB) | 当信号在 N 个样本中恰好是周期性时 |
| 汉宁窗 | 升余弦 | 中等 | 低 (-31 dB) | 通用频谱分析 |
| 汉明窗 | 改进余弦 | 中等 | 更低 (-42 dB) | 音频处理、语音分析 |
| 布莱克曼窗 | 三阶余弦 | 宽 | 非常低 (-58 dB) | 当需要关键的旁瓣抑制时 |```
Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```通过对信号在进行 DFT 之前逐元素地与窗口相乘来应用窗口：`X = DFT(x * w)`。

### DFT 属性

| 属性 | 时域 | 频域 |
|------|-----|------|
| 线性性 | a*x + b*y | a*X + b*Y |
| 时移 | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| 频移 | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| 卷积 | x * h | X * H（逐点） |
| 乘法 | x * h（逐点） | X * H（循环卷积，乘以 1/N） |
| 帕塞瓦尔定理 | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| 共轭对称性（实输入） | x[n] 实数 | X[k] = conj(X[N-k]) |

帕塞瓦尔定理指出，两个域中的总能量是相同的。能量在变换过程中得以保持。

### 与位置编码的联系

原始的 Transformer 使用正弦位置编码：

 /no_think

<>

The original Transformer uses sinusoidal positional encodings:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

where $ pos $ is the position in the sequence, $ i $ is the dimension, and $ d $ is the dimensionality of the model. These encodings are added to the input embeddings to provide the model with information about the position of the tokens in the sequence.

### Alternative positional encodings

Other approaches to positional encoding include:

- **Learned positional encodings**: These are learned during training, just like the other parameters of the model.
- **Segment embeddings**: These are used to distinguish between different segments of the input, such as the sentence and the question in a question-answering task.
- **Relative positional encodings**: These encode the relative position of tokens with respect to each other, rather than their absolute positions.

### Benefits of positional encodings

Positional encodings are important for the Transformer model because they provide the model with information about the order of the tokens in the sequence. Without positional encodings, the model would not be able to distinguish between different orders of the same tokens, which would make it difficult to process sequential data like text.

### Summary

Positional encodings are a crucial component of the Transformer model. They allow the model to process sequential data by providing information about the position of the tokens in the sequence. There are several different approaches to positional encodings, including sinusoidal positional encodings, learned positional encodings, segment embeddings, and relative positional encodings. Each of these approaches has its own advantages and disadvantages, and the choice of which to use depends on the specific task and the data being used.```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```每个维度对（2i, 2i+1）以不同的频率振荡。这些频率从高（维度0,1）到低（最后的维度）呈几何级数分布。这使得每个位置在所有频率带中都有唯一的模式——类似于傅里叶系数如何唯一标识一个信号。

这提供了以下关键特性：

- **唯一性：** 没有两个位置的编码是相同的。
- **有界值：** sin 和 cos 的值始终在 [-1, 1] 范围内。
- **相对位置：** 位置 p+k 的编码可以表示为位置 p 的编码的线性函数。模型可以学习关注相对位置。

### 与卷积神经网络（CNN）的联系

卷积层通过将学习到的滤波器（核）滑动到输入信号或图像上，对输入应用该滤波器。数学上，这被称为卷积操作。

根据卷积定理，这等价于以下步骤：
1. 对输入进行快速傅里叶变换（FFT）。
2. 对核进行快速傅里叶变换（FFT）。
3. 在频域中进行相乘。
4. 对结果进行逆快速傅里叶变换（IFFT）。

标准的卷积神经网络实现使用直接卷积（对于小的 3x3 核更高效）。但对于大核或全局卷积，基于 FFT 的方法显著更快。一些架构（如 FNet）完全用 FFT 替代注意力机制，实现 O(N log N) 的复杂度，而不是 O(N²) 的复杂度，从而达到具有竞争力的准确率。

### 频谱图和短时傅里叶变换（STFT）

一次 FFT 可以得到整个信号的频率内容，但无法提供这些频率出现的时间信息。一个啁啾信号（频率随时间增加的信号）和一个和弦（所有频率同时出现）可能具有相同的幅度谱。

短时傅里叶变换（STFT）通过在信号的重叠窗口上计算 FFT 来解决这个问题。结果是一个频谱图：一个二维表示，其中一轴是时间，另一轴是频率。每个点的强度显示了该时间点在该频率上的能量。```
STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram
```语谱图是音频机器学习模型的标准输入表示。语音识别模型（Whisper、DeepSpeech）使用的是梅尔语谱图（mel-spectrograms）——将频率映射到梅尔刻度的语谱图，这种映射方式更符合人类对音高的感知。

### 频率混叠（Aliasing）

如果信号中包含高于 fs/2（奈奎斯特频率）的频率，以 fs 的采样率进行采样会产生混叠的副本。以 100 Hz 的采样率对 90 Hz 的信号进行采样，会与 10 Hz 的信号看起来完全相同。仅从采样数据本身无法区分它们。```
Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.
```这就是为什么模数转换器包含抗混叠滤波器，用于在采样前去除高于奈奎斯特频率的频率。在机器学习中，当在没有适当的低通滤波的情况下对特征图进行下采样时，会出现混叠现象——一些架构通过使用抗混叠池化层来解决这个问题。

### 零填充不会提高分辨率

一个常见的误解：在进行FFT之前对信号进行零填充可以提高频率分辨率。实际上并非如此。零填充只是在现有的频率 bins 之间进行插值，使频谱看起来更平滑。但它无法揭示原始样本中没有的频率细节。

真正的频率分辨率只取决于观测时间 T = N / fs。为了分辨出间隔为 delta_f 的两个频率，你需要至少 T = 1 / delta_f 秒的数据。无论进行多少零填充，都无法改变这个基本限制。```figure
fourier-synthesis
```## 构建它

### 步骤 1：从零开始构建 DFT

O(N²) 的 DFT 可以直接从定义得出。```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```### 步骤 2：逆 DFT

相同结构，正指数，除以 N。```python
def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result
```### 步骤 3：FFT（Cooley-Tukey）

递归的 FFT 需要长度为 2 的幂。将其分为偶数和奇数部分，递归处理，再通过蝶形因子进行合并。```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result
```### 步骤 4：频谱分析辅助工具```python
def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]
```## 使用它

在实际工作中，请使用由高度优化的 C 库支持的 numpy 的 FFT。```python
import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]
```用于窗口处理和更高级的频谱分析：

```python
import numpy as np
from scipy import signal
``````python
from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)
```对于卷积：```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')
```对于频谱图：```python
from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2
```频谱图矩阵的形状为 (n_frequencies, n_time_frames)。每一列代表一个时间窗口的功率谱。这就是音频机器学习模型所使用的输入。

## 发布它

运行 `code/fourier.py` 以生成 `outputs/prompt-spectral-analyzer.md`。

## 练习

1. **纯音识别。** 创建一个未知频率（介于 1 和 50 Hz 之间）的单一正弦波信号，以 128 Hz 的采样率采样 1 秒。使用你的 DFT 来识别频率。验证答案是否匹配。现在添加标准差为 0.5 的高斯噪声并重复。噪声对频谱有什么影响？

2. **FFT 与 DFT 验证。** 生成一个长度为 64 的随机信号。计算 DFT（O(N^2)）和 FFT。验证所有系数在 1e-10 范围内是否匹配。在长度为 256、512、1024 和 2048 的信号上对这两个函数进行计时。绘制 DFT 时间与 FFT 时间的比率。

3. **通过示例证明卷积定理。** 创建信号 x = [1, 2, 3, 4, 0, 0, 0, 0] 和滤波器 h = [1, 1, 1, 0, 0, 0, 0, 0]。直接计算它们的循环卷积（嵌套循环）。然后通过 FFT 计算（变换、相乘、逆变换）。验证结果是否匹配。现在通过适当补零进行线性卷积。

4. **窗函数影响。** 创建一个由两个正弦波（10 Hz 和 12 Hz，非常接近）相加的信号。以 128 Hz 的采样率采样 1 秒。分别在无窗、汉宁窗和汉明窗下计算功率谱。哪种窗函数最易于区分两个峰值？为什么？

5. **位置编码分析。** 为 d_model = 128 和 max_pos = 512 生成正弦位置编码。对于每一对位置 (p1, p2)，计算它们编码的点积。显示点积仅依赖于 |p1 - p2|，而不是绝对位置。当距离增加时，点积会发生什么变化？

## 关键术语

| 术语 | 含义 |
|------|------|
| DFT（离散傅里叶变换） | 将 N 个时域样本转换为 N 个频域系数。每个系数是与该频率的复数正弦波的相关性 |
| FFT（快速傅里叶变换） | 一种 O(N log N) 算法，用于计算 DFT。Cooley-Tukey 算法递归地将偶数/奇数索引分开 |
| 逆 DFT | 从频域系数重建时域信号。与 DFT 公式相同，但指数符号反转并有 1/N 缩放 |
| 频率分量 | DFT 输出中的每个索引 k 代表频率 k*fs/N Hz。"分量"是离散的频率槽 |
| 直流分量 | X[0]，零频率系数。与信号均值成比例 |
| 奈奎斯特频率 | fs/2，采样率 fs 下可表示的最大频率。高于此频率的频率将产生混叠 |
| 功率谱 | \|X[k]\|^2，每个频率系数的平方模值。显示频率上的能量分布 |
| 相位谱 | angle(X[k])，每个频率分量的相位偏移。在分析中通常被忽略 |
| 谱泄漏 | 将非周期信号视为周期信号引起的虚假频率内容。通过加窗减少 |
| 窗函数 | 在进行 DFT 之前应用的衰减函数（如汉宁窗、汉明窗、布莱克曼窗），用于减少谱泄漏 |
| 旋转因子 | 用于 FFT 蝴蝶运算中组合子 DFT 的复指数 e^(-2*pi*i*k/N) |
| 卷积定理 | 时域中的卷积等于频域中的逐点相乘。是信号处理和卷积神经网络的基础 |
| 循环卷积 | 信号环绕的卷积。这是 DFT 自然计算的内容 |
| 线性卷积 | 没有环绕的标准卷积。通过在 DFT 前补零实现 |
| 帕塞瓦尔定理 | 总能量通过傅里叶变换保持不变。sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| 混叠 | 由于采样率不足，奈奎斯特频率以上的频率表现为较低频率 |

## 进一步阅读

- [Cooley & Tukey: 用于机器计算复傅里叶级数的算法 (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/) - 改变计算的原始 FFT 论文
- [3Blue1Brown: 但什么是傅里叶变换？](https://www.youtube.com/watch?v=spUNpyF58BY) - 傅里叶变换的最佳可视化介绍
- [Lee-Thorp 等人: FNet: 使用傅里叶变换混合标记 (2021)](https://arxiv.org/abs/2105.03824) - 在变换器中用 FFT 替换自注意力
- [Smith: 科学家和工程师的数字信号处理指南](http://www.dspguide.com/) - 免费在线教科书，深入讲解 FFT、加窗和频谱分析
- [Vaswani 等人: 注意力是全部你所需要的 (2017)](https://arxiv.org/abs/1706.03762) - 从傅里叶频率分解推导出的正弦位置编码
- [Radford 等人: Whisper (2022)](https://arxiv.org/abs/2212.04356) - 使用梅尔频谱图作为输入表示的语音识别
