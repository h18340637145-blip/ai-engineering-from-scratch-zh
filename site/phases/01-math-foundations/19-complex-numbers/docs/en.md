# 复数与旋转位置编码

> 欧拉公式与虚数维度。掌握复平面几何、共轭转置与 RoPE 旋转位置编码应用。

**Type:** 学习
**Language:** Python
**Prerequisites:** Phase 1, Lesson 02 (向量与矩阵运算)
**Time:** ~40 分钟

## 学习目标

- 在矩形形式和极坐标形式下执行复数运算（加、乘、除、共轭）
- 应用欧拉公式在复数指数和三角函数之间进行转换
- 使用复数单位根实现离散傅里叶变换
- 解释复数旋转如何在 RoPE 和变压器模型中的正弦位置编码中发挥作用

## 问题

你打开一篇关于傅里叶变换的论文，发现 `i` 无处不在。你查看变压器模型的位置编码，看到 `sin` 和 `cos` 在不同频率下出现——它们是复数指数的实部和虚部。你阅读量子计算的相关内容，发现所有内容都用复数向量空间来表达。

复数看起来很抽象。一个以 -1 的平方根为基础的数字系统感觉像是一个数学技巧。但这不是一个技巧。它是旋转和振荡的自然语言。每当下列情况发生时，复数都是合适的工具：某物旋转、振动或振荡。

如果不理解复数，你就无法理解离散傅里叶变换。你无法理解快速傅里叶变换（FFT）。你无法理解现代语言模型中 RoPE（旋转位置嵌入）的工作原理。你无法理解原始 Transformer 论文中的正弦位置编码为什么使用那些特定的频率。

本课从零开始构建复数运算，将其与几何联系起来，并向你展示复数在机器学习中确切的出现位置。

## 概念

### 什么是复数？

复数有两个部分：实部和虚部。```
z = a + bi

where:
  a is the real part
  b is the imaginary part
  i is the imaginary unit, defined by i^2 = -1
```就是这样。你将数轴扩展到一个平面。实数位于一个轴上，虚数位于另一个轴上。每一个复数都是这个平面上的一个点。

### 复数运算

**加法。** 将实部相加，将虚部相加。```
(a + bi) + (c + di) = (a + c) + (b + d)i

Example: (3 + 2i) + (1 + 4i) = 4 + 6i
```**乘法。** 使用分配律，并记住 i² = -1。```
(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

Example: (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                            = 3 + 14i - 8
                            = -5 + 14i
```**共轭。** 反转虚部的符号。```
conjugate of (a + bi) = a - bi
```复数与其共轭复数的乘积总是实数：```
(a + bi)(a - bi) = a^2 + b^2
```**除法。** 将分子和分母同时乘以分母的共轭。```
(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)
```这消除了分母中的虚部，给你一个干净的复数。

### 复数平面

复数平面将每个复数映射到一个二维点。水平轴是实轴，垂直轴是虚轴。```
z = 3 + 2i  corresponds to the point (3, 2)
z = -1 + 0i corresponds to the point (-1, 0) on the real axis
z = 0 + 4i  corresponds to the point (0, 4) on the imaginary axis
```复数同时表示原点处的一个点和一个向量。这种双重解释使得复数在几何中非常有用。

### 极坐标形式

平面上的任何一点都可以用它到原点的距离以及它与正实轴之间的角度来描述。```
z = r * (cos(theta) + i*sin(theta))

where:
  r = |z| = sqrt(a^2 + b^2)     (magnitude, or modulus)
  theta = atan2(b, a)             (phase, or argument)
```矩形形式（a + bi）适合加法。极坐标形式（r, theta）适合乘法。

**极坐标形式下的乘法。** 乘以模长，加上角度。```
z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))
```这就是为什么复数非常适合表示旋转。乘以一个模长为1的复数就是一个纯粹的旋转。

### 欧拉公式

复数指数与三角函数之间的桥梁：

$$```
e^(i*theta) = cos(theta) + i*sin(theta)
```这是本课最重要的公式。当 theta = pi 时：```
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

Therefore: e^(i*pi) + 1 = 0
```五个基本常数（e, i, pi, 1, 0）在一个方程中相互联系。

### 为什么欧拉公式对机器学习很重要

欧拉公式指出，当theta变化时，`e^(i*theta)`沿着单位圆运动。当theta = 0时，你位于(1, 0)。当theta = pi/2时，你位于(0, 1)。当theta = pi时，你位于(-1, 0)。当theta = 3*pi/2时，你位于(0, -1)。一次完整的旋转是theta = 2*pi。

这意味着复数指数是旋转。而旋转在信号处理和机器学习中无处不在。

### 与二维旋转的联系

将复数(x + yi)乘以e^(i*theta)会将点(x, y)绕原点旋转theta角度。```
Rotation via complex multiplication:
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

Rotation via matrix multiplication:
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]
```它们产生相同的结果。复数乘法就是二维旋转。旋转矩阵只是用矩阵表示的复数乘法。```mermaid
graph TD
    subgraph "Complex Multiplication = 2D Rotation"
        A["z = x + yi<br/>Point (x, y)"] -->|"multiply by e^(i*theta)"| B["z' = z * e^(i*theta)<br/>Point rotated by theta"]
    end
    subgraph "Equivalent Matrix Form"
        C["vector [x, y]"] -->|"multiply by rotation matrix"| D["[x cos theta - y sin theta,<br/> x sin theta + y cos theta]"]
    end
    B -.->|"same result"| D
```### 相量和旋转信号

一个复指数 e^(i*omega*t) 是一个以角频率 omega 绕单位圆旋转的点。随着 t 的增加，该点沿着圆周运动。

这个旋转点的实部是 cos(omega*t)。虚部是 sin(omega*t)。正弦信号是一个旋转复数的投影。```
e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

Real part:      cos(omega*t)    -- a cosine wave
Imaginary part: sin(omega*t)    -- a sine wave
```这是相量表示法。与其追踪一个波动的正弦波，不如追踪一个平稳旋转的箭头。相位变化变成了角度偏移。幅度变化变成了大小变化。信号的相加变成了向量相加。

### 单位根

N次单位根是在单位圆上等距分布的N个点：```
w_k = e^(2*pi*i*k/N)    for k = 0, 1, 2, ..., N-1
```对于 N = 4，根为：1, i, -1, -i（四个方位点）。
对于 N = 8，你会得到四个方位点加上四个对角线。

单位根是离散傅里叶变换的基础。DFT 将信号分解为这些 N 个等间距频率上的分量。

### 与 DFT 的联系

信号 x[0], x[1], ..., x[N-1] 的离散傅里叶变换为：```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
```每个 X[k] 衡量信号与单位根的第 k 个根的相关程度——即频率为 k 的复数正弦波。DFT 将信号分解为 N 个旋转的相量，并告诉你每个相量的幅度和相位。

### 为什么 i 不是虚数

“虚数”这个词是一个历史上的偶然。笛卡尔曾用这个词带有贬义。但 i 并不比人们最初拒绝负数时的负数更“虚”。负数回答的是“从 3 中减去什么数可以得到 5？”的问题；而虚数单位 i 回答的是“什么数平方后得到 -1？”的问题。

更有用的是：i 是一个 90 度旋转的操作符。将一个实数乘以 i 一次，就将它旋转 90 度到虚轴上。再乘以 i（i²），就再旋转 90 度——现在你指向的是负实轴方向。这就是为什么 i² = -1。这并不神秘，它只是由两个四分之一转构成的半转。

这就是为什么复数在工程中无处不在。任何涉及旋转的事物——电磁波、量子态、信号振荡、位置编码——都自然地用复数来描述。

### 复数指数与三角函数

在欧拉公式之前，工程师将信号写成 A*cos(omega*t + phi) 的形式——振幅 A，频率 omega，相位 phi。这虽然有效，但进行算术运算时非常繁琐。添加两个不同相位的余弦波需要使用三角恒等式。

使用复数指数后，同样的信号可以写成 A*e^(i*(omega*t + phi))。添加两个信号就只是添加两个复数。乘法（调制）只是幅度相乘和角度相加。相位偏移变为角度相加。频率偏移变为相量相乘。

整个信号处理领域都转向复数指数表示法，因为数学运算更清晰。所谓的“实信号”总是复数表示法的实部。虚部则作为会计记录被保留下来，使得所有代数运算自然地进行。

### 与 Transformer 的联系

**正弦位置编码**（原始 Transformer 论文）：

```python
def positional_encoding(position, dim):
    return torch.tensor([
        math.sin(position / (10000 ** (2 * i / dim))) for i in range(dim // 2)
    ] + [
        math.cos(position / (10000 ** (2 * i / dim))) for i in range(dim // 2)
    ])
``````
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```正弦和余弦对是不同频率的复数指数的实部和虚部。每个频率为编码位置提供了不同的“分辨率”。低频变化缓慢（粗略位置），高频变化迅速（精细位置）。它们共同为每个位置提供了一个独特的频率指纹。

**RoPE（旋转位置嵌入）** 进一步拓展了这一点。它明确地将查询和键向量与复数旋转矩阵相乘。两个标记之间的相对位置变成了旋转角度。注意力计算使用这些旋转后的向量，使模型通过复数乘法对相对位置变得敏感。

| 操作 | 代数形式 | 几何意义 |
|------|---|---|
| 加法 | (a+c) + (b+d)i | 平面中的向量加法 |
| 乘法 | (ac-bd) + (ad+bc)i | 旋转和缩放 |
| 共轭 | a - bi | 关于实轴反射 |
| 模长 | sqrt(a^2 + b^2) | 距离原点的距离 |
| 相位 | atan2(b, a) | 从正实轴的角度 |
| 除法 | 乘以共轭 | 反向旋转和重新缩放 |
| 幂运算 | r^n * e^(i*n*theta) | 旋转n次，按r^n缩放 |```mermaid
graph LR
    subgraph "Unit Circle"
        direction TB
        U1["e^(i*0) = 1"] -.-> U2["e^(i*pi/2) = i"]
        U2 -.-> U3["e^(i*pi) = -1"]
        U3 -.-> U4["e^(i*3pi/2) = -i"]
        U4 -.-> U1
    end
    subgraph "Applications"
        A1["Euler's formula:<br/>e^(i*theta) = cos + i*sin"]
        A2["DFT uses roots of unity:<br/>e^(2*pi*i*k/N)"]
        A3["RoPE uses rotation:<br/>q * e^(i*m*theta)"]
    end
    U1 --> A1
    U1 --> A2
    U1 --> A3
```

```figure
roots-of-unity
```## 构建它

### 第一步：复数类

构建一个支持算术运算、模长、相位以及矩形坐标和极坐标形式之间转换的复数类。```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)
```### 步骤 2：极坐标转换和欧拉公式```python
def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))
```验证：`euler(theta).magnitude()` 应该始终为 1.0。`euler(0)` 应该给出 (1, 0)。`euler(pi)` 应该给出 (-1, 0)。

### 步骤 3：旋转

将点 (x, y) 按角度 theta 旋转相当于一次复数乘法：```python
point = Complex(3, 4)
rotated = point * euler(math.pi / 4)
```幅度保持不变。只有角度发生变化。

### 步骤 4：从复数运算进行 DFT```python
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result
```这是 O(N²) 的 DFT。每个输出 X[k] 是信号样本乘以单位根的和。

### 步骤 5：逆 DFT

逆 DFT 从频谱重建原始信号。与正向 DFT 相比，唯一的变化是：在指数中取反号，并除以 N。```python
def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result
```这使你能够实现完美的重构。应用DFT，然后应用IDFT，你将能以机器精度恢复原始信号。没有信息丢失。

### 步骤6：单位根```python
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]
```验证两个性质：
- 每个根的模长正好是 1。
- 所有 N 个根的和为零（它们通过对称相互抵消）。

这些性质使得 DFT 可逆。单位根构成了频域的一个正交基。

## 使用它

Python 内置了对复数的支持。字面量 `j` 表示虚数单位。```python
z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(abs(z))

import cmath
print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))
```对于数组，numpy 本机支持复数：```python
import numpy as np

z = np.array([1+2j, 3+4j, 5+6j])
print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(128, d=1/128)
```## 发布它

运行 `code/complex_numbers.py` 以生成 `outputs/skill-complex-arithmetic.md`。

## 练习

1. **手动进行复数运算。** 计算 (2 + 3i) * (4 - i) 并使用代码验证结果。然后计算 (5 + 2i) / (1 - 3i)。在复数平面上绘制这两个结果，并确认乘法对第一个数进行了旋转和缩放。

2. **旋转序列。** 从点 (1, 0) 开始，将其乘以 e^(i*pi/6) 十二次。验证在进行十二次乘法后，你回到 (1, 0)。打印每一步的坐标，并确认它们描出了一个正十二边形。

3. **已知信号的 DFT。** 创建一个信号，该信号是 sin(2*pi*3*t) 和 0.5*sin(2*pi*7*t) 的和，并在 32 个点上采样。运行你的 DFT。验证幅度谱在频率 3 和 7 处有峰值，且频率 7 处的峰值高度是频率 3 处的一半。

4. **单位根的可视化。** 计算 8 次单位根。验证它们的和为零。验证将任意一个根乘以原根 e^(2*pi*i/8) 会得到下一个根。

5. **旋转矩阵等价性。** 对于 10 个随机角度和 10 个随机点，验证复数乘法与使用 2x2 旋转矩阵进行矩阵-向量乘法得到的结果是否相同。打印最大数值差异。

## 术语表

| 术语 | 含义 |
|------|------|
| 复数 | 一个数 a + bi，其中 a 是实部，b 是虚部，且 i^2 = -1 |
| 虚数单位 | 一个数 i，由 i^2 = -1 定义。在哲学意义上它不是虚的——它是一个旋转算子 |
| 复数平面 | 二维平面，其中 x 轴是实轴，y 轴是虚轴。也称为 Argand 平面 |
| 幅值（模） | 距离原点的距离：sqrt(a^2 + b^2)。写作 \|z\| |
| 相位（幅角） | 从正实轴的角度：atan2(b, a)。写作 arg(z) |
| 共轭 | 实轴的镜像：a + bi 的共轭是 a - bi |
| 极坐标形式 | 将 z 表示为 r * e^(i*theta) 而不是 a + bi。使乘法变得简单 |
| 欧拉公式 | e^(i*theta) = cos(theta) + i*sin(theta)。将指数函数与三角函数联系起来 |
| 相量 | 一个旋转的复数 e^(i*omega*t)，代表正弦信号 |
| 单位根 | N 个复数 e^(2*pi*i*k/N)（k = 0 到 N-1）。单位圆上等距分布的 N 个点 |
| DFT | 离散傅里叶变换。使用单位根将信号分解为复数正弦成分 |
| RoPE | 旋转位置嵌入。使用复数乘法在 Transformer 注意力中编码相对位置 |

## 进一步阅读

- [欧拉公式的视觉介绍](https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/) - 不使用复杂符号构建几何直觉
- [Su 等人：RoFormer (2021)](https://arxiv.org/abs/2104.09864) - 引入使用复数旋转的旋转位置嵌入的论文
- [Vaswani 等人：Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文，包含正弦位置编码
- [3Blue1Brown：欧拉公式与群论初步](https://www.youtube.com/watch?v=mvmuCPvRoWQ) - 可视化解释为什么 e^(i*pi) = -1
- [Needham：复数的视觉分析](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464) - 复数的最佳视觉化处理，充满几何洞察
- [Strang：线性代数导论，第 10 章](https://math.mit.edu/~gs/linearalgebra/) - 在线性代数和特征值的背景下介绍复数
