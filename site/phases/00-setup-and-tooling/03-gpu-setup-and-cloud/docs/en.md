# GPU 配置与云端环境

> 从驱动安装到云端实例配额。配置 NVIDIA CUDA、显存分析以及高效的云端计算环境。

**Type:** 学习
**Languages:** Bash, Python
**Prerequisites:** 开发环境搭建
**Time:** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 使用 T4 GPU 配置 Google Colab 以进行免费的云端实验
- 对 CPU 与 GPU 上的矩阵乘法进行基准测试并测量加速比
- 使用 fp16 经验法则估算可适配 VRAM 的最大模型规模

## The Problem

第 1-3 阶段的大部分课程在 CPU 上运行良好。但一旦开始训练 CNNs、transformers 或 LLMs（第 4+ 阶段），就需要 GPU 加速。在 CPU 上需要 8 小时的训练，在 GPU 上仅需 10 分钟。

你有三个选择：本地 GPU、云 GPU 或 Google Colab（免费）。

## The Concept```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```## 构建它

### 选项 1：本地 NVIDIA GPU

检查你是否拥有一个：

 /no_think

<>

## 构建它

### 选项 1：本地 NVIDIA GPU

检查你是否拥有一个：

 /no_think```bash
nvidia-smi
```安装带 CUDA 的 PyTorch:```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

可直接将本课程的笔记本上传至 Colab。

### 选项 3：云 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

 /```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```### 没有GPU？没问题。

大多数课程可以在CPU上运行。需要GPU的课程会明确说明并包含Colab链接。```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```## 构建它：GPU 与 CPU 性能测试```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```## 练习

1. 运行上述基准测试并比较CPU与GPU的耗时
2. 如果没有GPU，使用Google Colab运行并进行比较
3. 检查你的GPU内存大小，并估算可容纳的最大模型规模（经验法则：fp16每个参数需要2字节）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| CUDA | "GPU编程" | NVIDIA的并行计算平台，允许在GPU上运行代码 |
| VRAM | "GPU内存" | GPU上的显存，独立于系统内存。限制模型规模。 |
| fp16 | "半精度" | 16位浮点数，相比fp32精度损失很小但内存占用减少一半 |
| Tensor Core | "快速矩阵硬件" | 专用于矩阵乘法的GPU核心，速度比普通核心快4-8倍 |
