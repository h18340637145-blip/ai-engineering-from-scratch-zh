# 实时视觉 — 边缘部署

> 边缘推理是一门将准确率为90%的模型在具有2GB内存的设备上以每秒30帧的速度运行的学科。每增加一个百分点的准确率，就要以毫秒级的延迟为代价。

**类型:** 学习 + 构建
**语言:** Python
**先决条件:** 第4阶段第4课（图像分类），第10阶段第11课（量化）
**时间:** ~75分钟

## 学习目标

- 测量任何PyTorch模型的推理延迟、峰值内存和吞吐量，并理解FLOPs / 参数 / 延迟之间的权衡
- 使用PyTorch的训练后量化将视觉模型量化到INT8，并验证准确率损失小于1%
- 导出到ONNX并使用ONNX Runtime或TensorRT进行编译；列举最常见的三种导出失败情况及其修复方法
- 解释在边缘约束条件下选择MobileNetV3、EfficientNet-Lite、ConvNeXt-Tiny或MobileViT的时机

## 问题

训练时间的视觉模型是一个浮点怪物。1亿个参数，每次前向传递有10GFLOPs，2GB的VRAM。这些都无法适应手机、汽车的娱乐信息系统、工业相机或无人机。部署视觉系统意味着将相同的预测结果压缩到一个只有原预算1/100的预算中。

三个旋钮完成了大部分工作：模型选择（使用相同配方的更小架构）、量化（使用INT8而不是FP32）以及推理运行时（ONNX Runtime、TensorRT、Core ML、TFLite）。正确使用它们的差异在于，一个演示程序可以在工作站上运行，而一个产品可以部署在30美元的摄像头模块上。

本课首先建立测量学科（你无法优化你无法测量的东西），然后逐步讲解这三个旋钮。目标不是学习每一个边缘运行时，而是了解有哪些杠杆可用，并知道如何验证每个杠杆是否达到预期效果。

## 概念

### 三个预算

```mermaid
flowchart LR
    M["Model"] --> LAT["Latency<br/>ms per image"]
    M --> MEM["Memory<br/>peak MB"]
    M --> PWR["Power<br/>mJ per inference"]

    LAT --> SHIP["Ship / no-ship<br/>decision"]
    MEM --> SHIP
    PWR --> SHIP

    style LAT fill:#fecaca,stroke:#dc2626
    style MEM fill:#fef3c7,stroke:#d97706
    style PWR fill:#dbeafe,stroke:#2563eb
```- **延迟**: p50、p95、p99。仅使用 p50 的平均值会隐藏对实时系统至关重要的尾部行为。
- **峰值内存**: 设备曾经看到的最大值，而不是稳态平均值。这很重要，因为嵌入式目标上内存不足（OOM）是致命的。
- **功耗/能量**: 电池供电设备上每次推理的毫焦耳。通常由 CPU/GPU 利用率 * 时间来代理。

一个由 (模型, 延迟, 内存, 准确率) 组成的表格是边缘决策的依据。每个单元格都在目标设备上测量，而不是在工作站上测量。

### 测量规范

每个边缘性能配置文件都应该遵循的三条规则：

1. **预热** 模型，在测量之前用 5-10 次虚拟的前向传递进行预热。冷缓存和即时编译会产生不具代表性的首次数值。
2. **同步** GPU 工作负载，在定时块前后与 `torch.cuda.synchronize()` 同步。没有这个你测量的是内核调度，而不是内核执行。
3. **固定输入大小** 为生产分辨率。224x224 的延迟不代表 512x512 的延迟。

### FLOPs 作为代理

FLOPs（每次推理的浮点运算）是一个便宜且与设备无关的延迟代理。对于架构比较很有用，但作为绝对时间墙钟是误导性的。一个 FLOPs 多出 10% 的模型在实践中可能快 2 倍，因为它使用了硬件友好的操作（深度可分离卷积编译得很好，而大型 7x7 卷积则不行）。

规则：用 FLOPs 进行架构搜索，用设备上的延迟进行部署决策。

### 量化（一段话）

将 FP32 权重和激活替换为 INT8。模型大小减少 4 倍，内存带宽减少 4 倍，在具有 INT8 内核的硬件上计算量减少 2-4 倍（所有现代移动 SoC，所有带有 Tensor Cores 的 NVIDIA GPU）。在视觉任务上，使用训练后静态量化通常会损失 0.1-1 个百分点的准确率。

类型：

- **动态** — 将权重量化为 INT8，激活计算在 FP 中。容易实现，加速幅度小。
- **静态（训练后）** — 在小的校准集上量化权重并校准激活范围。比动态快很多。
- **量化感知训练（QAT）** — 在训练过程中模拟量化，使模型围绕量化进行学习。准确率最佳，需要有标签的数据。

在视觉任务中，训练后静态量化以 5% 的努力获得 95% 的收益。只有当训练后量化（PTQ）的准确率损失无法接受时才使用 QAT。

### 剪枝与知识蒸馏

- **剪枝** — 移除不重要的权重（基于幅度）或通道（结构化）。在参数过多的模型上效果良好；对已经紧凑的架构用处较小。
- **知识蒸馏** — 训练一个小的学生模型来模仿大老师的 logit。通常可以恢复因模型缩小而损失的大部分准确率。是生产边缘模型的标准做法。

### 推理运行时

- **PyTorch eager** — 慢，不适用于部署。仅供开发使用。
- **TorchScript** — 旧版。已被 `torch.compile` 和 ONNX 导出取代。
- **ONNX Runtime** — 中立的运行时。CPU、CUDA、CoreML、TensorRT、OpenVINO 都有 ONNX 提供商。从此处开始。
- **TensorRT** — NVIDIA 的编译器。在 NVIDIA GPU（工作站和 Jetson）上延迟最佳。可以与 ONNX Runtime 集成或单独使用。
- **Core ML** — Apple 的 iOS/macOS 运行时。需要 `.mlmodel` 或 `.mlpackage`。
- **TFLite** — Google 的 Android/ARM 运行时。需要 `.tflite`。
- **OpenVINO** — Intel 的 CPU/VPU 运行时。需要 `.xml` + `.bin`。

实践：导出 PyTorch -> ONNX -> 为目标选择运行时。ONNX 是通用语言。

### 边缘架构选择器

| 预算       | 模型                  | 原因                                                         |
|------------|-----------------------|--------------------------------------------------------------|
| < 3M 参数   | MobileNetV3-Small     | 无处不兼容，良好的基线                                       |
| 3-10M      | EfficientNet-Lite-B0  | TFLite 上每参数的最佳准确率                                  |
| 10-20M     | ConvNeXt-Tiny         | 每参数最佳准确率，对 CPU 友好                                |
| 20-30M     | MobileViT-S 或 EfficientViT | 带有 ImageNet 准确率的 Transformer                          |
| 30-80M     | Swin-V2-Tiny          | 如果堆栈支持窗口注意力                                       |

除非有特殊原因，否则将所有这些都量化为 INT8。

```figure
cnn-param-count
```

## 构建它

### 步骤 1：正确测量延迟

```python
import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }
```

预热，同步，使用 `time.perf_counter()`。报告百分位数，而不仅仅是平均值。

### 步骤 2：参数和 FLOP 数量

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    """
    Rough FLOP count for a conv/linear-only model. For production use `fvcore` or `ptflops`.
    """
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total
```

在实际项目中使用 `fvcore.nn.FlopCountAnalysis` 或 `ptflops`；它们能正确处理每种模块类型。

### 第 3 步：训练后静态量化

```python
def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model
```

三个步骤：配置、准备（插入观察者）、使用真实数据校准、转换（融合 + 量化）。需要将模型进行融合（`Conv -> BN -> ReLU` -> `ConvBnReLU`），这由 `torch.ao.quantization.fuse_modules` 处理。

### 步骤 4：导出到 ONNX

```python
def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path
```

`opset_version=17` 是 2026 年的安全默认值。`dynamic_axes` 让你能够使用任意批量大小运行 ONNX 模型。

### 步骤 5：基准测试并比较制度

```python
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms  p95={lat_fp32['p95_ms']:.2f}ms")
```

对 `resnet50`、`efficientnet_v2_s` 和 `convnext_tiny` 运行相同的函数，你就能得到用于部署决策所需的比较表格。

## 使用它

生产环境的堆栈通常会走向以下三种路径之一：

- **Web / 无服务器**：PyTorch -> ONNX -> ONNX Runtime（CPU 或 CUDA 提供者）。最容易实现，对大多数情况来说已经足够。
- **NVIDIA 边缘设备（Jetson、GPU 服务器）**：PyTorch -> ONNX -> TensorRT。延迟最低，但需要最多的工程努力。
- **移动端**：PyTorch -> ONNX -> Core ML（iOS）或 TFLite（Android）。导出前进行量化。

为了进行测量，`torch-tb-profiler`、`nvprof` / `nsys` 以及 macOS 上的 Instruments 可以提供逐层的分析。`benchmark_app`（OpenVINO）和 `trtexec`（TensorRT）可以提供独立的 CLI 数字。

## 发布它

本课将产出以下内容：

- `outputs/prompt-edge-deployment-planner.md` — 一个提示，根据目标设备和延迟 SLA 选择主干、量化策略和运行时。
- `outputs/skill-latency-profiler.md` — 一项技能，可以编写完整的延迟基准测试脚本，包括预热、同步、百分位数和内存跟踪。

## 练习

1. **（简单）** 在 CPU 上测量 `resnet18`、`mobilenet_v3_small`、`efficientnet_v2_s` 和 `convnext_tiny` 在 224x224 分辨率下的 p50 延迟。报告表格并指出哪种架构具有最佳的准确率每毫秒。
2. **（中等）** 对 `mobilenet_v3_small` 应用训练后静态量化。在 CIFAR-10 或类似数据集的一个保留子集上，报告 FP32 与 INT8 的延迟和准确率损失。
3. **（困难）** 将 `convnext_tiny` 导出为 ONNX，通过 `onnxruntime` 使用 `CPUExecutionProvider` 运行，并与 PyTorch 的 eager 基线进行延迟对比。识别 ONNX Runtime 第一次比 PyTorch 快的层，并解释原因。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------|
| 延迟 | “多快” | 从输入到输出的时间；p50/p95/p99 百分位数，而不是平均值 |
| FLOPs | “模型大小” | 每次前向传播的浮点操作数；计算成本的粗略代理 |
| INT8 量化 | “8 位” | 将 FP32 权重/激活值替换为 8 位整数；体积减少约 4 倍，速度加快 2-4 倍 |
| PTQ | “训练后量化” | 在不重新训练的情况下对训练好的模型进行量化；简单，通常足够 |
| QAT | “量化感知训练” | 在训练过程中模拟量化；准确率最佳，需要标注数据 |
| ONNX | “中立格式” | 所有主流推理运行时都支持的模型交换格式 |
| TensorRT | “NVIDIA 编译器” | 将 ONNX 编译为 NVIDIA GPU 的优化引擎 |
| 知识蒸馏 | “教师 -> 学生” | 训练一个小模型来模仿大模型的 logit；可以恢复大部分损失的准确率 |

## 进一步阅读

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) — 高效架构的复合缩放
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244) — 使用 h-swish 和 squeeze-excite 的以移动端为主的架构
- [TensorRT 优化实践指南 (NVIDIA)](https://developer.nvidia.com/blog/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/) — 如何真正获得论文中的吞吐量数字
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/) — 量化、图优化、提供者选择
