# Stable Diffusion — 架构与微调

> Stable Diffusion 是一种在预训练 VAE 的潜在空间中运行的 DDPM，通过交叉注意力进行文本条件化，使用快速确定性 ODE 求解器进行采样，并通过无分类器引导进行控制。

**类型:** 学习 + 使用
**语言:** Python
**先决条件:** 第四阶段第 10 课（扩散），第七阶段第 02 课（自注意力）
**时间:** ~75 分钟

## 学习目标

- 追踪 Stable Diffusion 管道的五个部分：VAE、文本编码器、U-Net、调度器、安全检查器 —— 以及它们各自的实际功能
- 解释潜在扩散以及为什么在 4x64x64 的潜在空间（而不是 3x512x512 的图像）中进行训练，可以在不损失质量的情况下将计算量减少 48 倍
- 使用 `diffusers` 生成图像，运行图像到图像，修复图像，以及 ControlNet 引导的生成
- 在一个小的自定义数据集上使用 LoRA 对 Stable Diffusion 进行微调，并在推理时加载 LoRA 适配器

## 问题

直接在 512x512 RGB 图像上训练 DDPM 是昂贵的。每个训练步骤都要通过一个 U-Net 进行反向传播，该 U-Net 接收到 3x512x512 = 786,432 个输入值，而采样则需要通过同一个 U-Net 进行 50 多次前向传递。在 Stable Diffusion 1.5（2022 年发布）的质量水平上，像素空间扩散需要大约 256 个 GPU 月的训练时间，并且在消费级 GPU 上每张图像需要 10-30 秒。

使开放权重的文本到图像生成变得实际的技巧是 **潜在扩散**（Rombach 等人，CVPR 2022）。训练一个 VAE，将 3x512x512 图像映射到 4x64x64 的潜在张量并返回，然后在该潜在空间中进行扩散。计算量减少 `(3*512*512)/(4*64*64) = 48x`。在相同的 GPU 上，采样时间从数十秒减少到不到两秒。

几乎所有现代图像生成模型 —— SDXL、SD3、FLUX、HunyuanDiT、Wan-Video —— 都是潜在扩散模型，对自动编码器、去噪器（U-Net 或 DiT）和文本条件化的不同变体。学习 Stable Diffusion 就是学习了模板。

## 概念

### 管道

```mermaid
flowchart LR
    TXT["Text prompt"] --> TE["Text encoder<br/>(CLIP-L or T5)"]
    TE --> CT["Text<br/>embedding"]

    NOISE["Noise<br/>4x64x64"] --> UNET["UNet<br/>(denoiser with<br/>cross-attention<br/>to text)"]
    CT --> UNET

    UNET --> SCHED["Scheduler<br/>(DPM-Solver++,<br/>Euler)"]
    SCHED --> LATENT["Clean latent<br/>4x64x64"]
    LATENT --> VAE["VAE decoder"]
    VAE --> IMG["512x512<br/>RGB image"]

    style TE fill:#dbeafe,stroke:#2563eb
    style UNET fill:#fef3c7,stroke:#d97706
    style SCHED fill:#fecaca,stroke:#dc2626
    style IMG fill:#dcfce7,stroke:#16a34a
```- **VAE** — 冻结的自编码器。编码器将图像转换为潜在表示（用于 img2img 和训练）。解码器将潜在表示转换回图像。
- **文本编码器** — CLIP 文本编码器（SD 1.x/2.x），CLIP-L + CLIP-G（SDXL），或 T5-XXL（SD3/FLUX）。生成一个 token 嵌入序列。
- **U-Net** — 去噪器。在每个分辨率层级中，具有从潜在表示到文本嵌入的交叉注意力层。
- **调度器** — 采样算法（DDIM、Euler、DPM-Solver++）。选择 sigma 值，将预测的噪声混合回潜在表示。
- **安全检查器** — 输出图像的可选 NSFW / 非法内容过滤器。

### 无分类器引导（CFG）

普通的文本条件设置为每个提示 `c` 学习 `epsilon_theta(x_t, t, c)`。CFG 训练相同的网络时，有 10% 的时间会丢弃 `c`（用空嵌入替换），从而得到一个可以同时预测条件和无条件噪声的单一模型。在推理时：

```
eps = eps_uncond + w * (eps_cond - eps_uncond)
```

`w` 是引导尺度。`w=0` 是无条件的，`w=1` 是普通条件，`w>1` 会推动输出更“依赖于提示”，但会牺牲多样性。SD 默认值为 `w=7.5`。

CFG 是文本到图像在生产质量上工作的关键原因。没有它，提示对输出的影响较弱；有了它，提示则占主导地位。

### 潜在空间几何

VAE 的 4 通道潜在空间不仅仅是一个压缩图像。它是一个流形，其中的算术大致对应语义编辑（提示工程和插值都存在于这里），同时也是扩散 U-Net 被训练花费全部建模预算的地方。解码一个随机的 4x64x64 潜在空间并不会生成看起来随机的图像，它会生成垃圾，因为只有潜在空间的一个特定子流形可以解码为有效的图像。

两个后果：

1. **Img2img** = 将图像编码为潜在空间，添加部分噪声，运行去噪器，解码。图像结构得以保留，因为编码是近似可逆的；内容则根据提示进行更改。
2. **Inpainting** = 与 img2img 相同，但去噪器只更新被遮罩的区域；未被遮罩的区域在编码的潜在空间中保持不变。

### U-Net 架构

SD 的 U-Net 是 Lesson 10 中 TinyUNet 的大型版本，增加了以下三个部分：

- **Transformer 块** 在每个空间分辨率下，包含自注意力 + 对文本嵌入的交叉注意力。
- **时间嵌入** 通过在正弦编码上使用 MLP 实现。
- **跳跃连接** 在编码器和解码器之间匹配的分辨率上。

SD 1.5 的总参数：约 8.6 亿。SDXL：约 26 亿。FLUX：约 120 亿。参数数量的激增主要发生在注意力层。

### LoRA 微调

对 Stable Diffusion 进行完整的微调需要 20+ GB 的 VRAM 并更新 8.6 亿个参数。LoRA（低秩适应）保持基础模型冻结，并将小的秩分解矩阵注入注意力层。SD 的一个 LoRA 适配器通常为 10-50 MB，在单个消费级 GPU 上训练需要 10-60 分钟，并在推理时作为即插即用的修改加载。

```
Original: W_q : (d_in, d_out)   frozen
LoRA:     W_q + alpha * (A @ B)   where A : (d_in, r), B : (r, d_out)

r is typically 4-32.
```LoRA 是几乎所有社区微调模型所采用的分布式方式。CivitAI 和 Hugging Face 主机上托管了数以百万计的此类模型。

### 你会看到的调度器

- **DDIM** — 确定性，约 50 步，简单。
- **Euler ancestral** — 随机性，30-50 步，生成的样本稍微更具创意。
- **DPM-Solver++ 2M Karras** — 确定性，20-30 步，生产环境默认选择。
- **LCM / TCD / Turbo** — 一致性模型和蒸馏变体；以牺牲部分质量为代价，仅需 1-4 步。

在 `diffusers` 中切换调度器只需要一行代码的更改，有时可以在不进行任何重新训练的情况下解决样本问题。

## 构建它

本课程使用 `diffusers` 进行端到端处理，而不是从零开始重建 Stable Diffusion。如果你需要重新构建的部分（VAE、文本编码器、U-Net、调度器）是其他课程的主题；在这里，目标是熟练掌握生产 API。

### 第一步：文本到图像

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="a dog riding a skateboard in tokyo, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("dog.png")
```

`float16` 在不损失可见质量的情况下将 VRAM 减半。使用默认的 DPM-Solver++ 的 `num_inference_steps=25` 与使用 DDIM 的 `num_inference_steps=50` 相匹配。

### 步骤 2：更换调度器

```python
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

调度器状态与 U-Net 权重是解耦的。你可以在 DDPM 上进行训练，并使用任何调度器进行采样。

### 步骤 3：图像到图像

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("dog.png").convert("RGB").resize((512, 512))
out = img2img(
    prompt="a dog riding a skateboard, oil painting",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]
```

`strength` 是在去噪之前要添加的噪声量（0.0 = 保持不变，1.0 = 完全再生）。0.5-0.7 是风格迁移的标准范围。

### 步骤 4：修复绘画

```python
from diffusers import StableDiffusionInpaintPipeline

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("dog.png").convert("RGB").resize((512, 512))
mask = Image.open("dog_mask.png").convert("L").resize((512, 512))

out = inpaint(
    prompt="a cat",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
).images[0]
```

掩膜中的白色像素是要再生的区域。黑色像素是保留的。

### 步骤 5：LoRA 加载

```python
pipe.load_lora_weights("sayakpaul/sd-lora-ghibli")
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a village square in ghibli style").images[0]
```

`lora_scale` 控制强度；0.0 = 无效果，1.0 = 完全生效。`fuse_lora` 将适配器烘焙到权重中以提高速度，但会阻止切换。在加载不同的适配器之前，请调用 `pipe.unfuse_lora()`。

### 步骤 6：LoRA 训练（草图）

真实的 LoRA 训练位于 `peft` 或 `diffusers.training`。概要如下：

```python
# Pseudocode
for step, batch in enumerate(dataloader):
    images, prompts = batch
    latents = vae.encode(images).latent_dist.sample() * 0.18215

    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    text_emb = text_encoder(tokenizer(prompts))

    pred_noise = unet(noisy_latents, t, text_emb)  # LoRA weights injected here

    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
```

仅 LoRA 矩阵接收梯度；基础 U-Net、VAE 和文本编码器被冻结。使用批量大小为 1 和梯度检查点，这可以适应 8 GB 的 VRAM。

## 使用它

在生产环境中，你实际做出的决策：

- **模型系列**：开源社区微调使用 SD 1.5，更高保真度使用 SDXL，最先进的且有严格许可要求使用 SD3 / FLUX。
- **调度器**：DPM-Solver++ 2M Karras 用于 20-30 步，延迟低于 1 秒时使用 LCM-LoRA。
- **精度**：在 4080/4090 上使用 `float16`，在 A100 及更新的设备上使用 `bfloat16`，当 VRAM 紧张时使用 `int8`（通过 `bitsandbytes` 或 `compel`）。
- **条件**：纯文本有效；为了更强的控制，可在基础流程上添加 ControlNet（canny、深度、姿态）。

对于批量生成，`AUTO1111` / `ComfyUI` 是社区工具；对于生产 API，使用 `diffusers` + `accelerate` 或 `optimum-nvidia` 并配合 TensorRT 编译。

## 发布它

本课将产生：

- `outputs/prompt-sd-pipeline-planner.md` — 一个提示，根据延迟预算、保真度目标和许可约束选择 SD 1.5 / SDXL / SD3 / FLUX 加上调度器和精度。
- `outputs/skill-lora-training-setup.md` — 一项技能，为自定义数据集编写完整的 LoRA 训练配置，包括标题、等级、批量大小和学习率。

## 练习

1. **(简单)** 使用 `guidance_scale` 在 `[1, 3, 5, 7.5, 10, 15]` 中生成相同的提示。描述图像如何变化。在什么引导值下会出现伪影？
2. **(中等)** 拿任何真实照片，通过 `StableDiffusionImg2ImgPipeline` 在 `strength` 的 `[0.2, 0.4, 0.6, 0.8, 1.0]` 中运行。哪个强度可以保持构图同时改变风格？为什么 1.0 完全忽略输入？
3. **(困难)** 在 10-20 张单个主体（宠物、标志、角色）的图像上训练一个 LoRA，并生成包含该主体的新场景。报告产生最佳身份保留而不过度拟合输入图像的 LoRA 等级和训练步骤。

## 关键术语

| 术语 | 人们所说 | 实际含义 |
|------|----------------|--------------|
| 潜在扩散 | “在潜在空间中扩散” | 在 VAE 潜在空间（4x64x64）而不是像素空间（3x512x512）中运行整个 DDPM；计算节省 48 倍 |
| VAE 缩放因子 | “0.18215” | 一个常数，用于重新缩放 VAE 的原始潜在变量，使其方差约为单位；硬编码在每个 SD 流程中 |
| 无分类器引导 | “CFG” | 混合条件和无条件噪声预测；影响推断的最重要参数 |
| 调度器 | “采样器” | 将噪声和模型预测转化为去噪潜在轨迹的算法 |
| LoRA | “低秩适配器” | 一组小秩分解矩阵，用于微调注意力层而无需修改基础权重 |
| 跨注意力 | “文本-图像注意力” | 潜在标记对文本标记的注意力；在每个 U-Net 层注入提示信息 |
| ControlNet | “结构条件” | 一个单独训练的适配器，通过额外输入（canny、深度、姿态、分割）引导 SD |
| DPM-Solver++ | “默认调度器” | 二阶确定性 ODE 求解器；在 2026 年，低步数（20-30）时质量最佳 |

## 进一步阅读

- [使用潜在扩散进行高分辨率图像合成（Rombach 等，2022）](https://arxiv.org/abs/2112.10752) — Stable Diffusion 论文；包括所有证明设计的消融实验
- [无分类器扩散引导（Ho & Salimans，2022）](https://arxiv.org/abs/2207.12598) — CFG 论文
- [LoRA：大型语言模型的低秩适配（Hu 等，2021）](https://arxiv.org/abs/2106.09685) — LoRA 最初用于 NLP；几乎没有修改就转移到了 SD
- [diffusers 文档](https://huggingface.co/docs/diffusers) — 每个 SD / SDXL / SD3 / FLUX 流程的参考
