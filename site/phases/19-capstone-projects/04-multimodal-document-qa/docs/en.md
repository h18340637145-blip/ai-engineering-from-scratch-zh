# 第 04 章 — 多模态文档问答（以视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿，已经从 OCR 再转文本，转向了以视觉优先的晚交互。ColPali、ColQwen2.5 和 ColQwen3-omni 会把每一页 PDF 当作一张图像，用多向量晚交互来编码，让查询直接关注图像 patch。面对财报 10-K、科研论文和手写笔记，这种方式明显优于先 OCR 再转文本。把整条流水线在 1 万页上完整做出来，并和 OCR-then-text 做并排对比。

**Type:** Capstone
**Languages:** Python（流水线）、TypeScript（查看器 UI）
**Prerequisites:** 第 4 章（计算机视觉）、第 5 章（NLP）、第 7 章（Transformer）、第 11 章（LLM 工程）、第 12 章（多模态）、第 17 章（基础设施）
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 小时

## 问题

企业手里的 PDF 往往会把 OCR 流水线难倒：扫描版 10-K 里有旋转表格，科研论文里满是公式，图表只有作为图像才看得懂，还有手写批注。把这些内容当成纯文本处理，会丢掉一半信号。2026 年的答案，是在原始页面图像上做晚交互式多向量检索。ColPali（Illuin Tech）率先提出了这条路；ColQwen2.5-v0.2 和 ColQwen3-omni 又把准确率推得更高。在 ViDoRe v3 上，视觉优先检索比 OCR-then-text 的得分高出明显一截，而且在图表、表格和手写体上差距更大。

代价是存储和延迟。一个 ColQwen embedding 每页大约是 2048 个 patch 向量，而不是一个 1024 维的单向量。原始存储会迅速膨胀。DocPruner（2026）带来了 50% 的裁剪，同时几乎不损失准确率。你要索引 1 万页，测 ViDoRe v3 的 nDCG@5，保证答案在 2s 内返回，并和 OCR-then-text 基线直接对比。

## 概念

晚交互的意思是，每个查询 token 都会和每个 patch token 打分，然后把每个查询 token 的最大分数加总。这样就能得到很细粒度的匹配，而不必先压成一个 pooled vector。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）负责存每个 patch 的 embedding，并在检索时执行 MaxSim。

回答器是一个视觉语言模型，它会把查询和检索到的 top-k 页面图像一起输入，然后输出带有证据区域（bounding box 或页码引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记号，可以再接一个 OCR 回退（Nougat、dots.ocr）作为可选文本通道。

评估本质上是一个二维矩阵。一个轴是内容类型（纯文本段落、密集表格、柱状 / 折线图、手写笔记、公式）；另一个轴是检索方式（视觉优先晚交互、OCR-then-text、混合式）。每个格子都要给出 nDCG@5 和答案准确率。最终报告就是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack

- 页面渲染：PyMuPDF（fitz），180 DPI，统一为竖版
- 晚交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队版本）
- 索引：带多向量字段的 Vespa、Qdrant multi-vector，或带 MaxSim 的 AstraDB
- 裁剪：DocPruner 2026 策略（保留高方差 patch，在准确率损失小于 0.5% 的前提下压缩 50%）
- OCR 回退（公式 / 密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为回退
- 评测：ViDoRe v3 基准、用于多页推理的 M3DocVQA
- 查看器 UI：带证据区域 canvas 覆盖层的 Next.js 15

## Build It

1. **采集。** 遍历一个包含 10-K、科研论文和扫描文档的 1 万页 PDF 语料。把每页渲染成 1536x2048 的 PNG，并持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每一页图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个、维度 128 的 patch embedding。再用 DocPruner 保留信号最强的一半。写入 Vespa multi-vector 字段或 Qdrant multi-vector。

3. **查询。** 对每个新查询，用查询塔做 embedding（token 级 embedding）。再对索引运行 MaxSim：对每个查询 token，在页面 patch embedding 上取最大点积，再求和。返回 top-k 页面。

4. **合成答案。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词：“只使用提供的页面来回答。每条结论都用 (doc_id, page) 引用，并标明区域类型（figure、table、paragraph）。”

5. **证据区域。** 对答案做后处理，提取引用到的区域。如果 VLM 输出了 bounding box（Qwen3-VL 会这样），就在查看器里把它们画成覆盖层。

6. **OCR 回退。** 对于被判定为公式密集的页面（基于图像方差的启发式判断），运行 Nougat 或 dots.ocr，并把 OCR 文本作为额外通道与图像一起输入。

7. **评测。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在同一语料上跑 OCR-then-text 流水线，使用相同的答案器。输出一个内容类型 × 方案 的矩阵。

8. **UI。** 先做 Streamlit 原型；再做带逐页证据区域覆盖层的 Next.js 15 生产版查看器。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付

`outputs/skill-doc-qa.md` 描述的是最终交付物：一个以视觉优先为核心的多模态文档问答系统，针对特定语料调优，并在 ViDoRe v3 上和 OCR-then-text 基线做评估对比。

| 权重 | 标准 | 评测方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 相对 OCR-文本基线和公开排行榜的基准分数 |
| 20 | 证据区域落地 | 被引用区域里真正包含答案片段的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩率、索引 p95、答案 p95 |
| 20 | 多页推理 | 在一个人工标注的 100 题多页集合上的准确率 |
| 15 | 源文检查 UX | 查看器清晰度、覆盖层保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料上比较 ColQwen2.5-v0.2 和 ColQwen3-omni。哪类页面一个能答对、另一个会漏掉？给索引加一个“内容类别”标签，用来按类型路由。

2. 激进裁剪 embedding（75%、90%）。找出压缩悬崖：也就是 ViDoRe nDCG@5 掉到 OCR 基线以下的临界点。

3. 做一个混合方案：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，再用 cross-encoder 重排。这个混合方案能不能超过单独方案？在哪些场景最有帮助？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测准确率 / 成本曲线。

5. 加上手写笔记支持。渲染手写语料，用 ColQwen 做 embedding 并测检索效果，再和手写 OCR 流水线比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 晚交互 | “ColPali 风格检索” | 查询 token 分别和页面 patch 打分，最后由 MaxSim 聚合 |
| 多向量 | “逐 patch embedding” | 每个文档有很多向量，而不是一个 pooled vector |
| MaxSim | “晚交互打分” | 对每个查询 token，在文档向量上取最大相似度再求和 |
| DocPruner | “Patch 压缩” | 2026 年的裁剪方法，在几乎不损失准确率的前提下保留 50% 的 patch |
| ViDoRe v3 | “文档检索基准” | 2026 年衡量视觉文档检索的标准基准 |
| 证据区域 | “引用 bounding box” | 源页面上定位答案片段的 bbox |
| OCR 回退 | “公式通道” | 在视觉之外、为公式或表格密集页面配套使用的文本流水线 |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 晚交互文档检索参考实现
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 可直接用于生产的 checkpoint
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa multi-vector 教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant multi-vector 支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 另一种索引方案
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 另一种托管索引方案
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退
