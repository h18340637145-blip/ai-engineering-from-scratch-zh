# AI 数据管理与存储架构

> 海量数据需要高效的存储与传输。掌握 Parquet 格式、Hugging Face 数据集流式读取与去重技术。

**Type:** 构建
**Language:** Python
**Prerequisites:** 无
**Time:** ~40 分钟

## 学习目标

- Load, stream, and cache datasets using the Hugging Face `datasets` library
- Convert between CSV, JSON, Parquet, and Arrow formats and explain their tradeoffs
- Create reproducible train/validation/test splits with fixed random seeds
- Manage large model and dataset files using `.gitignore`, Git LFS, or DVC

## The Problem

Every AI project starts with data. You need to find datasets, download them, convert between formats, split them for training and evaluation, and version them so experiments are reproducible. Doing this manually every time is slow and error-prone. You need a repeatable workflow.

## The Concept

 /think

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets library"]
    B --> C["Load / Stream"]
    C --> D["Local Cache<br/>~/.cache/huggingface/"]
    B --> E["Format Conversion<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["Data Splits<br/>train / val / test"]
    F --> G["Your Training Pipeline"]
```Hugging Face `datasets` 库是加载 AI 工作数据的标准方式。它开箱即用，可处理下载、缓存、格式转换和流式处理。

## 构建它

### 步骤1：安装 datasets 库

 /think

```bash
pip install datasets huggingface_hub
```

### Step 2: Load a dataset

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print(dataset["train"][0])
```

这会下载IMDB电影评论数据集。首次下载后，会从缓存路径`~/.cache/huggingface/datasets/`加载。

### 步骤3：流式处理大型数据集

某些数据集太大无法完全存储在磁盘上。流式处理可以逐行加载数据，而无需下载完整文件。

```python
dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

流式处理为你提供一个 `IterableDataset`。你可以在数据到达时逐行处理。内存使用量与数据集大小无关，始终保持恒定。

### 步骤4：数据集格式

`datasets` 库内部使用 Apache Arrow。根据流水线的需求，可以转换为其他格式。

 /think

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

格式对比：

| 格式 | 大小 | 读取速度 | 最适合用于 |
|------|------|-----------|-----|
| CSV | 大 | 慢 | 人类可读性，电子表格 |
| JSON | 大 | 慢 | API，嵌套数据 |
| Parquet | 小 | 快 | 分析，列式查询 |
| Arrow | 小 | 最快 | 内存处理（`datasets` 内部使用的格式） |

对于 AI 任务，Parquet 是最佳的存储格式。Arrow 是内存中处理的格式。CSV 和 JSON 用于数据交换。

### 步骤 5：数据划分

每个机器学习项目都需要三个数据划分：

- **训练集**：模型从此学习（通常为 80%）
- **验证集**：训练过程中检查进度（通常为 10%）
- **测试集**：训练完成后进行最终评估（通常为 10%）

一些数据集已经预先划分好了。如果没有，自行划分：

 /

```python
dataset = load_dataset("stanfordnlp/imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

始终设置种子以确保可重复性。相同的种子每次都会生成相同的分割。

### 第6步：下载并缓存模型

模型是大型文件。`huggingface_hub`库负责处理下载和缓存。

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

模型缓存至 `~/.cache/huggingface/hub/`。一旦下载完成，后续运行时会立即加载。

### 第7步：处理大文件

模型权重和大型数据集不应纳入git。三个选项：

**选项A：.gitignore（最简单）**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**Option B: Git LFS (track large files in git)**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```Git LFS 在你的仓库中存储指针，而实际文件则存储在单独的服务器上。GitHub 为你提供 1 GB 的免费空间。

**选项 C: DVC (data version control)**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```DVC 创建小型的 `.dvc` 文件，这些文件指向你的数据。数据本身存储在 S3、GCS 或其他远程存储后端中。

| 方法 | 复杂度 | 最适合 |
|------|------|------|
| .gitignore | 低 | 个人项目、可以重新获取的下载数据 |
| Git LFS | 中等 | 通过 git 共享模型权重的团队 |
| DVC | 高 | 可复现的实验、大型数据集、团队协作 |

对于本课程，`.gitignore` 已经足够。当需要在不同机器上复现精确的实验时，请使用 DVC。

### 步骤 8：存储模式

**本地存储** 适用于约 10 GB 以下的数据集。HF 缓存会自动处理这种情况。

**云存储** 用于更大的数据集或需要跨机器共享的数据：

 /think

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC integrates with S3 and GCS directly:

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

对于本课程，本地存储已足够。当在远程GPU实例上进行微调时，云存储才变得相关。

## 本课程使用的数据集

| 数据集 | 课程 | 大小 | 所学内容 |
|-------|-----|-----|--------|
| IMDB | 分词、分类 | 84 MB | 文本分类基础 |
| WikiText | 语言建模 | 181 MB | 下一词预测 |
| SQuAD | 问答系统 | 35 MB | 问答、跨度提取 |
| Common Crawl (子集) | 嵌入 | 可变 | 大规模文本处理 |
| MNIST | 视觉基础 | 21 MB | 图像分类基础 |
| COCO (子集) | 多模态 | 可变 | 图像-文本对 |

你现在不需要下载所有这些数据集。每节课会指定其所需内容。

## 使用方法

运行实用脚本以验证所有内容是否正常工作：

 /think

```bash
python code/data_utils.py
```

这会下载一个小型数据集，对其进行转换、拆分，并打印摘要。

## Ship It

本课生成以下内容：
- `code/data_utils.py` - 可重复使用的数据加载和缓存工具
- `outputs/prompt-data-helper.md` - 用于查找任务合适数据集的提示

## 练习

1. 使用 `mrpc` 配置加载 `glue` 数据集并检查前5个示例
2. 流式传输 `c4` 数据集并计算10秒内能处理多少示例
3. 将数据集转换为Parquet格式并将其文件大小与CSV进行比较
4. 使用固定种子创建70/15/15的训练/验证/测试拆分并验证各部分大小

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 数据集拆分 | "训练数据" | ML生命周期不同阶段使用的命名子集（train/val/test） |
| 流式处理 | "延迟加载" | 不下载完整数据集而从远程源逐行处理数据 |
| Parquet | "压缩的CSV" | 为分析查询和存储效率优化的列式文件格式 |
| Arrow | "快速数据框" | datasets库内部使用的内存列式格式，支持零拷贝读取 |
| Git LFS | "大文件的Git" | 一个扩展，将大文件存储在git仓库之外，同时在版本控制中保留指针 |
| DVC | "数据的Git" | 与云存储集成的数据集和模型版本控制系统 |
| 缓存 | "已下载" | 默认存储在 ~/.cache/huggingface/ 的先前获取数据的本地副本 |
