# Jupyter Notebook 最佳实践

> 交互式探索非常适合实验，但生产代码需要严谨性。掌握 Notebook 技巧、魔法命令以及重构流程。

**Type:** 构建
**Languages:** Python, Jupyter
**Prerequisites:** 无
**Time:** ~35 分钟

## 学习目标

- 安装并启动 JupyterLab、Jupyter Notebook 或带有 Jupyter 扩展的 VS Code
- 使用魔法命令 (`%timeit`, `%%time`, `%matplotlib inline`) 进行基准测试和内联可视化
- 区分何时使用 Notebook 和脚本，并应用“在 Notebook 中探索，在脚本中部署”的工作流程
- 识别并避免常见的 Notebook 陷阱：执行顺序错误、隐藏状态和内存泄漏

## 问题

每篇 AI 论文、教程和 Kaggle 比赛都使用 Jupyter Notebook。它们允许你分段运行代码、内联查看输出、将代码与解释混合，并快速迭代。如果你尝试在没有 Notebook 的情况下学习 AI，那就相当于在没有草稿纸的情况下做数学作业。

但 Notebook 确实存在真正的陷阱。人们用它们做各种事情，包括它们并不擅长的事情。了解何时使用 Notebook 以及何时使用脚本，将帮助你避免之后的调试噩梦。

## 概念

Notebook 是一个单元格列表。每个单元格要么是代码，要么是文本。```mermaid
graph TD
    A["**Markdown Cell**\n# My Experiment\nTesting learning rate 0.01"] --> B["**Code Cell** ► Run\nmodel.fit(X, y, lr=0.01)\n---\nOutput: loss = 0.342"]
    B --> C["**Code Cell** ► Run\nplt.plot(losses)\n---\nOutput: inline plot"]
```内核是一个在后台运行的 Python 进程。当你运行一个单元格时，它会将代码发送给内核，内核执行代码并将结果返回。所有单元格共享同一个内核，因此变量在单元格之间是持久的。```mermaid
graph LR
    A[Notebook UI] <--> B[Kernel\nPython process]
    B --> C[Keeps variables in memory]
    B --> D[Runs cells in whatever order you click]
    B --> E[Dies when you restart it]
```“无论你点击什么顺序”的部分既是超能力，也是脚枪。

## 构建它

### 第一步：选择你的界面

三个选项，一个格式：

| 界面       | 安装方法                 | 最适合场景                             |
|-----------|---------|----------|
| JupyterLab | `pip install jupyterlab` 然后 `jupyter lab` | 完整的 IDE 体验，多个标签页，文件浏览器，终端 |
| Jupyter Notebook | `pip install notebook` 然后 `jupyter notebook` | 简单，轻量，一次一个笔记本 |
| VS Code | 安装 "Jupyter" 扩展 | 已经在你的编辑器中，Git 集成，调试 |

所有三个界面都读写相同的 `.ipynb` 文件。选择你喜欢的即可。在人工智能工作中，JupyterLab 是最常用的。```bash
pip install jupyterlab
jupyter lab
```### 步骤 2：重要的键盘快捷键

你有这两种模式。按下 `Escape` 进入命令模式（左侧蓝色条），按下 `Enter` 进入编辑模式（绿色条）。

**命令模式（最常用）：**

| 键 | 动作 |
|-----|--------|
| `Shift+Enter` | 运行单元格，并移动到下一个 |
| `A` | 在上方插入单元格 |
| `B` | 在下方插入单元格 |
| `DD` | 删除单元格 |
| `M` | 转换为 Markdown |
| `Y` | 转换为代码 |
| `Z` | 撤销单元格操作 |
| `Ctrl+Shift+H` | 显示所有快捷键 |

**编辑模式：**

| 键 | 动作 |
|-----|--------|
| `Tab` | 自动补全 |
| `Shift+Tab` | 显示函数签名 |
| `Ctrl+/` | 切换注释 |

`Shift+Enter` 是你每天会使用上千次的快捷键。先学习它。

### 步骤 3：单元格类型

**代码单元格** 运行 Python 并显示输出：```python
import numpy as np
data = np.random.randn(1000)
data.mean(), data.std()
```输出：`(0.0032, 0.9987)`

**Markdown单元格**渲染格式化文本。使用它们来记录你正在做什么以及原因。支持标题、加粗、斜体、LaTeX数学公式（`$E = mc^2$`）、表格和图片。

### 第4步：魔术命令

这些不是Python。它们是Jupyter特定的命令，以`%`（行魔术）或`%%`（单元格魔术）开头。

**计时你的代码：**

 /no_think

<>

输出：`(0.0032, 0.9987)`

**Markdown单元格**渲染格式化文本。使用它们来记录你正在做什么以及原因。支持标题、加粗、斜体、LaTeX数学公式（`$E = mc^2$`）、表格和图片。

### 第4步：魔术命令

这些不是Python。它们是Jupyter特定的命令，以`%`（行魔术）或`%%`（单元格魔术）开头。

**计时你的代码：**```python
%timeit np.random.randn(10000)
```Output: `45.2 us +/- 1.3 us per loop````python
%%time
model.fit(X_train, y_train, epochs=10)
```输出：`Wall time: 2.34 s`

`%timeit` 运行代码多次并取平均值。`%%time` 运行一次。对微基准测试使用 `%timeit`，对训练运行使用 `%%time`。

**启用内联图表：**```python
%matplotlib inline
```每个 `plt.plot()` 或 `plt.show()` 现在都可以直接在笔记本中渲染。

**无需离开笔记本即可安装包：**```python
!pip install scikit-learn
````!` 前缀可以运行任何 shell 命令。

**检查环境变量：**```python
%env CUDA_VISIBLE_DEVICES
```### 步骤 5：内联显示丰富的输出

Notebook 会自动显示单元格中的最后一个表达式。但你可以进行控制：```python
import pandas as pd

df = pd.DataFrame({
    "model": ["Linear", "Random Forest", "Neural Net"],
    "accuracy": [0.72, 0.89, 0.94],
    "training_time": [0.1, 2.3, 45.6]
})
df
```这会生成一个格式化的 HTML 表格，而不是文本转储。图表也是如此：```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Inline Plot")
plt.show()
```图表显示在单元格的正下方。这就是为什么笔记本在人工智能工作中占主导地位。你可以同时看到数据、图表和代码。

对于图像：```python
from IPython.display import Image, display
display(Image(filename="architecture.png"))
```### 第6步：Google Colab

Colab 是一个免费的云端 Jupyter 笔记本。它为你提供 GPU、预安装的库以及 Google Drive 集成。无需任何设置。

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 上传本课程中的任何 `.ipynb` 文件
3. 运行时 > 更改运行时类型 > T4 GPU（免费）

Colab 与本地 Jupyter 的区别：
- 会话之间文件不会保留（需保存到 Drive 或下载）
- 预安装：numpy、pandas、matplotlib、torch、tensorflow、sklearn
- 使用 `from google.colab import files` 上传/下载文件
- 使用 `from google.colab import drive; drive.mount('/content/drive')` 进行持久存储
- 会话在90分钟无活动后超时（免费版本）

## 使用它

### 笔记本 vs 脚本：何时使用哪种

| 使用笔记本 | 使用脚本 |
|-------------------|-----------------|
| 探索数据集 | 训练流水线 |
| 模型原型设计 | 可重用的工具 |
| 可视化结果 | 任何包含 `if __name__` 的内容 |
| 解释你的工作 | 按计划运行的代码 |
| 快速实验 | 生产代码 |
| 课程练习 | 包和库 |

规则：**在笔记本中探索，在脚本中部署**。

人工智能中常见的工作流程：
1. 在笔记本中探索数据
2. 在笔记本中对模型进行原型设计
3. 一旦它运行正常，将代码移动到 `.py` 文件中
4. 将这些 `.py` 文件重新导入笔记本中，以便进行进一步的实验

### 常见陷阱

**执行顺序错误。** 你先运行单元格5，然后单元格2，再单元格7。笔记本在你的机器上运行正常，但其他人从上到下运行时会出现问题。解决办法：内核 > 重启并运行全部，然后再分享。

**隐藏状态。** 你删除了一个单元格，但它创建的变量仍然在内存中。笔记本看起来干净，但依赖于一个“幽灵”单元格。解决办法：定期重启内核。

**内存泄漏。** 加载一个4GB的数据集，训练模型，加载另一个数据集。内存没有释放。解决办法：使用 `del variable_name` 和 `gc.collect()`，或重启内核。

## 部署它

本课将生成：
- `outputs/prompt-notebook-helper.md` 用于调试笔记本问题

## 练习

1. 打开 JupyterLab，创建一个笔记本，并使用 `%timeit` 来比较列表推导式和 numpy 在创建一个包含100,000个随机数的数组时的性能
2. 创建一个包含markdown和代码单元格的笔记本，加载一个CSV文件，显示一个数据框，并绘制一个图表。然后运行内核 > 重启并运行全部，以验证它是否能够从上到下正常运行
3. 从 `code/notebook_tips.py` 复制代码，粘贴到 Colab 笔记本中，并使用免费的 GPU 运行它

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 内核 | “运行我代码的东西” | 一个独立的 Python 进程，用于执行单元格并保持变量在内存中 |
| 单元格 | “一个代码块” | 笔记本中一个可独立运行的单元，可以是代码或markdown |
| 魔法命令 | “Jupyter 技巧” | 以 `%` 或 `%%` 为前缀的特殊命令，用于控制笔记本环境 |
| `.ipynb` | “笔记本文件” | 包含单元格、输出和元数据的 JSON 文件。代表 IPython 笔记本 |

## 进一步阅读

- [JupyterLab 文档](https://jupyterlab.readthedocs.io/) 了解完整功能集
- [Google Colab 常见问题](https://research.google.com/colaboratory/faq.html) 了解 Colab 的特定限制和功能
- [28 个 Jupyter 笔记本技巧](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) 了解高级用户的快捷键
