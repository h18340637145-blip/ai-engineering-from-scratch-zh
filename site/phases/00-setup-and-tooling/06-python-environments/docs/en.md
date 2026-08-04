# Python 环境隔离与包管理

> 彻底解决‘在我电脑上跑得好好的’难题。深度探究 `sys.path`、Wheel 编译、依赖锁定与隔离机制。

**Type:** 学习
**Languages:** Python, Bash
**Prerequisites:** 开发环境搭建
**Time:** ~30 分钟

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境
- 编写包含可选依赖组的 `pyproject.toml` 并生成锁文件以实现可复现性
- 诊断并修复常见问题：全局安装、pip/conda 混用、CUDA 版本不匹配
- 为存在依赖冲突的项目实现按阶段划分的环境策略

## The Problem

你为一个微调项目安装了 PyTorch 2.4。下周另一个项目需要 PyTorch 2.1，因为其 CUDA 构建被固定。你全局升级后，第一个项目崩溃。你降级后，第二个项目崩溃。

这就是依赖地狱。在 AI/ML 工作中这经常发生，因为：

- PyTorch、JAX 和 TensorFlow 各自携带自己的 CUDA 绑定
- 模型库会固定特定框架版本
- 全局 `pip install` 会覆盖之前安装的内容
- CUDA 11.8 构建无法与 CUDA 12.x 驱动配合（反之亦然）

解决方案：每个项目都拥有自己的隔离环境和专属包。

## The Concept```mermaid
graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end
```## 构建它

### 选项 1: uv venv（推荐）

`uv` 是最快的 Python 包管理器（比 pip 快 10-100 倍）。它在一个工具中处理虚拟环境、Python 版本和依赖解析。```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```安装软件包：```bash
uv pip install torch numpy
```使用 `pyproject.toml` 一步创建项目：```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```### 选项 2: venv (内置)

如果你无法安装 `uv`，Python 自带了 `venv`:```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy
```比 `uv` 慢，但可以在所有安装了 Python 的地方运行。

### 选项3：conda（当您需要时）

Conda 管理非 Python 依赖项，如 CUDA 工具包、cuDNN 和 C 库。在以下情况下使用它：

- 您需要特定版本的 CUDA 工具包，而无需在系统范围内安装
- 您在共享集群上，无法安装系统包
- 某个库的安装说明中提到 "use conda"```bash
# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```一个规则：如果你使用 conda 来管理一个环境，那么该环境中的所有包都必须使用 conda 来安装。将 `pip install` 混入 conda 环境中会导致难以调试的依赖冲突。

### 本课程策略：分阶段策略

你可以为整个课程创建一个环境。不要这样做。不同的阶段需要不同的（有时是冲突的）依赖项。

策略：```
ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed
````code/env_setup.sh` 中的脚本为本课程创建了基础环境。

## pyproject.toml 基础

每个 Python 项目都应该有一个 `pyproject.toml`。它在一个文件中替换了 `setup.py`、`setup.cfg` 和 `requirements.txt`。```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```然后安装：```bash
uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything
```## 锁定文件

锁定文件将每个依赖项（包括传递依赖）固定到确切的版本。这保证了可重复性：从锁定文件安装的任何人都会获得完全相同的软件包。```bash
# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```将 lockfile 提交到 git。当有人克隆仓库时，他们会从 lockfile 安装并获取相同的版本。

## 常见错误

### 1. 全局安装 /no_think

<>

将 lockfile 提交到 git。当有人克隆仓库时，他们会从 lockfile 安装并获取相同的版本。

## 常见错误

### 1. 全局安装```bash
pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment
```检查你的软件包去向：

 /no_think

<>

检查你的软件包去向：

 /no_think```bash
which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip
```### 2. 混合使用 pip 和 conda```bash
conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything
```如果必须在 conda 中使用 pip（某些包只能通过 pip 安装），请先安装所有 conda 包，最后再安装 pip 包。

### 3. 忘记激活环境 /no_think

<>

如果必须在 conda 中使用 pip（某些包只能通过 pip 安装），请先安装所有 conda 包，最后再安装 pip 包。

### 3. 忘记激活环境```bash
python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found
```你的 shell 提示符应该显示环境名称：```
(.venv) $ python train.py
```### 4. 将 .venv 提交到 git```bash
echo ".venv/" >> .gitignore
```虚拟环境的大小为200MB至2GB。它们是本地的，无法在不同机器之间传输。请提交 `pyproject.toml` 和锁文件（lockfile）。

### 5. CUDA版本不匹配```bash
nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.
```## 使用它

运行设置脚本以创建你的课程环境：

 /no_think

<>

## 使用它

运行设置脚本来创建你的课程环境：

 /no_think```bash
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```这会在仓库根目录下创建一个 `.venv`，其中已安装并验证了核心依赖项。

## 练习

1. 运行 `env_setup.sh` 并验证所有检查项是否通过
2. 创建第二个虚拟环境，在其中安装不同版本的 numpy，并确认两个环境是隔离的
3. 为一个需要同时使用 PyTorch 和 Anthropic SDK 的项目编写 `pyproject.toml`
4. 故意全局安装一个包（不激活虚拟环境），注意它被安装的位置，然后卸载它

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 虚拟环境 | "一个 venv" | 包含独立 Python 解释器和包的隔离目录，与系统 Python 分离 |
| 锁文件 | "固定依赖" | 列出每个包及其确切版本的文件，确保不同机器上的安装完全一致 |
| pyproject.toml | "新的 setup.py" | 标准的 Python 项目配置文件，替代 setup.py/setup.cfg/requirements.txt |
| 传递依赖 | "依赖的依赖" | 包 B 依赖 C；如果你安装了依赖 B 的 A，那么 C 是 A 的传递依赖 |
| CUDA 不匹配 | "我的 GPU 不工作" | PyTorch 编译时使用的 CUDA 版本与你的 GPU 驱动支持的版本不一致 |
