# 编辑器配置与 AI 开发工具

> 把你的编辑器打造为 AI 开发利器。配置 IDE 解释器、Ruff 格式化、类型检查与 AI 编码助手。

**Type:** 构建
**Languages:** JSON, Python
**Prerequisites:** 无
**Time:** ~30 分钟

## 学习目标

- 安装配备 Python、Jupyter、代码检查和远程 SSH 所需核心扩展的 VS Code
- 为 AI 工作流配置保存时自动格式化、类型检查和笔记本输出滚动功能
- 设置 Remote SSH，使远程 GPU 机器上的代码编辑和调试如同本地操作
- 评估编辑器替代方案（Cursor、Windsurf、Neovim）及其在 AI 工作中的取舍

## The Problem

你将在编辑器中花费数千小时编写 Python、运行笔记本、调试训练循环并通过 SSH 连接到 GPU 机器。配置不当的编辑器会使每次会话都充满摩擦：没有自动补全、没有类型提示、没有内联错误、需要手动格式化，以及笨拙的终端工作流程。

正确的配置只需 20 分钟。跳过它，每天将损失 20 分钟。

## The Concept

AI 工程编辑器的配置需要以下五个要素：

 /think```mermaid
graph TD
    L5["5. Remote Development<br/>SSH into GPU boxes, cloud VMs"] --> L4
    L4["4. Terminal Integration<br/>Run scripts, debug, monitor GPU"] --> L3
    L3["3. AI-Specific Settings<br/>Auto-format, type checking, rulers"] --> L2
    L2["2. Extensions<br/>Python, Jupyter, Pylance, GitLens"] --> L1
    L1["1. Base Editor<br/>VS Code — free, extensible, universal"]
```## 构建它

### 步骤1：安装VS Code

VS Code是推荐的编辑器。它是免费的，可在所有操作系统上运行，具有顶级的Jupyter笔记本支持，扩展生态系统涵盖了AI工作所需的一切。

从 [code.visualstudio.com](https://code.visualstudio.com/) 下载它。

从终端验证：

 /think```bash
code --version
```如果在 macOS 上未找到 `code`，请打开 VS Code，按下 `Cmd+Shift+P`，输入 "Shell Command"，然后选择 "Install 'code' command in PATH"。

### 步骤2：安装必要的扩展

在 VS Code 中打开集成终端 (`` Ctrl+` `` 在所有平台上)，并安装对 AI 工作至关重要的扩展：

 /think```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```每个扩展的功能：

| 扩展 | 原因 |
|------|-----|
| Python | 语言支持，虚拟环境检测，运行/调试 |
| Pylance | 快速类型检查，自动补全，导入解析 |
| Jupyter | 在 VS Code 中运行笔记本，变量资源管理器 |
| GitLens | 查看谁修改了什么，内联 Git 责任追溯 |
| Remote SSH | 将远程 GPU 机器上的文件夹作为本地打开 |
| Debugpy | Python 的逐步调试 |
| Black Formatter | 保存时自动格式化，保持风格一致 |
| Ruff | 快速代码检查，捕捉常见错误 |

本课的 `code/.vscode/extensions.json` 文件包含完整的推荐列表。当你打开项目文件夹时，VS Code 会提示你安装它们。

### 步骤3：配置设置

从本课的 `code/.vscode/settings.json` 复制设置，或通过 `Settings > Open Settings (JSON)` 手动应用它们。

AI 工作的关键设置：```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```为何这些事项重要：

- **基本类型检查**：在运行之前捕获错误的参数类型。节省调试张量形状不匹配和错误API参数的时间。
- **保存时格式化**：再也不用担心格式问题。Black会处理。
- **88和120处的标尺**：Black在88处换行。120标记显示文档字符串和注释是否过长。
- **笔记本输出滚动**：训练循环会打印数千行。没有滚动功能时，输出面板会变得混乱。
- **自动保存**：你会忘记保存。训练脚本会运行过时的代码。自动保存可以防止这种情况。

### 第4步：终端集成

VS Code的集成终端是运行训练脚本、监控GPU和管理环境的地方。

正确设置：

 /think```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```有用的快捷键：

| 操作 | macOS | Linux/Windows |
|------|-------|--------------|
| 切换终端 | `` Ctrl+` `` | `` Ctrl+` `` |
| 新建终端 | `` Ctrl+Shift+` `` | `` Ctrl+Shift+` `` |
| 分屏终端 | `Cmd+\` | `Ctrl+Shift+5` |

分屏终端很有用：一个用于运行脚本，一个用于通过 `nvidia-smi -l 1` 或 `watch -n 1 nvidia-smi` 监控 GPU。

### 步骤5：远程开发（通过SSH连接到GPU服务器）

这是AI工作中最重要的扩展功能。你将在远程机器（云虚拟机、实验室服务器、Lambda、Vast.ai）上运行训练。远程SSH让你可以打开远程文件系统、编辑文件、运行终端并进行调试，就像所有内容都在本地一样。

设置步骤：

1. 安装Remote SSH扩展（在步骤2中已完成）。
2. 按下 `Ctrl+Shift+P`（或 `Cmd+Shift+P`），输入 "Remote-SSH: Connect to Host"。
3. 输入 `user@your-gpu-box-ip`。
4. VS Code 会自动在远程机器上安装其服务端组件。

要实现无密码访问，请设置SSH密钥：

 /home/username/.ssh/id_rsa```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@your-gpu-box-ip
```

Add the host to `~/.ssh/config` for convenience:

```
Host gpu-box
    HostName 203.0.113.50
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```现在 `Remote-SSH: Connect to Host > gpu-box` 可以立即连接。

## 替代方案

### Cursor

[cursor.com](https://cursor.com) 是一个带有内置 AI 代码生成功能的 VS Code 分支版本。它使用相同的扩展生态系统和设置格式。如果你使用 Cursor，本课的所有内容仍然适用。导入相同的 `settings.json` 和 `extensions.json`。

### Windsurf

[windsurf.com](https://windsurf.com) 是另一个以 AI 为核心的 VS Code 分支版本。同样的情况：相同的扩展、相同的设置格式、相同的远程 SSH 支持。

### Vim/Neovim

如果你已经使用 Vim 或 Neovim 并且能高效工作，可以继续使用。AI Python 工作的最低配置：

- **pyright** 或 **pylsp** 用于类型检查（通过 Mason 或手动安装）
- **nvim-lspconfig** 用于语言服务器集成
- **jupyter-vim** 或 **molten-nvim** 用于类似笔记本的执行
- **telescope.nvim** 用于文件/符号搜索
- **none-ls.nvim** 配合 black 和 ruff 用于格式化/代码检查

如果你尚未使用 Vim，现在不要开始。学习曲线将与学习 AI 工程竞争。使用 VS Code。

## 使用方法

通过此设置，你的日常工作流程如下：

1. 在 VS Code 中打开项目文件夹（或通过远程 SSH 连接到 GPU 服务器）。
2. 在编辑器中使用自动补全、类型提示和内联错误编写 Python。
3. 使用 Jupyter 扩展在内联运行 Jupyter 笔记本。
4. 使用集成终端运行训练脚本、`uv pip install` 和 GPU 监控。
5. 在提交前使用 GitLens 查看更改内容。

## 练习

1. 安装 VS Code 和第 2 步中列出的所有扩展
2. 将本课的 `settings.json` 复制到你的 VS Code 配置中
3. 打开一个 Python 文件并验证 Pylance 是否显示类型提示并保存时使用 Black 格式化
4. 如果你有访问远程机器的权限，请设置远程 SSH 并在远程机器上打开一个文件夹

## 术语表

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| LSP | "自动补全引擎" | 语言服务器协议：一种标准，允许编辑器从特定语言的服务器获取类型信息、补全和诊断 |
| Pylance | "Python 插件" | 微软的 Python 语言服务器，使用 Pyright 进行类型检查和 IntelliSense |
| Remote SSH | "在服务器上工作" | VS Code 扩展，它在远程机器上运行一个轻量级服务器，并将 UI 流式传输到本地编辑器 |
| 保存时格式化 | "自动美化" | 每次保存时，编辑器运行格式化工具（Black, Ruff），确保代码风格始终一致 |
